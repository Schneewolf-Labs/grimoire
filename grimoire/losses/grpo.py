import torch

from .online import OnlineMethod
from .utils import (
    DEFAULT_FUSED_CHUNK_SIZE,
    forward_per_token_logps,
    masked_avg_logps,
)


class GRPOMethod(OnlineMethod):
    """GRPO (Group Relative Policy Optimization) — an online RL method.

    Paper: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
           arXiv:2402.03300

    Two-phase online method (see ``OnlineMethod``):

    1. ``rollout(model, batch)`` — generates G completions per prompt, scores
       them with ``reward_fn``, and computes group-relative advantages (plus
       frozen-reference log-probs for the KL term). Runs under ``no_grad``.
    2. ``__call__(model, experience)`` — the loss phase, a pure function of
       the experience batch.

    Loss = -mean(advantages * min(ratio, clipped_ratio)) + beta * KL

    where ratio = pi(y|x) / pi_old(y|x) and advantages are normalized within
    each group of G completions. Grimoire performs a single policy update per
    generation step (mu=1 in the paper), so pi_old == pi and the ratio is
    identically 1 — the clipping never activates and the objective reduces to
    REINFORCE with group-normalized advantages. The ratio form is kept for
    parity with the paper's objective.

    The KL penalty is estimated against a frozen reference policy using the
    non-negative k3 estimator: exp(ref - pi) - (ref - pi) - 1. The reference
    is ``ref_model`` if provided, else the base model via ``disable_adapter()``
    for PEFT models. If neither is available, the KL term is skipped (with a
    warning) — pass beta=0 to silence it.

    Two bias fixes from "Understanding R1-Zero-Like Training" (Dr. GRPO,
    arXiv:2503.20783) are available as options:

    - ``scale_rewards=False`` drops the per-group std normalization, which
      otherwise up-weights groups whose rewards barely differ (question-level
      difficulty bias).
    - ``loss_type="dr_grpo"`` aggregates per-token log-probs as
      sum / max_new_tokens instead of the per-sequence mean, removing the
      length bias that makes long wrong answers cheaper than short ones.

    ``dynamic_sampling=True`` (from DAPO, arXiv:2503.14476) excludes
    zero-variance groups — where every completion got the same reward, so
    every advantage is exactly 0 — from the loss average, so uninformative
    groups don't dilute the gradient of the informative ones.
    """

    def __init__(
        self,
        reward_fn,
        tokenizer,
        num_generations=4,
        beta=0.04,
        epsilon=0.2,
        max_new_tokens=512,
        temperature=1.0,
        label_pad_token_id=-100,
        ref_model=None,
        scale_rewards=True,
        loss_type="grpo",
        dynamic_sampling=False,
        fused=True,
        fused_chunk_size=DEFAULT_FUSED_CHUNK_SIZE,
    ):
        super().__init__(
            reward_fn=reward_fn,
            tokenizer=tokenizer,
            num_generations=num_generations,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            label_pad_token_id=label_pad_token_id,
            ref_model=ref_model,
            fused=fused,
            fused_chunk_size=fused_chunk_size,
        )
        if loss_type not in ("grpo", "dr_grpo"):
            raise ValueError(f"loss_type must be 'grpo' or 'dr_grpo', got {loss_type!r}")
        self.beta = beta
        self.epsilon = epsilon
        self.scale_rewards = scale_rewards
        self.loss_type = loss_type
        self.dynamic_sampling = dynamic_sampling

    # --- online phase ----------------------------------------------------- #

    @torch.no_grad()
    def rollout(self, model, batch):
        """Turn a prompt-only batch into a scored experience batch.

        Called by the trainer before the loss. Everything here is the online /
        environment-interaction part: generation, decoding, the reward call,
        advantage estimation, and the frozen reference pass. No gradients.

        Returns a batch dict the loss phase consumes: ``input_ids`` /
        ``attention_mask`` / ``labels`` for the prompt+completion sequences,
        per-sequence ``advantages``, optional ``ref_logps`` for the KL term,
        an optional per-sequence ``advantage_mask`` (dynamic sampling), and
        ``rollout_metrics`` (reward stats surfaced to the logger).
        """
        B = batch["input_ids"].size(0)
        G = self.num_generations

        gen = self._generate_and_score(model, batch)
        rewards = gen["rewards"]  # [B*G]

        advantages = self._compute_advantages(rewards, B, G)  # [B*G]

        # Dynamic sampling (DAPO): a group where every completion got the same
        # reward has all-zero advantages and carries no learning signal — mask
        # it out of the loss average instead of letting it dilute the batch.
        advantage_mask = None
        zero_variance_frac = 0.0
        if self.dynamic_sampling:
            grouped = rewards.view(B, G)
            informative = (grouped != grouped[:, :1]).any(dim=1)  # [B]
            advantage_mask = informative.repeat_interleave(G).float()  # [B*G]
            zero_variance_frac = 1.0 - informative.float().mean().item()

        # Frozen reference log-probs for the KL penalty (still no grad).
        ref_logps = None
        if self.beta > 0:
            ref_logps = self._reference_logps(
                model, gen["input_ids"], gen["attention_mask"], gen["labels"]
            )

        metrics = {
            "rewards_mean": rewards.mean().item(),
            "rewards_std": rewards.std().item(),
            "advantages_mean": advantages.mean().item(),
            "completion_length": float(gen["completion_length"]),
        }
        if self.dynamic_sampling:
            metrics["zero_variance_groups"] = zero_variance_frac

        return {
            "input_ids": gen["input_ids"],
            "attention_mask": gen["attention_mask"],
            "labels": gen["labels"],
            "advantages": advantages,
            "advantage_mask": advantage_mask,
            "ref_logps": ref_logps,
            "rollout_metrics": metrics,
        }

    def _compute_advantages(self, rewards, B, G):
        """Group-relative advantages: (r - mean(group)) [/ std(group)].

        The std division is skipped with ``scale_rewards=False`` (Dr. GRPO).
        Subclasses (RLOO) override this to change the baseline.
        """
        rewards_grouped = rewards.view(B, G)
        advantages = rewards_grouped - rewards_grouped.mean(dim=1, keepdim=True)
        if self.scale_rewards:
            group_std = rewards_grouped.std(dim=1, keepdim=True).clamp(min=1e-8)
            advantages = advantages / group_std
        return advantages.view(B * G)

    # --- loss phase (pure) ------------------------------------------------ #

    def __call__(self, model, batch, training=True):
        if not training:
            # Eval is not meaningful for GRPO, and eval batches are prompt-only
            # (the trainer runs no rollout there). Return zero.
            return torch.zeros((), device=batch["input_ids"].device), {}

        advantages = batch["advantages"]
        advantage_mask = batch.get("advantage_mask")
        ref_logps = batch["ref_logps"]

        # Policy log-probs (WITH grad) over the generated experience. The KL
        # term always uses the per-sequence average (matching ref_logps); the
        # policy term uses sum / max_new_tokens under dr_grpo (length-bias fix).
        per_token_logps, loss_mask = forward_per_token_logps(
            model, batch["input_ids"], batch["attention_mask"], batch["labels"],
            self.label_pad_token_id, fused=self.fused, fused_chunk_size=self.fused_chunk_size,
        )
        avg_logps = masked_avg_logps(per_token_logps, loss_mask)  # [B*G]
        if self.loss_type == "dr_grpo":
            logps = (per_token_logps * loss_mask).sum(-1) / self.max_new_tokens
        else:
            logps = avg_logps

        # Single update per generation step, so pi_old == pi; ratio == 1. The
        # ratio/clip form is kept for parity with the paper's PPO-style objective.
        old_logps = logps.detach()
        ratio = torch.exp(logps - old_logps)
        clipped = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        per_seq_objective = torch.min(
            ratio * advantages.detach(), clipped * advantages.detach()
        )

        if ref_logps is not None:
            log_ratio = ref_logps - avg_logps
            per_seq_kl = torch.exp(log_ratio) - log_ratio - 1.0  # k3 estimator
        else:
            per_seq_kl = None

        if advantage_mask is not None:
            denom = advantage_mask.sum().clamp(min=1)
            policy_loss = -(per_seq_objective * advantage_mask).sum() / denom
            kl = (per_seq_kl * advantage_mask).sum() / denom if per_seq_kl is not None else None
        else:
            policy_loss = -per_seq_objective.mean()
            kl = per_seq_kl.mean() if per_seq_kl is not None else None

        if kl is not None:
            loss = policy_loss + self.beta * kl
        else:
            kl = torch.zeros((), device=logps.device)
            loss = policy_loss

        metrics = dict(batch["rollout_metrics"])
        metrics.update(
            kl=kl.detach().item(),
            policy_loss=policy_loss.detach().item(),
            ratio_mean=ratio.detach().mean().item(),
        )
        return loss, metrics
