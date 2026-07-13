import logging

import torch

from ..data.grpo import GRPOCollator
from .utils import (
    DEFAULT_FUSED_CHUNK_SIZE,
    _disable_grad_checkpointing,
    _unwrap_model,
    forward_per_token_logps,
    masked_avg_logps,
)

logger = logging.getLogger(__name__)


class GRPOMethod:
    """GRPO (Group Relative Policy Optimization) — an online RL method.

    Paper: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
           arXiv:2402.03300

    Unlike the offline losses, GRPO is a two-phase online method:

    1. ``rollout(model, batch)`` — the ONLINE phase. Generates G completions per
       prompt, scores them with ``reward_fn``, and computes group-relative
       advantages (plus frozen-reference log-probs for the KL term). Runs
       entirely under ``torch.no_grad()``. The trainer calls this before the
       loss whenever a method exposes it; offline losses simply don't.
    2. ``__call__(model, experience)`` — the loss phase. A pure function of the
       experience batch, structurally identical to the offline losses: forward
       the policy, take response-token log-probs, return a scalar. All the
       generation / decoding / environment interaction lives in ``rollout``, not
       here — the loss never touches the tokenizer or the reward function.

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

    Note: the rollout phase calls ``model.generate()``, which needs full weight
    access — so GRPO requires ZeRO-2 or lower (or FSDP), not ZeRO-3. The trainer
    enforces this before training starts.
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
        fused=True,
        fused_chunk_size=DEFAULT_FUSED_CHUNK_SIZE,
    ):
        if ref_model is not None and ref_model.training:
            raise ValueError("ref_model must be in eval mode (call ref_model.eval() first)")
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        self.num_generations = num_generations
        self.beta = beta
        self.epsilon = epsilon
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.label_pad_token_id = label_pad_token_id
        self.ref_model = ref_model
        self.fused = fused
        self.fused_chunk_size = fused_chunk_size
        self._pad_token_id = 0
        self._warned_no_ref = False

    def create_collator(self, pad_token_id):
        self._pad_token_id = pad_token_id
        return GRPOCollator(pad_token_id=pad_token_id)

    def _avg_logps(self, model, input_ids, attention_mask, labels):
        per_token_logps, loss_mask = forward_per_token_logps(
            model, input_ids, attention_mask, labels, self.label_pad_token_id,
            fused=self.fused, fused_chunk_size=self.fused_chunk_size,
        )
        return masked_avg_logps(per_token_logps, loss_mask)

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
        and ``rollout_metrics`` (reward stats surfaced to the logger).
        """
        input_ids = batch["input_ids"]  # [B, prompt_len], left-padded
        attention_mask = batch["attention_mask"]  # [B, prompt_len]
        B = input_ids.size(0)
        G = self.num_generations
        prompt_len = input_ids.size(1)

        # 1. Generate G completions per prompt. generate() needs the unwrapped
        # model (DDP/FSDP wrappers don't forward .generate); the loss phase gets
        # the wrapped model back from the trainer for correct gradient sync.
        repeated_ids = input_ids.repeat_interleave(G, dim=0)  # [B*G, prompt_len]
        repeated_mask = attention_mask.repeat_interleave(G, dim=0)
        gen_model = _unwrap_model(model)

        with _disable_grad_checkpointing(model):
            was_training = gen_model.training
            gen_model.eval()
            generated = gen_model.generate(
                input_ids=repeated_ids,
                attention_mask=repeated_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self._pad_token_id,
                # The trainer sets model.config.use_cache = False for training;
                # be explicit or generation runs cache-less (quadratic decode).
                use_cache=True,
            )  # [B*G, prompt_len + completion_len]
            if was_training:
                gen_model.train()

        completion_ids = generated[:, prompt_len:]
        completion_length = completion_ids.size(1)

        # Mask completion tokens after the first EOS (generate() pads there, and
        # pad_token_id may collide with real token IDs, e.g. pad == eos).
        completion_mask = self._completion_mask(completion_ids)
        gen_attention_mask = torch.cat([repeated_mask, completion_mask], dim=1)

        # Labels: mask the prompt region and padding after EOS.
        gen_labels = generated.clone()
        gen_labels[:, :prompt_len] = self.label_pad_token_id
        gen_labels[:, prompt_len:] = torch.where(
            completion_mask.bool(), completion_ids, self.label_pad_token_id
        )

        # 2. Score completions with the reward function (the environment).
        prompt_texts = self.tokenizer.batch_decode(repeated_ids, skip_special_tokens=True)
        completion_texts = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        del repeated_ids, completion_ids
        rewards = self.reward_fn(prompt_texts, completion_texts)  # list[float] or tensor
        del prompt_texts, completion_texts
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, dtype=torch.float32, device=input_ids.device)
        elif rewards.device != input_ids.device:
            rewards = rewards.to(input_ids.device)  # [B*G]

        # 3. Group-relative advantages.
        rewards_grouped = rewards.view(B, G)
        group_mean = rewards_grouped.mean(dim=1, keepdim=True)
        group_std = rewards_grouped.std(dim=1, keepdim=True).clamp(min=1e-8)
        advantages = ((rewards_grouped - group_mean) / group_std).view(B * G)  # [B*G]

        # 4. Frozen reference log-probs for the KL penalty (still no grad).
        ref_logps = None
        if self.beta > 0:
            ref_logps = self._reference_logps(model, generated, gen_attention_mask, gen_labels)

        return {
            "input_ids": generated,
            "attention_mask": gen_attention_mask,
            "labels": gen_labels,
            "advantages": advantages,
            "ref_logps": ref_logps,
            "rollout_metrics": {
                "rewards_mean": rewards.mean().item(),
                "rewards_std": rewards.std().item(),
                "advantages_mean": advantages.mean().item(),
                "completion_length": float(completion_length),
            },
        }

    # --- loss phase (pure) ------------------------------------------------ #

    def __call__(self, model, batch, training=True):
        if not training:
            # Eval is not meaningful for GRPO, and eval batches are prompt-only
            # (the trainer runs no rollout there). Return zero.
            return torch.zeros((), device=batch["input_ids"].device), {}

        advantages = batch["advantages"]
        ref_logps = batch["ref_logps"]

        # Policy log-probs (WITH grad) over the generated experience.
        logps = self._avg_logps(
            model, batch["input_ids"], batch["attention_mask"], batch["labels"],
        )  # [B*G]

        # Single update per generation step, so pi_old == pi; ratio == 1. The
        # ratio/clip form is kept for parity with the paper's PPO-style objective.
        old_logps = logps.detach()
        ratio = torch.exp(logps - old_logps)
        clipped = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        policy_loss = -torch.mean(
            torch.min(ratio * advantages.detach(), clipped * advantages.detach())
        )

        if ref_logps is not None:
            log_ratio = ref_logps - logps
            kl = (torch.exp(log_ratio) - log_ratio - 1.0).mean()  # k3 estimator
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

    # --- helpers ---------------------------------------------------------- #

    def _completion_mask(self, completion_ids):
        """Mask of real completion tokens: 1 up to and including the first EOS."""
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is None:
            return torch.ones_like(completion_ids)

        comp_len = completion_ids.size(1)
        is_eos = completion_ids == eos_id
        has_eos = is_eos.any(dim=1)
        first_eos = torch.where(
            has_eos,
            is_eos.int().argmax(dim=1),
            torch.full_like(has_eos, comp_len, dtype=torch.long),
        )
        positions = torch.arange(comp_len, device=completion_ids.device)
        return (positions.unsqueeze(0) <= first_eos.unsqueeze(1)).long()

    def _reference_logps(self, model, input_ids, attention_mask, labels):
        """Average log-probs under the frozen reference policy, or None if unavailable."""
        if self.ref_model is not None:
            return self._avg_logps(self.ref_model, input_ids, attention_mask, labels)
        if hasattr(model, "disable_adapter"):
            with _disable_grad_checkpointing(model), model.disable_adapter():
                return self._avg_logps(model, input_ids, attention_mask, labels)
        if not self._warned_no_ref:
            logger.warning(
                "GRPOMethod: beta > 0 but no reference policy is available "
                "(no ref_model and the model has no disable_adapter()). "
                "Skipping the KL penalty — pass ref_model or set beta=0."
            )
            self._warned_no_ref = True
        return None
