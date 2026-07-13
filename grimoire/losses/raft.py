import torch

from .online import OnlineMethod
from .utils import DEFAULT_FUSED_CHUNK_SIZE, forward_per_token_logps


class RAFTMethod(OnlineMethod):
    """RAFT (Reward-rAnked FineTuning) — best-of-N rejection sampling + SFT.

    Paper: "RAFT: Reward rAnked FineTuning for Generative Foundation Model
           Alignment" arXiv:2304.06767

    The simplest online method — a two-phase method (see ``OnlineMethod``):

    1. ``rollout(model, batch)`` — generates G completions per prompt, scores
       them with ``reward_fn``, and keeps ONLY the highest-reward completion
       of each group.
    2. ``__call__(model, experience)`` — plain NLL on the winner's response
       tokens: exactly the SFT loss, on data the policy just generated.

    No reference model, no KL term, no advantage weighting: the policy
    imitates its own best samples, so reward only ever enters through the
    argmax. A strong, hard-to-misconfigure baseline before reaching for
    GRPO/RLOO.

    ``min_reward`` guards the imitation: a winner is still just the best of
    G samples, and early in training the best of G may be garbage. Groups
    whose winning reward falls below ``min_reward`` are masked out of the
    loss (contributing zero gradient) instead of being imitated. The
    ``filtered_winners`` metric reports the masked fraction — if it sits at
    1.0, the model isn't clearing the floor yet.
    """

    def __init__(
        self,
        reward_fn,
        tokenizer,
        num_generations=4,
        min_reward=None,
        max_new_tokens=512,
        temperature=1.0,
        label_pad_token_id=-100,
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
            fused=fused,
            fused_chunk_size=fused_chunk_size,
        )
        self.min_reward = min_reward

    @torch.no_grad()
    def rollout(self, model, batch):
        """Turn a prompt-only batch into a best-of-G SFT batch.

        Returns ``input_ids`` / ``attention_mask`` / ``labels`` for the B
        winning prompt+completion sequences (one per prompt), a per-row
        ``keep_mask`` (winners below ``min_reward`` are masked out of the
        loss), and ``rollout_metrics``.
        """
        B = batch["input_ids"].size(0)
        G = self.num_generations

        gen = self._generate_and_score(model, batch)
        rewards_grouped = gen["rewards"].view(B, G)

        # Keep the best completion per group. Rows in gen are grouped by
        # prompt (i*G..(i+1)*G-1), so offset the within-group argmax.
        row_offset = torch.arange(B, device=rewards_grouped.device) * G
        best_rewards = rewards_grouped.max(dim=1).values  # [B]
        best_idx = row_offset + rewards_grouped.argmax(dim=1)  # [B]

        # Reward floor: never imitate a winner that is merely the best of a
        # bad group. Masked (not dropped) so the loss phase always sees B
        # rows and the gradient graph survives an all-filtered batch.
        if self.min_reward is not None:
            keep_mask = (best_rewards >= self.min_reward).float()  # [B]
        else:
            keep_mask = torch.ones_like(best_rewards)

        metrics = {
            "rewards_mean": rewards_grouped.mean().item(),
            "rewards_std": rewards_grouped.std().item(),
            "best_reward": best_rewards.mean().item(),
            "completion_length": float(gen["completion_length"]),
        }
        if self.min_reward is not None:
            metrics["filtered_winners"] = 1.0 - keep_mask.mean().item()

        return {
            "input_ids": gen["input_ids"][best_idx],
            "attention_mask": gen["attention_mask"][best_idx],
            "labels": gen["labels"][best_idx],
            "keep_mask": keep_mask,
            "rollout_metrics": metrics,
        }

    # --- loss phase (pure) ------------------------------------------------ #

    def __call__(self, model, batch, training=True):
        if not training:
            # Eval batches are prompt-only (the trainer runs no rollout there).
            return torch.zeros((), device=batch["input_ids"].device), {}

        # SFT loss on the kept winners: NLL over response tokens (prompt
        # masked), token-averaged across kept rows only.
        per_token_logps, loss_mask = forward_per_token_logps(
            model, batch["input_ids"], batch["attention_mask"], batch["labels"],
            self.label_pad_token_id, fused=self.fused, fused_chunk_size=self.fused_chunk_size,
        )
        loss_mask = loss_mask * batch["keep_mask"].unsqueeze(1)
        loss = -(per_token_logps * loss_mask).sum() / loss_mask.sum().clamp(min=1)

        return loss, dict(batch["rollout_metrics"])
