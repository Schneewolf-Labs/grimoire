import torch

from .online import OnlineMethod
from .utils import forward_per_token_logps


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

    No reference model, no KL term, no advantage weighting, and only one
    hyperparameter that matters (``num_generations``): the policy imitates
    its own best samples, so reward only ever enters through the argmax.
    A strong, hard-to-misconfigure baseline before reaching for GRPO/RLOO.
    """

    @torch.no_grad()
    def rollout(self, model, batch):
        """Turn a prompt-only batch into a best-of-G SFT batch.

        Returns ``input_ids`` / ``attention_mask`` / ``labels`` for the B
        winning prompt+completion sequences (one per prompt) and
        ``rollout_metrics``.
        """
        B = batch["input_ids"].size(0)
        G = self.num_generations

        gen = self._generate_and_score(model, batch)
        rewards_grouped = gen["rewards"].view(B, G)

        # Keep the best completion per group. Rows in gen are grouped by
        # prompt (i*G..(i+1)*G-1), so offset the within-group argmax.
        row_offset = torch.arange(B, device=rewards_grouped.device) * G
        best_idx = row_offset + rewards_grouped.argmax(dim=1)  # [B]

        return {
            "input_ids": gen["input_ids"][best_idx],
            "attention_mask": gen["attention_mask"][best_idx],
            "labels": gen["labels"][best_idx],
            "rollout_metrics": {
                "rewards_mean": rewards_grouped.mean().item(),
                "rewards_std": rewards_grouped.std().item(),
                "best_reward": rewards_grouped.max(dim=1).values.mean().item(),
                "completion_length": float(gen["completion_length"]),
            },
        }

    # --- loss phase (pure) ------------------------------------------------ #

    def __call__(self, model, batch, training=True):
        if not training:
            # Eval batches are prompt-only (the trainer runs no rollout there).
            return torch.zeros((), device=batch["input_ids"].device), {}

        # SFT loss on the winners: NLL over response tokens (prompt masked).
        per_token_logps, loss_mask = forward_per_token_logps(
            model, batch["input_ids"], batch["attention_mask"], batch["labels"],
            self.label_pad_token_id, fused=self.fused, fused_chunk_size=self.fused_chunk_size,
        )
        loss = -(per_token_logps * loss_mask).sum() / loss_mask.sum().clamp(min=1)

        return loss, dict(batch["rollout_metrics"])
