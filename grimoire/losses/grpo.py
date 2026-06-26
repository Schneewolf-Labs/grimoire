from dataclasses import dataclass
from typing import Literal, Optional

import torch


@dataclass
class GRPOLossOutput:
    """Result of a GRPOLoss call.

    ``loss`` is a scalar tensor (carries grad). The remaining fields are detached
    Python floats that the trainer forwards to callbacks for logging.
    """

    loss: torch.Tensor
    kl: float
    clip_frac: float
    ratio_mean: float


class GRPOLoss:
    """Group-Relative Policy Optimization loss (DeepSeekMath / Dr.GRPO variants).

    Paper: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open
           Language Models" (arXiv:2402.03300) and "Understanding R1-Zero-Like
           Training: A Critical Perspective" (Dr.GRPO, arXiv:2503.20783).

    This is a PURE function of precomputed per-token log-probabilities and
    per-sequence advantages — no model, no generation, no I/O. ``GRPOTrainer``
    produces all the inputs (rollout generation + forward passes) and calls this
    to turn them into a scalar.

        L_GRPO = mean_t( -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A) + beta * KL )

        ratio = exp(logprobs - old_logprobs)          (per token)
        A     = advantages                            (per sequence, broadcast)
        KL    = exp(ref - logprobs) - (ref - logprobs) - 1   (k3 estimator)

    Args:
        beta:       KL penalty coefficient toward the reference policy. 0.0
                    disables the KL term entirely (and lets the trainer skip
                    reference log-probs / pass ``ref_logprobs=None``).
        epsilon:    PPO-style clip range for the importance ratio (default 0.2).
                    Only matters when num_iterations > 1 (old != current policy);
                    when they are equal the ratio is exactly 1 and nothing clips.
        loss_type:  "grpo"    -> per-sequence length normalization (original).
                    "dr_grpo" -> divide by a constant (padded completion width),
                                 no per-sequence length normalization (bias-corrected).
        scale_rewards: documents the advantage contract for the two variants.
                    True (GRPO): advantages divide by the group reward std.
                    False (Dr.GRPO): advantages are only mean-centered. The
                    trainer computes advantages and reads this flag — the loss
                    itself receives already-normalized advantages.
    """

    def __init__(
        self,
        beta: float = 0.04,
        epsilon: float = 0.2,
        loss_type: Literal["grpo", "dr_grpo"] = "grpo",
        scale_rewards: bool = True,
    ):
        if loss_type not in ("grpo", "dr_grpo"):
            raise ValueError(f"loss_type must be 'grpo' or 'dr_grpo', got {loss_type!r}")
        self.beta = beta
        self.epsilon = epsilon
        self.loss_type = loss_type
        self.scale_rewards = scale_rewards

    def __call__(
        self,
        *,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        ref_logprobs: Optional[torch.Tensor],
        advantages: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> GRPOLossOutput:
        # Importance ratio between the current and the sampling (old) policy.
        # When num_iterations == 1 the trainer passes old == logprobs.detach(),
        # so ratio is exactly 1 and the clip is a no-op.
        ratio = torch.exp(logprobs - old_logprobs)  # (N, T)
        adv = advantages.unsqueeze(1)  # (N, 1)

        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv
        per_token_loss = -torch.min(unclipped, clipped)  # (N, T)

        per_token_kl = None
        if self.beta != 0.0 and ref_logprobs is not None:
            # k3 estimator: unbiased and non-negative (Schulman).
            diff = ref_logprobs - logprobs
            per_token_kl = torch.exp(diff) - diff - 1.0
            per_token_loss = per_token_loss + self.beta * per_token_kl

        mask = completion_mask.to(per_token_loss.dtype)

        if self.loss_type == "grpo":
            # Per-sequence length normalization, then average over sequences.
            seq_loss = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            loss = seq_loss.mean()
        else:  # dr_grpo — divide by a constant (the padded completion width)
            loss = (per_token_loss * mask).sum() / (mask.size(0) * mask.size(1))

        # Detached metrics over completion tokens only.
        with torch.no_grad():
            tok = mask.sum().clamp(min=1.0)
            ratio_mean = ((ratio * mask).sum() / tok).item()
            is_clipped = (
                ((ratio < 1 - self.epsilon) & (adv < 0))
                | ((ratio > 1 + self.epsilon) & (adv > 0))
            ).to(mask.dtype)
            clip_frac = ((is_clipped * mask).sum() / tok).item()
            kl = (
                ((per_token_kl * mask).sum() / tok).item()
                if per_token_kl is not None
                else 0.0
            )

        return GRPOLossOutput(loss=loss, kl=kl, clip_frac=clip_frac, ratio_mean=ratio_mean)
