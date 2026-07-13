import torch.nn.functional as F

from ..data.preference import PreferenceCollator
from .utils import (
    DEFAULT_FUSED_CHUNK_SIZE,
    _log1mexp,
    concatenate_preference,
    forward_per_token_logps,
    masked_avg_logps,
)


class ORPOLoss:
    """ORPO (Odds Ratio Preference Optimization) loss.

    Paper: "ORPO: Monolithic Preference Optimization without Reference Model"
           arXiv:2403.07691

    Loss = NLL(chosen) + beta * -mean(log(sigmoid(log_odds_ratio)))

    No reference model needed — the odds ratio between chosen and rejected
    responses provides the preference signal directly.
    """

    def __init__(self, beta=0.1, label_pad_token_id=-100, fused=True,
                 fused_chunk_size=DEFAULT_FUSED_CHUNK_SIZE):
        self.beta = beta
        self.label_pad_token_id = label_pad_token_id
        self.fused = fused
        self.fused_chunk_size = fused_chunk_size
        self._pad_token_id = 0

    def __call__(self, model, batch, training=True):
        if not training:
            return self._eval_forward(model, batch)
        return self._train_forward(model, batch)

    def create_collator(self, pad_token_id):
        self._pad_token_id = pad_token_id
        return PreferenceCollator(pad_token_id=pad_token_id, label_pad_token_id=self.label_pad_token_id)

    def _train_forward(self, model, batch):
        len_chosen = batch["chosen_input_ids"].size(0)

        # Concatenate chosen + rejected for a single forward pass
        input_ids, attention_mask, labels = self._concatenate(batch)

        # Per-token log-probs — chunked from hidden states when fused, so the
        # full [batch, seq, vocab] logits tensor is never materialized.
        per_token_logps, loss_mask = forward_per_token_logps(
            model, input_ids, attention_mask, labels, self.label_pad_token_id,
            fused=self.fused, fused_chunk_size=self.fused_chunk_size,
        )
        del input_ids, attention_mask, labels  # Free concatenated tensors

        # NLL on chosen response tokens — flat average matching F.cross_entropy
        chosen_mask = loss_mask[:len_chosen]
        chosen_nll = -(per_token_logps[:len_chosen] * chosen_mask).sum() / chosen_mask.sum().clamp(min=1)

        # Average log-probability per sequence (for odds ratio)
        all_logps = masked_avg_logps(per_token_logps, loss_mask)
        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        # Odds ratio: log(odds_chosen / odds_rejected)
        # where odds(x) = P(x) / (1 - P(x))
        # In log space: log_odds = log_p - log(1-p) = log_p - log1mexp(log_p)
        log_odds = (chosen_logps - rejected_logps) - (
            _log1mexp(chosen_logps) - _log1mexp(rejected_logps)
        )
        ratio = F.logsigmoid(log_odds)
        or_loss = -self.beta * ratio.mean()

        total_loss = chosen_nll + or_loss

        metrics = {
            "nll_loss": chosen_nll.detach().item(),
            "or_loss": or_loss.detach().item(),
            "chosen_rewards": (self.beta * chosen_logps.detach()).mean().item(),
            "rejected_rewards": (self.beta * rejected_logps.detach()).mean().item(),
            "log_odds_ratio": log_odds.detach().mean().item(),
            "reward_margin": (self.beta * (chosen_logps - rejected_logps).detach()).mean().item(),
            "reward_accuracy": (chosen_logps > rejected_logps).float().mean().item(),
        }

        return total_loss, metrics

    def _eval_forward(self, model, batch):
        """Eval uses the same forward pass as training."""
        return self._train_forward(model, batch)

    def _concatenate(self, batch):
        """Concatenate chosen and rejected into a single batch, padding to equal length."""
        return concatenate_preference(batch, self._pad_token_id, self.label_pad_token_id)
