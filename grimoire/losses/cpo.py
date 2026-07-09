import torch.nn.functional as F

from ..data.preference import PreferenceCollator
from .utils import (
    DEFAULT_FUSED_CHUNK_SIZE,
    concatenate_preference,
    forward_per_token_logps,
    masked_avg_logps,
)


class CPOLoss:
    """CPO (Contrastive Preference Optimization) loss.

    Paper: "CPO: Change is Hard: A Closer Look at Suboptimal Engagements
            with LLM Alignment"
           arXiv:2312.02143

    Loss = L_SFT(chosen) + beta * L_preference

    L_SFT        = CrossEntropy on chosen response tokens (prompt masked)
    L_preference = -mean(log(sigmoid(beta * (avg_logp_chosen - avg_logp_rejected))))

    Reference-free like ORPO/SimPO, but combines SFT regularization with
    a contrastive preference term (theoretically cleaner than ORPO's odds ratio).
    """

    def __init__(self, beta=0.1, label_smoothing=0.0, label_pad_token_id=-100, fused=True,
                 fused_chunk_size=DEFAULT_FUSED_CHUNK_SIZE):
        self.beta = beta
        self.label_smoothing = label_smoothing
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
        nll_loss = -(per_token_logps[:len_chosen] * chosen_mask).sum() / chosen_mask.sum().clamp(min=1)

        # Average log-probability per sequence
        all_logps = masked_avg_logps(per_token_logps, loss_mask)
        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        # Preference loss: -log sigmoid(beta * (avg_logp_chosen - avg_logp_rejected))
        # With label smoothing: -(1-eps)*logsigmoid(x) - eps*logsigmoid(-x)
        logits_diff = chosen_logps - rejected_logps
        scaled_diff = self.beta * logits_diff
        preference_loss = -(
            (1 - self.label_smoothing) * F.logsigmoid(scaled_diff)
            + self.label_smoothing * F.logsigmoid(-scaled_diff)
        ).mean()

        loss = nll_loss + self.beta * preference_loss

        # Implicit rewards: beta * avg_logp (reference-free)
        chosen_rewards = (self.beta * chosen_logps).detach()
        rejected_rewards = (self.beta * rejected_logps).detach()

        metrics = {
            "nll_loss": nll_loss.detach().item(),
            "preference_loss": preference_loss.detach().item(),
            "chosen_rewards": chosen_rewards.mean().item(),
            "rejected_rewards": rejected_rewards.mean().item(),
            "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
            "reward_accuracy": (chosen_logps > rejected_logps).float().mean().item(),
            "logps_diff": logits_diff.detach().mean().item(),
        }

        return loss, metrics

    def _eval_forward(self, model, batch):
        """Eval uses the same forward pass as training."""
        return self._train_forward(model, batch)

    def _concatenate(self, batch):
        """Concatenate chosen and rejected into a single batch, padding to equal length."""
        return concatenate_preference(batch, self._pad_token_id, self.label_pad_token_id)
