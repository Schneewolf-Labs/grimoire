import torch
import torch.nn.functional as F

from ..data.preference import PreferenceCollator


class ORPOLoss:
    """ORPO (Odds Ratio Preference Optimization) loss.

    Paper: "ORPO: Monolithic Preference Optimization without Reference Model"
           arXiv:2403.07691

    Loss = NLL(chosen) + beta * -mean(log(sigmoid(log_odds_ratio)))

    No reference model needed — the odds ratio between chosen and rejected
    responses provides the preference signal directly.
    """

    def __init__(self, beta=0.1, label_pad_token_id=-100):
        self.beta = beta
        self.label_pad_token_id = label_pad_token_id
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

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits

        # NLL loss on chosen response tokens only
        chosen_nll = self._compute_nll(logits[:len_chosen], labels[:len_chosen])

        # Log probabilities (average per sequence for length-invariance)
        all_logps = self._get_batch_logps(logits, labels)
        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        # Odds ratio: log(odds_chosen / odds_rejected)
        # where odds(x) = P(x) / (1 - P(x))
        # In log space: log_odds = log_p - log(1-p) = log_p - log1p(-exp(log_p))
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
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
        """Eval uses NLL on chosen sequences only (same as standard LM eval)."""
        outputs = model(
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
            labels=batch["chosen_labels"],
        )
        return outputs.loss, {}

    def _concatenate(self, batch):
        """Concatenate chosen and rejected into a single batch, padding to equal length."""
        max_len = max(batch["chosen_input_ids"].size(1), batch["rejected_input_ids"].size(1))

        input_ids = torch.cat([
            _pad_dim1(batch["chosen_input_ids"], max_len, self._pad_token_id),
            _pad_dim1(batch["rejected_input_ids"], max_len, self._pad_token_id),
        ], dim=0)

        attention_mask = torch.cat([
            _pad_dim1(batch["chosen_attention_mask"], max_len, 0),
            _pad_dim1(batch["rejected_attention_mask"], max_len, 0),
        ], dim=0)

        labels = torch.cat([
            _pad_dim1(batch["chosen_labels"], max_len, self.label_pad_token_id),
            _pad_dim1(batch["rejected_labels"], max_len, self.label_pad_token_id),
        ], dim=0)

        return input_ids, attention_mask, labels

    def _compute_nll(self, logits, labels):
        """Cross-entropy loss on response tokens (prompt tokens masked with -100)."""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.label_pad_token_id,
        )

    def _get_batch_logps(self, logits, labels):
        """Average log probability per sequence over response tokens only."""
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]

        loss_mask = shift_labels != self.label_pad_token_id
        safe_labels = torch.where(loss_mask, shift_labels, 0)

        # gather + logsumexp avoids materializing the full [batch, seq, vocab] log_softmax tensor
        gathered_logits = torch.gather(shift_logits, dim=2, index=safe_labels.unsqueeze(2)).squeeze(2)
        per_token_logps = gathered_logits - torch.logsumexp(shift_logits, dim=-1)

        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)


def _pad_dim1(tensor, length, value):
    """Pad a 2D tensor along dim=1 (sequence length) to the target length."""
    if tensor.size(1) >= length:
        return tensor
    pad = torch.full(
        (tensor.size(0), length - tensor.size(1)),
        value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor, pad], dim=1)
