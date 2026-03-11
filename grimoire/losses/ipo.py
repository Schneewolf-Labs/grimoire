import torch

from ..data.preference import PreferenceCollator
from .orpo import _pad_dim1


class IPOLoss:
    """IPO (Identity Preference Optimization) loss.

    Paper: "A General Theoretical Paradigm to Understand Learning from Human Feedback"
           arXiv:2310.12036 (Azar et al., DeepMind)

    Loss = mean((log(pi/pi_ref)(chosen) - log(pi/pi_ref)(rejected) - 1/(2*beta))^2)

    Replaces DPO's log-sigmoid with a squared loss to prevent overfitting
    on noisy preference data. The 1/(2*beta) term acts as a target margin.
    Requires a frozen reference model like DPO.
    """

    def __init__(self, ref_model, beta=0.1, label_pad_token_id=-100):
        if ref_model.training:
            raise ValueError("ref_model must be in eval mode (call ref_model.eval() first)")
        self.ref_model = ref_model
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

        # Policy log-probs
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        all_logps = self._get_batch_logps(logits, labels)
        del logits
        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        # Reference log-probs (frozen, no grad)
        with torch.no_grad():
            ref_logits = self.ref_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
            ref_logps = self._get_batch_logps(ref_logits, labels)
            del ref_logits
            ref_chosen_logps = ref_logps[:len_chosen]
            ref_rejected_logps = ref_logps[len_chosen:]

        # IPO loss: ((log_ratio_chosen - log_ratio_rejected) - 1/(2*beta))^2
        pi_logratios = chosen_logps - rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits_diff = pi_logratios - ref_logratios
        loss = ((logits_diff - 1.0 / (2.0 * self.beta)) ** 2).mean()

        # Implicit rewards: beta * (log pi(y|x) - log pi_ref(y|x))
        chosen_rewards = self.beta * (chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = self.beta * (rejected_logps - ref_rejected_logps).detach()

        metrics = {
            "chosen_rewards": chosen_rewards.mean().item(),
            "rejected_rewards": rejected_rewards.mean().item(),
            "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
            "reward_accuracy": (chosen_rewards > rejected_rewards).float().mean().item(),
            "log_odds_ratio": logits_diff.detach().mean().item(),
        }

        return loss, metrics

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

    def _get_batch_logps(self, logits, labels):
        """Average log probability per sequence over response tokens only."""
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]

        loss_mask = shift_labels != self.label_pad_token_id
        safe_labels = torch.where(loss_mask, shift_labels, 0)

        # gather + logsumexp avoids materializing the full [batch, seq, vocab] log_softmax tensor
        gathered_logits = torch.gather(shift_logits, dim=2, index=safe_labels.unsqueeze(2)).squeeze(2)
        per_token_logps = gathered_logits - torch.logsumexp(shift_logits, dim=-1)

        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1)
