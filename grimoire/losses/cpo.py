import torch
import torch.nn.functional as F

from ..data.preference import PreferenceCollator
from .orpo import _pad_dim1


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

    def __init__(self, beta=0.1, label_smoothing=0.0, label_pad_token_id=-100):
        self.beta = beta
        self.label_smoothing = label_smoothing
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

        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits

        # SFT loss on chosen (using model's cross-entropy) — compute before freeing logits
        chosen_logits = logits[:len_chosen]
        chosen_labels = labels[:len_chosen]
        shift_logits_nll = chosen_logits[..., :-1, :].contiguous()
        shift_labels_nll = chosen_labels[..., 1:].contiguous()
        nll_loss = F.cross_entropy(
            shift_logits_nll.view(-1, shift_logits_nll.size(-1)),
            shift_labels_nll.view(-1),
            ignore_index=self.label_pad_token_id,
        )
        del shift_logits_nll, shift_labels_nll

        all_logps = self._get_batch_logps(logits, labels)
        del logits
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
