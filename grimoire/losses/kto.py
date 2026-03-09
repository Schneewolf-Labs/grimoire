import torch
import torch.nn.functional as F

from ..data.kto import KTOCollator


class KTOLoss:
    """KTO (Kahneman-Tversky Optimization) loss.

    Paper: "KTO: Model Alignment as Prospect Theoretic Optimization"
           arXiv:2402.01306

    Works with unpaired binary feedback: each example is a prompt + response
    with a boolean label (desirable or undesirable). No chosen/rejected pairs needed.

    Desirable loss:   lambda_d * (1 - sigmoid(beta * (log_ratio - KL_ref)))
    Undesirable loss: lambda_u * (1 - sigmoid(beta * (KL_ref - log_ratio)))

    where log_ratio = avg_logp_policy - avg_logp_ref, and KL_ref is the batch
    mean of (avg_logp_policy - avg_logp_ref).

    Requires a frozen reference model to compute baseline log-probabilities.
    """

    def __init__(self, ref_model, beta=0.1, lambda_d=1.0, lambda_u=1.0, label_pad_token_id=-100):
        self.ref_model = ref_model
        self.beta = beta
        self.lambda_d = lambda_d
        self.lambda_u = lambda_u
        self.label_pad_token_id = label_pad_token_id
        self._pad_token_id = 0

    def __call__(self, model, batch, training=True):
        if not training:
            return self._eval_forward(model, batch)
        return self._train_forward(model, batch)

    def create_collator(self, pad_token_id):
        self._pad_token_id = pad_token_id
        return KTOCollator(pad_token_id=pad_token_id, label_pad_token_id=self.label_pad_token_id)

    def _train_forward(self, model, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        kto_label = batch["kto_label"]  # bool tensor: True=desirable, False=undesirable

        # Policy log-probs
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        policy_logps = self._get_batch_logps(logits, labels)

        # Reference log-probs (frozen, no grad)
        with torch.no_grad():
            ref_logits = self.ref_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
            ref_logps = self._get_batch_logps(ref_logits, labels)

        # Log ratios and KL estimate
        log_ratio = policy_logps - ref_logps
        kl_ref = (policy_logps - ref_logps).detach().mean().clamp(min=0)

        # Split by label
        desirable_mask = kto_label
        undesirable_mask = ~kto_label

        # Compute loss
        loss = torch.tensor(0.0, device=input_ids.device)
        n_terms = 0

        if desirable_mask.any():
            desirable_loss = self.lambda_d * (1 - F.sigmoid(self.beta * (log_ratio[desirable_mask] - kl_ref)))
            loss = loss + desirable_loss.mean()
            n_terms += 1

        if undesirable_mask.any():
            undesirable_loss = self.lambda_u * (1 - F.sigmoid(self.beta * (kl_ref - log_ratio[undesirable_mask])))
            loss = loss + undesirable_loss.mean()
            n_terms += 1

        # Implicit rewards: beta * log(pi/pi_ref)
        rewards = self.beta * log_ratio.detach()
        desirable_rewards = rewards[desirable_mask] if desirable_mask.any() else torch.zeros(1, device=input_ids.device)
        undesirable_rewards = rewards[undesirable_mask] if undesirable_mask.any() else torch.zeros(1, device=input_ids.device)

        metrics = {
            "chosen_rewards": desirable_rewards.mean().item(),
            "rejected_rewards": undesirable_rewards.mean().item(),
            "reward_margin": (desirable_rewards.mean() - undesirable_rewards.mean()).item(),
            "reward_accuracy": (
                (desirable_rewards > 0).float().sum() + (undesirable_rewards < 0).float().sum()
            ).item() / max(len(rewards), 1),
            "kl_ref": kl_ref.item(),
        }

        return loss, metrics

    def _eval_forward(self, model, batch):
        """Eval uses NLL on all sequences (same as standard LM eval)."""
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return outputs.loss, {}

    def _get_batch_logps(self, logits, labels):
        """Average log probability per sequence over response tokens only."""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].clone()

        loss_mask = shift_labels != self.label_pad_token_id
        shift_labels[shift_labels == self.label_pad_token_id] = 0

        per_token_logps = torch.gather(
            F.log_softmax(shift_logits, dim=-1),
            dim=2,
            index=shift_labels.unsqueeze(2),
        ).squeeze(2)

        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
