import torch
import torch.nn.functional as F

from ..data.kto import KTOCollator
from .utils import (
    DEFAULT_FUSED_CHUNK_SIZE,
    _disable_grad_checkpointing,
    forward_per_token_logps,
    masked_avg_logps,
)


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

    def __init__(self, ref_model=None, beta=0.1, lambda_d=1.0, lambda_u=1.0, label_pad_token_id=-100,
                 fused=True, fused_chunk_size=DEFAULT_FUSED_CHUNK_SIZE):
        if ref_model is not None and ref_model.training:
            raise ValueError("ref_model must be in eval mode (call ref_model.eval() first)")
        self.ref_model = ref_model
        self.beta = beta
        self.lambda_d = lambda_d
        self.lambda_u = lambda_u
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
        return KTOCollator(pad_token_id=pad_token_id, label_pad_token_id=self.label_pad_token_id)

    def _avg_logps(self, model, input_ids, attention_mask, labels):
        per_token_logps, loss_mask = forward_per_token_logps(
            model, input_ids, attention_mask, labels, self.label_pad_token_id,
            fused=self.fused, fused_chunk_size=self.fused_chunk_size,
        )
        return masked_avg_logps(per_token_logps, loss_mask)

    def _train_forward(self, model, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        kto_label = batch["kto_label"]  # bool tensor: True=desirable, False=undesirable
        device = input_ids.device

        # Reference log-probs: cached values, or a no-grad forward.  The
        # frozen pass runs BEFORE the policy forward so its activations never
        # coexist with the policy's autograd graph — lower peak memory.
        if "ref_logps" in batch:
            ref_logps = batch["ref_logps"].to(device)
        else:
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_logps = self._avg_logps(self.ref_model, input_ids, attention_mask, labels)
                elif hasattr(model, "disable_adapter"):
                    with _disable_grad_checkpointing(model), model.disable_adapter():
                        ref_logps = self._avg_logps(model, input_ids, attention_mask, labels)
                else:
                    raise ValueError("KTOLoss requires either a ref_model, cached ref log probs in the batch, or a PEFT model with disable_adapter()")

        # Policy log-probs
        policy_logps = self._avg_logps(model, input_ids, attention_mask, labels)
        del input_ids, attention_mask, labels  # Free batch tensors

        # Log ratios and KL estimate
        log_ratio = policy_logps - ref_logps
        kl_ref = (policy_logps - ref_logps).detach().mean().clamp(min=0)

        # Split by label
        desirable_mask = kto_label
        undesirable_mask = ~kto_label

        # Compute loss
        loss = torch.tensor(0.0, device=device)
        n_terms = 0

        if desirable_mask.any():
            desirable_loss = self.lambda_d * (1 - F.sigmoid(self.beta * (log_ratio[desirable_mask] - kl_ref)))
            loss = loss + desirable_loss.mean()
            n_terms += 1

        if undesirable_mask.any():
            undesirable_loss = self.lambda_u * (1 - F.sigmoid(self.beta * (kl_ref - log_ratio[undesirable_mask])))
            loss = loss + undesirable_loss.mean()
            n_terms += 1

        if n_terms > 1:
            loss = loss / n_terms

        # Implicit rewards: beta * log(pi/pi_ref)
        rewards = self.beta * log_ratio.detach()
        desirable_rewards = rewards[desirable_mask] if desirable_mask.any() else torch.zeros(1, device=device)
        undesirable_rewards = rewards[undesirable_mask] if undesirable_mask.any() else torch.zeros(1, device=device)

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
        """Eval uses the same forward pass as training."""
        return self._train_forward(model, batch)
