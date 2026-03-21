import torch

from ..data.preference import PreferenceCollator
from .utils import get_batch_logps, concatenate_preference, _disable_grad_checkpointing


class IPOLoss:
    """IPO (Identity Preference Optimization) loss.

    Paper: "A General Theoretical Paradigm to Understand Learning from Human Feedback"
           arXiv:2310.12036 (Azar et al., DeepMind)

    Loss = mean((log(pi/pi_ref)(chosen) - log(pi/pi_ref)(rejected) - 1/(2*beta))^2)

    Replaces DPO's log-sigmoid with a squared loss to prevent overfitting
    on noisy preference data. The 1/(2*beta) term acts as a target margin.
    Requires a frozen reference model like DPO.
    """

    def __init__(self, ref_model=None, beta=0.1, label_pad_token_id=-100):
        if ref_model is not None and ref_model.training:
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
        all_logps = get_batch_logps(logits, labels, self.label_pad_token_id)
        del logits
        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        # Reference log-probs: use cached values if available, else compute
        if "ref_chosen_logps" in batch:
            del input_ids, attention_mask, labels  # Free concatenated tensors
            ref_chosen_logps = batch["ref_chosen_logps"].to(chosen_logps.device)
            ref_rejected_logps = batch["ref_rejected_logps"].to(chosen_logps.device)
        else:
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_logits = self.ref_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
                elif hasattr(model, "disable_adapter"):
                    with _disable_grad_checkpointing(model), model.disable_adapter():
                        ref_logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
                else:
                    raise ValueError("IPOLoss requires either a ref_model, cached ref log probs in the batch, or a PEFT model with disable_adapter()")
                del input_ids, attention_mask  # Free concatenated tensors
                ref_logps = get_batch_logps(ref_logits, labels, self.label_pad_token_id)
                del ref_logits, labels
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
        """Eval uses the same forward pass as training."""
        return self._train_forward(model, batch)

    def _concatenate(self, batch):
        """Concatenate chosen and rejected into a single batch, padding to equal length."""
        return concatenate_preference(batch, self._pad_token_id, self.label_pad_token_id)
