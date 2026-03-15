import torch.nn.functional as F

from ..data.preference import PreferenceCollator
from .utils import get_batch_logps, concatenate_preference


class SimPOLoss:
    """SimPO (Simple Preference Optimization) loss.

    Paper: "SimPO: Simple Preference Optimization with a Reference-Free Reward"
           arXiv:2405.14734

    Loss = -mean(log(sigmoid(beta * (avg_logp_chosen - avg_logp_rejected - gamma))))

    No reference model needed — uses length-normalized average log probability
    as an implicit reward, with a target reward margin gamma.
    """

    def __init__(self, beta=2.0, gamma=0.5, label_pad_token_id=-100):
        self.beta = beta
        self.gamma = gamma
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
        del input_ids, attention_mask  # Free concatenated tensors
        all_logps = get_batch_logps(logits, labels, self.label_pad_token_id)
        del logits, labels
        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        # SimPO loss: -log sigmoid(beta * (avg_logp_chosen - avg_logp_rejected - gamma))
        logits_diff = chosen_logps - rejected_logps - self.gamma
        loss = -F.logsigmoid(self.beta * logits_diff).mean()

        # Implicit rewards: beta * avg_logp (no reference model)
        chosen_rewards = (self.beta * chosen_logps).detach()
        rejected_rewards = (self.beta * rejected_logps).detach()

        metrics = {
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

    def _get_batch_logps(self, logits, labels):
        """Average log probability per sequence over response tokens only."""
        return get_batch_logps(logits, labels, self.label_pad_token_id)
