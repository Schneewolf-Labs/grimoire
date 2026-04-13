import torch

from ..data.preference import PreferenceCollator
from .utils import concatenate_preference


class RewardModelLoss:
    """Bradley-Terry pairwise ranking loss for reward model training.

    Trains a model to assign higher scalar rewards to chosen sequences
    over rejected sequences. The model should output a single scalar
    per sequence (e.g., AutoModelForSequenceClassification with num_labels=1).

    Loss formula:
        L = -mean(log(sigmoid(r_chosen - r_rejected - margin)))

    Args:
        margin: Minimum reward gap to enforce between chosen/rejected (default 0.0).
        pad_token_id: Token ID used for padding in concatenated batches.
        label_pad_token_id: Label padding value (passed through but not used for loss).
    """

    def __init__(self, margin=0.0, pad_token_id=0, label_pad_token_id=-100):
        self.margin = margin
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, model, batch, training=True):
        input_ids, attention_mask, _ = concatenate_preference(
            batch, self.pad_token_id, self.label_pad_token_id
        )

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        rewards = outputs.logits
        if rewards.ndim > 1:
            rewards = rewards.squeeze(-1)

        half = rewards.size(0) // 2
        rewards_chosen = rewards[:half]
        rewards_rejected = rewards[half:]

        loss = -torch.nn.functional.logsigmoid(
            rewards_chosen - rewards_rejected - self.margin
        ).mean()

        with torch.no_grad():
            accuracy = (rewards_chosen > rewards_rejected).float().mean().item()
            reward_margin = (rewards_chosen - rewards_rejected).mean().item()
            chosen_reward = rewards_chosen.mean().item()
            rejected_reward = rewards_rejected.mean().item()

        metrics = {
            "accuracy": accuracy,
            "reward_margin": reward_margin,
            "chosen_reward": chosen_reward,
            "rejected_reward": rejected_reward,
        }

        return loss, metrics

    def create_collator(self, pad_token_id):
        return PreferenceCollator(pad_token_id=pad_token_id)
