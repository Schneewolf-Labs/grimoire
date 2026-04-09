import torch

from ..data.sft import SFTCollator
from .utils import _per_token_logps


class SFTLoss:
    """Standard supervised fine-tuning loss (next-token prediction).

    Uses row-by-row log-prob computation (same as preference training paths)
    to avoid allocating a .contiguous() copy of the full logits tensor.
    Prompt tokens masked with -100 in labels are excluded from loss.
    """

    def __init__(self, label_pad_token_id=-100):
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, model, batch, training=True):
        forward_kwargs = dict(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        if "position_ids" in batch:
            forward_kwargs["position_ids"] = batch["position_ids"]
        logits = model(**forward_kwargs).logits
        labels = batch["labels"]

        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        del logits, labels

        loss_mask = shift_labels != self.label_pad_token_id
        vocab_size = shift_logits.size(-1)
        safe_labels = torch.where(loss_mask, shift_labels, 0).clamp(max=vocab_size - 1)

        per_token_logps = _per_token_logps(shift_logits, safe_labels)
        del shift_logits, safe_labels

        loss = -(per_token_logps * loss_mask).sum() / loss_mask.sum().clamp(min=1)
        return loss, {}

    def create_collator(self, pad_token_id):
        return SFTCollator(pad_token_id=pad_token_id)
