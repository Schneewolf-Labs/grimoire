from ..data.sft import SFTCollator
from .utils import safe_cross_entropy_nll


class SFTLoss:
    """Standard supervised fine-tuning loss (next-token prediction).

    Computes cross-entropy with label clamping to avoid CUDA illegal memory
    access when token IDs exceed the model's vocabulary size (can happen with
    extended/abliterated tokenizers).  Prompt tokens masked with -100 in labels
    are excluded from loss.
    """

    def __call__(self, model, batch, training=True):
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        loss = safe_cross_entropy_nll(outputs.logits, batch["labels"])
        return loss, {}

    def create_collator(self, pad_token_id):
        return SFTCollator(pad_token_id=pad_token_id)
