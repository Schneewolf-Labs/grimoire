from ..data.sft import SFTCollator
from .utils import DEFAULT_FUSED_CHUNK_SIZE, forward_per_token_logps


class SFTLoss:
    """Standard supervised fine-tuning loss (next-token prediction).

    Uses the shared vectorized log-prob computation (same as the preference
    training paths) to avoid allocating a .contiguous() copy of the full
    logits tensor.  With ``fused=True`` (default) and a model that supports
    it, the loss is computed chunk-by-chunk from the final hidden states
    without materializing the [batch, seq, vocab] logits tensor at all.
    Prompt tokens masked with -100 in labels are excluded from loss.
    """

    def __init__(self, label_pad_token_id=-100, fused=True, fused_chunk_size=DEFAULT_FUSED_CHUNK_SIZE):
        self.label_pad_token_id = label_pad_token_id
        self.fused = fused
        self.fused_chunk_size = fused_chunk_size

    def __call__(self, model, batch, training=True):
        per_token_logps, loss_mask = forward_per_token_logps(
            model,
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
            self.label_pad_token_id,
            fused=self.fused,
            fused_chunk_size=self.fused_chunk_size,
            position_ids=batch.get("position_ids"),
        )

        loss = -(per_token_logps * loss_mask).sum() / loss_mask.sum().clamp(min=1)
        return loss, {}

    def create_collator(self, pad_token_id):
        return SFTCollator(pad_token_id=pad_token_id)
