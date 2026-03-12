from ..data.sft import SFTCollator


class SFTLoss:
    """Standard supervised fine-tuning loss (next-token prediction).

    Uses the model's built-in cross-entropy with ignore_index=-100,
    so prompt tokens masked with -100 in labels are excluded from loss.
    """

    def __call__(self, model, batch, training=True):
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            use_cache=False,
        )
        return outputs.loss, {}

    def create_collator(self, pad_token_id):
        return SFTCollator(pad_token_id=pad_token_id)
