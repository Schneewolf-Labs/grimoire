import torch


class SFTCollator:
    """Pads SFT sequences to the max length in each batch."""

    def __init__(self, pad_token_id=0, label_pad_token_id=-100):
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        batch_size = len(features)

        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        labels = torch.full((batch_size, max_len), self.label_pad_token_id, dtype=torch.long)

        for i, f in enumerate(features):
            seq_len = len(f["input_ids"])
            input_ids[i, :seq_len] = torch.tensor(f["input_ids"], dtype=torch.long)
            attention_mask[i, :seq_len] = torch.tensor(f["attention_mask"], dtype=torch.long)
            labels[i, :seq_len] = torch.tensor(f["labels"], dtype=torch.long)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def tokenize_sft(
    example,
    tokenizer,
    max_length=2048,
    max_prompt_length=None,
    text_field=None,
    prompt_field=None,
    response_field=None,
):
    """Tokenize a single example for SFT training.

    Two modes:
      1. text_field: tokenize a single text column, all tokens contribute to loss
      2. prompt_field + response_field: concatenate, mask prompt tokens in labels

    Args:
        max_length: Maximum total sequence length (prompt + response).
        max_prompt_length: Maximum prompt length in tokens. Longer prompts are
            truncated so more of the response is preserved for training.

    Use with dataset.map():
        dataset = dataset.map(
            lambda x: tokenize_sft(x, tokenizer, max_length=2048, prompt_field="prompt", response_field="response"),
            remove_columns=dataset.column_names,
        )
    """
    if text_field:
        tokens = tokenizer(example[text_field], max_length=max_length, truncation=True)
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "labels": list(tokens["input_ids"]),
        }

    if prompt_field and response_field:
        prompt = example[prompt_field]
        response = example[response_field]

        prompt_tokens = tokenizer(prompt, add_special_tokens=False)
        prompt_len = len(prompt_tokens["input_ids"])
        if max_prompt_length:
            prompt_len = min(prompt_len, max_prompt_length)

        tokens = tokenizer(prompt + response, max_length=max_length, truncation=True)

        labels = list(tokens["input_ids"])
        mask_len = min(prompt_len, len(labels))
        labels[:mask_len] = [-100] * mask_len

        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "labels": labels,
        }

    raise ValueError("Provide either text_field or both prompt_field and response_field")
