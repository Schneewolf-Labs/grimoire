import torch


class KTOCollator:
    """Pads KTO sequences to the max length in each batch, preserving kto_label."""

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

        kto_labels = torch.tensor([f["kto_label"] for f in features], dtype=torch.bool)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "kto_label": kto_labels,
        }

        # Pass through cached reference log probs when present
        if "ref_logps" in features[0]:
            batch["ref_logps"] = torch.tensor(
                [f["ref_logps"] for f in features]
            )

        return batch


def tokenize_kto(
    example,
    tokenizer,
    max_length=2048,
    prompt_field="prompt",
    response_field="response",
    label_field="label",
):
    """Tokenize a single example for KTO training.

    Each example has a prompt, a response, and a boolean label indicating
    whether the response is desirable (True) or undesirable (False).

    Use with dataset.map():
        dataset = dataset.map(
            lambda x: tokenize_kto(x, tokenizer, max_length=2048),
            remove_columns=dataset.column_names,
        )
    """
    prompt = example[prompt_field]
    response = example[response_field]

    prompt_tokens = tokenizer(prompt, add_special_tokens=False)
    prompt_len = len(prompt_tokens["input_ids"])

    tokens = tokenizer(prompt + response, max_length=max_length, truncation=True)

    labels = list(tokens["input_ids"])
    mask_len = min(prompt_len, len(labels))
    labels[:mask_len] = [-100] * mask_len

    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "labels": labels,
        "kto_label": bool(example[label_field]),
    }
