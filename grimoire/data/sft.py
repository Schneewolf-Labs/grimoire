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


class PackedSFTCollator:
    """Packs multiple SFT sequences into single rows to minimize padding waste.

    Uses first-fit decreasing bin packing to fill rows up to max_length.
    Produces position_ids that reset at each sequence boundary, which
    flash attention 2 uses to isolate attention between packed sequences.

    Requires the model to support the ``position_ids`` forward argument
    and flash attention 2 (``attn_implementation="flash_attention_2"``)
    for proper attention isolation between packed sequences.
    """

    def __init__(self, pad_token_id=0, label_pad_token_id=-100, max_length=2048):
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id
        self.max_length = max_length

    def __call__(self, features):
        # Sort by length descending for better packing (first-fit decreasing)
        sorted_features = sorted(features, key=lambda f: len(f["input_ids"]), reverse=True)

        # Each bin: [ids_list, mask_list, labels_list, position_ids_list]
        bins = []

        for f in sorted_features:
            ids = list(f["input_ids"]) if not isinstance(f["input_ids"], list) else f["input_ids"]
            mask = list(f["attention_mask"]) if not isinstance(f["attention_mask"], list) else f["attention_mask"]
            labels = list(f["labels"]) if not isinstance(f["labels"], list) else f["labels"]
            seq_len = len(ids)

            if seq_len > self.max_length:
                ids = ids[:self.max_length]
                mask = mask[:self.max_length]
                labels = labels[:self.max_length]
                seq_len = self.max_length

            # Try to fit into an existing bin
            placed = False
            for bin_data in bins:
                if len(bin_data[0]) + seq_len <= self.max_length:
                    bin_data[0].extend(ids)
                    bin_data[1].extend(mask)
                    bin_data[2].extend(labels)
                    bin_data[3].extend(range(seq_len))
                    placed = True
                    break

            if not placed:
                bins.append([list(ids), list(mask), list(labels), list(range(seq_len))])

        # Pad all bins to the longest bin length
        max_len = max(len(b[0]) for b in bins)
        batch_size = len(bins)

        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        labels = torch.full((batch_size, max_len), self.label_pad_token_id, dtype=torch.long)
        position_ids = torch.zeros(batch_size, max_len, dtype=torch.long)

        for i, (ids, mask, lbls, pos) in enumerate(bins):
            seq_len = len(ids)
            input_ids[i, :seq_len] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :seq_len] = torch.tensor(mask, dtype=torch.long)
            labels[i, :seq_len] = torch.tensor(lbls, dtype=torch.long)
            position_ids[i, :seq_len] = torch.tensor(pos, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "position_ids": position_ids,
        }


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
