import torch


class PreferenceCollator:
    """Pads chosen/rejected preference pairs to the max length in each batch."""

    def __init__(self, pad_token_id=0, label_pad_token_id=-100):
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features):
        max_chosen = max(len(f["chosen_input_ids"]) for f in features)
        max_rejected = max(len(f["rejected_input_ids"]) for f in features)
        batch_size = len(features)

        batch = {
            "chosen_input_ids": torch.full((batch_size, max_chosen), self.pad_token_id, dtype=torch.long),
            "chosen_attention_mask": torch.zeros(batch_size, max_chosen, dtype=torch.long),
            "chosen_labels": torch.full((batch_size, max_chosen), self.label_pad_token_id, dtype=torch.long),
            "rejected_input_ids": torch.full((batch_size, max_rejected), self.pad_token_id, dtype=torch.long),
            "rejected_attention_mask": torch.zeros(batch_size, max_rejected, dtype=torch.long),
            "rejected_labels": torch.full((batch_size, max_rejected), self.label_pad_token_id, dtype=torch.long),
        }

        for i, f in enumerate(features):
            c_len = len(f["chosen_input_ids"])
            batch["chosen_input_ids"][i, :c_len] = torch.tensor(f["chosen_input_ids"], dtype=torch.long)
            batch["chosen_attention_mask"][i, :c_len] = torch.tensor(f["chosen_attention_mask"], dtype=torch.long)
            batch["chosen_labels"][i, :c_len] = torch.tensor(f["chosen_labels"], dtype=torch.long)

            r_len = len(f["rejected_input_ids"])
            batch["rejected_input_ids"][i, :r_len] = torch.tensor(f["rejected_input_ids"], dtype=torch.long)
            batch["rejected_attention_mask"][i, :r_len] = torch.tensor(f["rejected_attention_mask"], dtype=torch.long)
            batch["rejected_labels"][i, :r_len] = torch.tensor(f["rejected_labels"], dtype=torch.long)

        # Pass through cached reference log probs when present
        if "ref_chosen_logps" in features[0]:
            batch["ref_chosen_logps"] = torch.tensor(
                [f["ref_chosen_logps"] for f in features]
            )
            batch["ref_rejected_logps"] = torch.tensor(
                [f["ref_rejected_logps"] for f in features]
            )

        return batch


def tokenize_preference(
    example,
    tokenizer,
    max_length=2048,
    max_prompt_length=None,
    prompt_field="prompt",
    chosen_field="chosen",
    rejected_field="rejected",
):
    """Tokenize a single example for preference training (ORPO/DPO).

    Use with dataset.map():
        dataset = dataset.map(
            lambda x: tokenize_preference(x, tokenizer, max_length=2048),
            remove_columns=dataset.column_names,
        )
    """
    prompt = example[prompt_field]
    chosen = example[chosen_field]
    rejected = example[rejected_field]

    prompt_tokens = tokenizer(prompt, add_special_tokens=False)
    prompt_len = len(prompt_tokens["input_ids"])
    if max_prompt_length:
        prompt_len = min(prompt_len, max_prompt_length)

    chosen_tokens = tokenizer(prompt + chosen, max_length=max_length, truncation=True)
    chosen_labels = list(chosen_tokens["input_ids"])
    mask_len = min(prompt_len, len(chosen_labels))
    chosen_labels[:mask_len] = [-100] * mask_len

    rejected_tokens = tokenizer(prompt + rejected, max_length=max_length, truncation=True)
    rejected_labels = list(rejected_tokens["input_ids"])
    mask_len = min(prompt_len, len(rejected_labels))
    rejected_labels[:mask_len] = [-100] * mask_len

    return {
        "chosen_input_ids": chosen_tokens["input_ids"],
        "chosen_attention_mask": chosen_tokens["attention_mask"],
        "chosen_labels": chosen_labels,
        "rejected_input_ids": rejected_tokens["input_ids"],
        "rejected_attention_mask": rejected_tokens["attention_mask"],
        "rejected_labels": rejected_labels,
    }
