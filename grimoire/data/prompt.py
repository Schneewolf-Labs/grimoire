"""Prompt-only collator and tokenization for online RL methods (PPO, GRPO)."""

import torch


class PromptCollator:
    """Left-pads prompt-only sequences for generation compatibility.

    Left-padding ensures model.generate() appends tokens at the correct
    position for all sequences regardless of their original lengths.
    """

    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        batch_size = len(features)

        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)

        for i, f in enumerate(features):
            seq_len = len(f["input_ids"])
            # Left-pad: place tokens at the right end
            input_ids[i, max_len - seq_len:] = torch.tensor(f["input_ids"], dtype=torch.long)
            attention_mask[i, max_len - seq_len:] = torch.tensor(f["attention_mask"], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


def tokenize_prompt(
    example,
    tokenizer,
    max_length=512,
    prompt_field="prompt",
):
    """Tokenize a prompt for online RL methods (PPO, GRPO).

    Only tokenizes the prompt — completions are generated during training.

    Use with dataset.map():
        dataset = dataset.map(
            lambda x: tokenize_prompt(x, tokenizer, max_length=512),
            remove_columns=dataset.column_names,
        )
    """
    tokens = tokenizer(
        example[prompt_field],
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )

    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
    }
