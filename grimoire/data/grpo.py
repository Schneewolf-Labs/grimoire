import torch


class GRPOCollator:
    """Left-pads prompt-only sequences for GRPO training.

    GRPO only needs prompts — completions are generated during training.
    Each example has input_ids and attention_mask (no labels).

    Prompts are LEFT-padded: decoder-only generation requires the prompt to
    end at the last position so completions directly continue it. With right
    padding, the model would generate after a run of pad tokens.
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
            input_ids[i, max_len - seq_len:] = torch.tensor(f["input_ids"], dtype=torch.long)
            attention_mask[i, max_len - seq_len:] = torch.tensor(f["attention_mask"], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


def tokenize_grpo(
    example,
    tokenizer,
    max_prompt_length=512,
    prompt_field="prompt",
):
    """Tokenize a prompt for GRPO training (no response).

    GRPO generates completions on-the-fly during training, so only prompts
    are tokenized ahead of time.

    Use with dataset.map():
        dataset = dataset.map(
            lambda x: tokenize_grpo(x, tokenizer, max_prompt_length=512),
            remove_columns=dataset.column_names,
        )
    """
    prompt = example[prompt_field]

    tokens = tokenizer(prompt, max_length=max_prompt_length, truncation=True)

    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
    }
