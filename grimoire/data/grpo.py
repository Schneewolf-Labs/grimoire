import torch


class GRPOCollator:
    """Left-pads prompt-only sequences for online-method training.

    Online methods (GRPO/RLOO/Online DPO/RAFT) only need prompts —
    completions are generated during training. Each example has input_ids
    and attention_mask (no labels).

    Prompts are LEFT-padded: decoder-only generation requires the prompt to
    end at the last position so completions directly continue it. With right
    padding, the model would generate after a run of pad tokens.

    Any other keys on the features are treated as metadata: they are gathered
    into ``batch["metadata"]`` — a list of dicts, one per example — instead of
    being tensorized. Online methods forward them to the reward function,
    aligned to the generated completions. Use
    ``tokenize_grpo(metadata_fields=...)`` to carry dataset columns
    (e.g. expected outputs, test specs) through to the reward.
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

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        meta_keys = [k for k in features[0] if k not in ("input_ids", "attention_mask")]
        if meta_keys:
            batch["metadata"] = [{k: f[k] for k in meta_keys} for f in features]

        return batch


def tokenize_grpo(
    example,
    tokenizer,
    max_prompt_length=512,
    prompt_field="prompt",
    metadata_fields=None,
):
    """Tokenize a prompt for online-method training (no response).

    Online methods (GRPO/RLOO/Online DPO/RAFT) generate completions
    on-the-fly during training, so only prompts are tokenized ahead of time.

    ``metadata_fields`` names dataset columns to copy through untouched.
    GRPOCollator gathers them into ``batch["metadata"]`` and the rollout
    passes them to the reward function as a third argument — use this for
    correctness/test-based rewards (expected outputs, test specs) instead of
    smuggling them through the prompt string.

    Use with dataset.map():
        dataset = dataset.map(
            lambda x: tokenize_grpo(x, tokenizer, max_prompt_length=512,
                                    metadata_fields=["expected_stdout"]),
            remove_columns=dataset.column_names,
        )
    """
    prompt = example[prompt_field]

    tokens = tokenizer(prompt, max_length=max_prompt_length, truncation=True)

    out = {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
    }
    for name in metadata_fields or []:
        out[name] = example[name]
    return out
