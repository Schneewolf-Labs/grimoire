import torch


class PromptCollator:
    """Left-pads prompt-only batches for GRPO rollout generation.

    GRPO datasets are prompt-only — completions are generated online during
    training. Prompts are LEFT-padded (not right-padded like SFT) so that every
    prompt in the batch ends at the same column. That makes the prompt/completion
    boundary uniform across the batch after ``model.generate()``, which the
    trainer relies on to slice completion log-probs cleanly.

    Any dataset columns other than ``input_ids``/``attention_mask`` are passed
    through under ``batch["columns"]`` as ``{name: [values...]}`` (one entry per
    prompt) so the trainer can forward them to the reward function.
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
            # Left-pad: place the prompt at the right edge of the row.
            input_ids[i, max_len - seq_len:] = torch.tensor(f["input_ids"], dtype=torch.long)
            attention_mask[i, max_len - seq_len:] = torch.tensor(f["attention_mask"], dtype=torch.long)

        batch = {"input_ids": input_ids, "attention_mask": attention_mask}

        # Pass through any extra columns (solution/answer/...) for the reward fn.
        passthrough = set()
        for f in features:
            passthrough.update(k for k in f if k not in ("input_ids", "attention_mask"))
        if passthrough:
            batch["columns"] = {k: [f.get(k) for f in features] for k in sorted(passthrough)}

        return batch


def tokenize_prompt(
    row,
    tokenizer,
    *,
    prompt_key="prompt",
    system_key="system",
    max_prompt_length=None,
    apply_chat_template=True,
):
    """Tokenize a prompt for GRPO training (prompt-only, no response).

    Mirror of ``tokenize_sft``/``tokenize_preference`` but emits ONLY the
    tokenized prompt (``input_ids``/``attention_mask``) plus a passthrough of all
    original columns, so the reward function can receive ``solution``/``answer``/
    etc. via ``**columns``. Completions are generated on-the-fly by the trainer.

    When ``apply_chat_template`` is True and the tokenizer has a chat template,
    the prompt (and optional system message at ``system_key``) is rendered as a
    chat turn with a trailing generation prompt. Otherwise the raw prompt string
    is tokenized directly.

    Truncation keeps the TAIL of the prompt (the generation-relevant suffix) when
    a chat template is applied, and uses the tokenizer's default truncation
    otherwise.

    Use with ``dataset.map()`` (keep the original columns so the reward fn can
    see them — do NOT pass ``remove_columns``)::

        dataset = dataset.map(
            lambda x: tokenize_prompt(x, tokenizer, max_prompt_length=512),
        )
    """
    prompt = row[prompt_key]
    system = row.get(system_key) if system_key else None

    template = getattr(tokenizer, "chat_template", None)
    if apply_chat_template and template and hasattr(tokenizer, "apply_chat_template"):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
        )
        if max_prompt_length is not None and len(input_ids) > max_prompt_length:
            input_ids = input_ids[-max_prompt_length:]
        attention_mask = [1] * len(input_ids)
    else:
        kwargs = {}
        if max_prompt_length is not None:
            kwargs = {"max_length": max_prompt_length, "truncation": True}
        encoded = tokenizer(prompt, **kwargs)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

    out = {"input_ids": input_ids, "attention_mask": attention_mask}
    # Pass through original columns so the reward fn can receive them via **columns.
    for key, value in row.items():
        if key not in out:
            out[key] = value
    return out
