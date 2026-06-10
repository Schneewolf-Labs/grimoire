def encode_prompt_response(
    tokenizer,
    prompt,
    response,
    max_length=None,
    max_prompt_length=None,
    label_pad_token_id=-100,
):
    """Tokenize a prompt + response pair with an exact prompt/response boundary.

    The prompt is tokenized with special tokens (BOS etc.) and the response
    without, so the label mask covers exactly the prompt tokens. Tokenizing
    ``prompt + response`` as a single string and measuring the prompt
    separately is off by one whenever the tokenizer prepends BOS, and can
    misalign when the tokenizer merges tokens across the boundary.

    Args:
        max_length: Maximum total sequence length (prompt + response).
        max_prompt_length: Maximum prompt length in tokens. Longer prompts
            are truncated so more of the response is preserved for training.
        label_pad_token_id: Label value for masked (prompt) positions.
    """
    prompt_ids = list(tokenizer(prompt)["input_ids"])
    # Some tokenizers append EOS to standalone text — drop it, the response
    # continues the sequence.
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None and prompt_ids and prompt_ids[-1] == eos_id:
        prompt_ids = prompt_ids[:-1]

    response_ids = list(tokenizer(response, add_special_tokens=False)["input_ids"])

    if max_prompt_length and len(prompt_ids) > max_prompt_length:
        prompt_ids = prompt_ids[:max_prompt_length]

    input_ids = prompt_ids + response_ids
    if max_length:
        input_ids = input_ids[:max_length]

    prompt_len = min(len(prompt_ids), len(input_ids))
    labels = [label_pad_token_id] * prompt_len + input_ids[prompt_len:]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }
