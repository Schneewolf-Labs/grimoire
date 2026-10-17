PROMPT_TRUNCATION_MODES = ("keep_end", "keep_start")


def truncate_prompt_ids(prompt_ids, max_prompt_length, mode="keep_end", bos_token_id=None):
    """Cut a tokenized prompt down to ``max_prompt_length`` tokens.

    ``mode="keep_end"`` (the default everywhere in grimoire) drops tokens from
    the *start* of the prompt and keeps its end. The end of a chat-formatted
    prompt is the part the response is conditioned on — the latest user turn,
    the tool results, and the assistant generation header that the response
    must follow. Cutting from the right instead (``"keep_start"``) keeps the
    system prompt but throws all of that away, so the model is trained to
    answer a conversation it never saw, starting mid-template. A leading BOS
    token is preserved in ``keep_end`` mode so the sequence still starts the
    way the model expects.

    ``"keep_start"`` is the previous behavior, kept for callers that need it.
    """
    if mode not in PROMPT_TRUNCATION_MODES:
        raise ValueError(
            f"prompt_truncation must be one of {PROMPT_TRUNCATION_MODES}, got {mode!r}"
        )
    if not max_prompt_length or len(prompt_ids) <= max_prompt_length:
        return prompt_ids
    if mode == "keep_start":
        return prompt_ids[:max_prompt_length]

    head = []
    if (
        bos_token_id is not None
        and max_prompt_length > 1
        and prompt_ids
        and prompt_ids[0] == bos_token_id
    ):
        head = prompt_ids[:1]
    keep = max_prompt_length - len(head)
    return head + prompt_ids[len(prompt_ids) - keep:]


def encode_prompt_response(
    tokenizer,
    prompt,
    response,
    max_length=None,
    max_prompt_length=None,
    label_pad_token_id=-100,
    prompt_truncation="keep_end",
):
    """Tokenize a prompt + response pair with an exact prompt/response boundary.

    The prompt is tokenized with special tokens (BOS etc.) and the response
    without, so the label mask covers exactly the prompt tokens. Tokenizing
    ``prompt + response`` as a single string and measuring the prompt
    separately is off by one whenever the tokenizer prepends BOS, and can
    misalign when the tokenizer merges tokens across the boundary.

    Args:
        max_length: Maximum total sequence length (prompt + response). When
            the pair is longer, the end of the response is cut.
        max_prompt_length: Maximum prompt length in tokens. Longer prompts
            are truncated so more of the response is preserved for training.
        label_pad_token_id: Label value for masked (prompt) positions.
        prompt_truncation: Which part of an over-long prompt survives:
            ``"keep_end"`` (default) keeps the final tokens — the latest turns
            and the generation header the response follows — plus a leading
            BOS; ``"keep_start"`` keeps the first tokens. See
            :func:`truncate_prompt_ids`.
    """
    prompt_ids = list(tokenizer(prompt)["input_ids"])
    # Some tokenizers append EOS to standalone text — drop it, the response
    # continues the sequence.
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None and prompt_ids and prompt_ids[-1] == eos_id:
        prompt_ids = prompt_ids[:-1]

    response_ids = list(tokenizer(response, add_special_tokens=False)["input_ids"])

    prompt_ids = truncate_prompt_ids(
        prompt_ids,
        max_prompt_length,
        mode=prompt_truncation,
        bos_token_id=getattr(tokenizer, "bos_token_id", None),
    )

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
