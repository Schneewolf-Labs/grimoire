import torch


def get_batch_logps(logits, labels, label_pad_token_id=-100):
    """Average log probability per sequence over response tokens only.

    Shared by all loss functions that need per-sequence log probabilities
    (ORPO, DPO, SimPO, KTO, CPO, IPO, GRPO, and the reference log prob cache).

    Processes each sequence individually so that gather + logsumexp always
    operate on contiguous memory — ``logits[:, :-1, :]`` is a non-contiguous
    view, but each row ``logits[i, :-1, :]`` is contiguous.
    """
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]

    loss_mask = shift_labels != label_pad_token_id
    vocab_size = shift_logits.size(-1)
    safe_labels = torch.where(loss_mask, shift_labels, 0).clamp(max=vocab_size - 1)

    # Per-row gather + logsumexp for contiguous CUDA kernel inputs
    per_token_logps = torch.zeros_like(loss_mask, dtype=logits.dtype)
    for i in range(shift_logits.size(0)):
        row_logits = shift_logits[i]
        row_labels = safe_labels[i]
        per_token_logps[i] = (
            torch.gather(row_logits, dim=1, index=row_labels.unsqueeze(1)).squeeze(1)
            - torch.logsumexp(row_logits, dim=-1)
        )

    return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1)


def pad_dim1(tensor, length, value):
    """Pad a 2D tensor along dim=1 (sequence length) to the target length."""
    if tensor.size(1) >= length:
        return tensor
    pad = torch.full(
        (tensor.size(0), length - tensor.size(1)),
        value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor, pad], dim=1)


def concatenate_preference(batch, pad_token_id, label_pad_token_id=-100):
    """Concatenate chosen and rejected into a single batch, padding to equal length.

    Shared by ORPO, DPO, SimPO, CPO, and IPO loss functions.
    """
    max_len = max(batch["chosen_input_ids"].size(1), batch["rejected_input_ids"].size(1))

    input_ids = torch.cat([
        pad_dim1(batch["chosen_input_ids"], max_len, pad_token_id),
        pad_dim1(batch["rejected_input_ids"], max_len, pad_token_id),
    ], dim=0)

    attention_mask = torch.cat([
        pad_dim1(batch["chosen_attention_mask"], max_len, 0),
        pad_dim1(batch["rejected_attention_mask"], max_len, 0),
    ], dim=0)

    labels = torch.cat([
        pad_dim1(batch["chosen_labels"], max_len, label_pad_token_id),
        pad_dim1(batch["rejected_labels"], max_len, label_pad_token_id),
    ], dim=0)

    return input_ids, attention_mask, labels
