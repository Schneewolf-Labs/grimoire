from contextlib import contextmanager

import torch
import torch.nn.functional as F

_LN2 = -0.6931471805599453  # -ln(2)


@contextmanager
def _disable_grad_checkpointing(model):
    """Temporarily disable gradient checkpointing for a reference forward pass.

    Gradient checkpointing + torch.no_grad() + quantized models (bitsandbytes
    4-bit/8-bit) causes CUDA illegal memory access.  This mirrors the
    workaround in GrimoireTrainer.evaluate().
    """
    was_enabled = getattr(model, "is_gradient_checkpointing", False)
    if was_enabled:
        model.gradient_checkpointing_disable()
    try:
        yield
    finally:
        if was_enabled:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )


def _log1mexp(x):
    """Numerically stable log(1 - exp(x)) for x <= 0.

    Uses two branches to avoid catastrophic cancellation:
    - x < -ln(2): log1p(-exp(x))   — exp(x) < 0.5, so 1-exp(x) > 0.5
    - x >= -ln(2): log(-expm1(x))  — expm1 is accurate near 0

    Matches the approach used by TRL and recommended by Machler (2012).
    """
    return torch.where(
        x < _LN2,
        torch.log1p(-torch.exp(x)),
        torch.log(-torch.expm1(x)),
    )


def _per_token_logps(shift_logits, safe_labels):
    """Compute per-token log probabilities with dtype-aware numerics.

    For float32/64, uses gather + logsumexp (fast, numerically stable).
    For bf16/fp16, falls back to F.log_softmax which internally upcasts
    to float32 — torch.logsumexp on half-precision with large vocab dims
    (e.g. 152k) can produce subtly wrong values that lead to NaN gradients
    and CUDA errors downstream.  This matches TRL's selective_log_softmax.

    Processes each row individually so CUDA kernels always get contiguous
    memory (logits[:, :-1, :] is a non-contiguous view, but each row is
    contiguous).
    """
    rows = []
    if shift_logits.dtype in (torch.float32, torch.float64):
        for i in range(shift_logits.size(0)):
            row_logits = shift_logits[i]
            row_labels = safe_labels[i]
            rows.append(
                torch.gather(row_logits, dim=1, index=row_labels.unsqueeze(1)).squeeze(1)
                - torch.logsumexp(row_logits, dim=-1)
            )
    else:
        # bf16 / fp16: F.log_softmax upcasts internally for stability
        for i in range(shift_logits.size(0)):
            row_logps = F.log_softmax(shift_logits[i], dim=-1)
            rows.append(
                torch.gather(row_logps, dim=1, index=safe_labels[i].unsqueeze(1)).squeeze(1)
            )
    return torch.stack(rows)


def get_batch_logps(logits, labels, label_pad_token_id=-100):
    """Average log probability per sequence over response tokens only.

    Shared by all loss functions that need per-sequence log probabilities
    (ORPO, DPO, SimPO, KTO, CPO, IPO, GRPO, and the reference log prob cache).
    """
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]

    loss_mask = shift_labels != label_pad_token_id
    vocab_size = shift_logits.size(-1)
    safe_labels = torch.where(loss_mask, shift_labels, 0).clamp(max=vocab_size - 1)

    per_token_logps = _per_token_logps(shift_logits, safe_labels)

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
