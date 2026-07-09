import inspect
from contextlib import contextmanager

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

_LN2 = -0.6931471805599453  # -ln(2)

# Tokens per chunk for the fused linear+loss path.  At 128k vocab and bf16,
# one chunk of logits is ~256 MB — the peak instead of batch*seq*vocab.
DEFAULT_FUSED_CHUNK_SIZE = 1024


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

    Uses fully vectorized batch operations to maximize GPU utilization.
    """
    idx = safe_labels.unsqueeze(2)  # [B, seq, 1]
    if shift_logits.dtype in (torch.float32, torch.float64):
        gathered = torch.gather(shift_logits, dim=2, index=idx).squeeze(2)
        return gathered - torch.logsumexp(shift_logits, dim=-1)
    else:
        # bf16 / fp16: F.log_softmax upcasts internally for stability
        log_probs = F.log_softmax(shift_logits, dim=-1)
        return torch.gather(log_probs, dim=2, index=idx).squeeze(2)


def per_token_logps_from_logits(logits, labels, label_pad_token_id=-100):
    """Per-token log probabilities [B, seq-1] and the response-token mask.

    Values at masked positions are unspecified — always multiply by the
    returned loss mask before reducing.
    """
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]

    loss_mask = shift_labels != label_pad_token_id
    vocab_size = shift_logits.size(-1)
    safe_labels = torch.where(loss_mask, shift_labels, 0).clamp(max=vocab_size - 1)

    return _per_token_logps(shift_logits, safe_labels), loss_mask


def masked_avg_logps(per_token_logps, loss_mask):
    """Average log probability per sequence over response tokens only."""
    return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1)


def get_batch_logps(logits, labels, label_pad_token_id=-100):
    """Average log probability per sequence over response tokens only.

    Shared by all loss functions that need per-sequence log probabilities
    (ORPO, DPO, SimPO, KTO, CPO, IPO, GRPO, and the reference log prob cache).
    """
    per_token_logps, loss_mask = per_token_logps_from_logits(logits, labels, label_pad_token_id)
    return masked_avg_logps(per_token_logps, loss_mask)


def _unwrap_model(model):
    """Peel accelerate/DDP/torch.compile/PEFT wrappers to reach the base model."""
    m = model
    for _ in range(8):
        if hasattr(m, "get_base_model"):  # PEFT
            m = m.get_base_model()
        elif hasattr(m, "_orig_mod"):  # torch.compile
            m = m._orig_mod
        elif hasattr(m, "module") and isinstance(getattr(m, "module"), torch.nn.Module):  # DDP / FSDP
            m = m.module
        else:
            break
    return m


def _fused_logits_kwarg(model):
    """Name of the model's logits-trimming forward kwarg, or None if the
    fused linear+loss path can't be used with this model.

    The fused path needs the model to (a) expose its lm_head via
    ``get_output_embeddings()`` and (b) accept ``logits_to_keep`` (or the older
    ``num_logits_to_keep``) so the forward pass skips computing full logits.
    """
    base = _unwrap_model(model)
    get_emb = getattr(base, "get_output_embeddings", None)
    if get_emb is None or get_emb() is None:
        return None
    try:
        params = inspect.signature(base.forward).parameters
    except (TypeError, ValueError):
        return None
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in params:
            return name
    return None


def _logit_transform(model):
    """Post-lm_head logit transform declared by the model config, or None.

    Some architectures modify logits after the lm_head projection — Gemma-2/3
    softcapping, Granite ``logits_scaling``, Cohere ``logit_scale``.  The fused
    path bypasses the model's own lm_head call, so it must replay these.
    """
    config = getattr(_unwrap_model(model), "config", None)
    if config is None:
        return None
    if hasattr(config, "get_text_config"):
        config = config.get_text_config()
    softcap = getattr(config, "final_logit_softcapping", None)
    scale_mul = getattr(config, "logit_scale", None)
    scale_div = getattr(config, "logits_scaling", None)
    if softcap is None and scale_mul is None and scale_div is None:
        return None

    def transform(logits):
        if scale_mul is not None:
            logits = logits * scale_mul
        if scale_div is not None:
            logits = logits / scale_div
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap
        return logits

    return transform


def fused_per_token_logps(hidden_states, lm_head, labels, label_pad_token_id=-100,
                          chunk_size=DEFAULT_FUSED_CHUNK_SIZE, logit_transform=None):
    """Per-token log probabilities computed chunk-by-chunk from final hidden
    states, without ever materializing the full [batch, seq, vocab] logits.

    Only response tokens (unmasked labels) are pushed through the lm_head —
    prompt and padding positions never get logits at all.  During training each
    chunk runs under activation checkpointing, so its logits are recomputed in
    backward instead of stored; peak logits memory is one chunk in both passes.

    Returns (per_token_logps [B, seq-1], loss_mask) matching
    ``per_token_logps_from_logits`` (masked positions are zero here).
    """
    shift_hidden = hidden_states[:, :-1, :]
    shift_labels = labels[..., 1:]
    loss_mask = shift_labels != label_pad_token_id

    batch_size, seq_len = shift_labels.shape
    flat_hidden = shift_hidden.reshape(-1, shift_hidden.size(-1))
    token_idx = loss_mask.reshape(-1).nonzero(as_tuple=True)[0]
    sel_hidden = flat_hidden[token_idx]
    sel_labels = shift_labels.reshape(-1)[token_idx]

    head_dtype = getattr(getattr(lm_head, "weight", None), "dtype", None)

    def compute_chunk(h, y):
        if head_dtype is not None and h.dtype != head_dtype:
            h = h.to(head_dtype)
        logits = lm_head(h)  # [chunk, vocab] — the only logits ever created
        if logit_transform is not None:
            logits = logit_transform(logits)
        y = y.unsqueeze(1).clamp(max=logits.size(-1) - 1)
        if logits.dtype in (torch.float32, torch.float64):
            gathered = torch.gather(logits, 1, y).squeeze(1)
            return gathered - torch.logsumexp(logits, dim=-1)
        # bf16 / fp16: F.log_softmax upcasts internally for stability
        return torch.gather(F.log_softmax(logits, dim=-1), 1, y).squeeze(1)

    needs_grad = torch.is_grad_enabled() and (
        sel_hidden.requires_grad or any(p.requires_grad for p in lm_head.parameters())
    )

    chunks = []
    for start in range(0, sel_hidden.size(0), chunk_size):
        h = sel_hidden[start:start + chunk_size]
        y = sel_labels[start:start + chunk_size]
        if needs_grad:
            chunks.append(torch.utils.checkpoint.checkpoint(compute_chunk, h, y, use_reentrant=False))
        else:
            chunks.append(compute_chunk(h, y))

    if chunks:
        values = torch.cat(chunks)
        per_token_logps = shift_hidden.new_zeros(batch_size * seq_len, dtype=values.dtype)
        per_token_logps = per_token_logps.index_put((token_idx,), values)
    else:
        per_token_logps = shift_hidden.new_zeros(batch_size * seq_len)
    return per_token_logps.view(batch_size, seq_len), loss_mask


def forward_per_token_logps(model, input_ids, attention_mask, labels,
                            label_pad_token_id=-100, fused=True,
                            fused_chunk_size=DEFAULT_FUSED_CHUNK_SIZE,
                            position_ids=None):
    """Model forward + per-token log probabilities [B, seq-1] and loss mask.

    When ``fused`` and the model supports it, the forward pass computes logits
    for only the final position (``logits_to_keep=1``) and per-token log-probs
    are computed chunk-by-chunk from the final hidden states — the full
    [batch, seq, vocab] logits tensor is never materialized.  Falls back to
    the standard full-logits path otherwise, so it is always safe to call.
    """
    forward_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, "use_cache": False}
    if position_ids is not None:
        forward_kwargs["position_ids"] = position_ids

    logits_kwarg = _fused_logits_kwarg(model) if fused else None
    if logits_kwarg is not None:
        outputs = model(**forward_kwargs, output_hidden_states=True, **{logits_kwarg: 1})
        hidden_states = outputs.hidden_states[-1]
        lm_head = _unwrap_model(model).get_output_embeddings()
        return fused_per_token_logps(
            hidden_states, lm_head, labels, label_pad_token_id,
            chunk_size=fused_chunk_size, logit_transform=_logit_transform(model),
        )

    logits = model(**forward_kwargs).logits
    return per_token_logps_from_logits(logits, labels, label_pad_token_id)


def pad_dim1(tensor, length, value):
    """Pad a 2D tensor along dim=1 (sequence length) to the target length."""
    if tensor.size(1) >= length:
        return tensor
    return F.pad(tensor, (0, length - tensor.size(1)), value=value)


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
