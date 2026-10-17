"""Memory-bounded eager attention for models that only support eager (e.g. DeepSeek-V4).

DeepSeek-V4's attention mixes a sliding window, compressed KV entries with a per-query
block bias, and learned sink logits, so transformers ships it eager-only. Eager
attention materializes the full (heads x seq x kv) logits several times over (logits,
logits + sink, max-subtracted, softmax). With 64 heads at 6k tokens that is several
~6 GB tensors per layer, which runs out of memory on long rows even when the weights fit.

``chunked_eager_attention`` computes the same math one block of queries at a time and
checkpoints each block, so a layer's backward recomputes one block's softmax at a time
instead of keeping all of them. Shared KV heads are broadcast instead of repeated (MQA
with 64 query heads would otherwise copy K and V 64 times).
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)

ATTN_IMPLEMENTATION = "grimoire_chunked"
CHUNKED_ATTENTION_MODEL_TYPES = ("deepseek_v4",)
_LOGIT_BUDGET = 1 << 27          # logits per block (heads x queries x keys), ~256 MB in bf16


def _attention_block(query, key, value, mask, sinks, scaling):
    """Eager attention with optional sink logits for one block of queries.

    query (B, H, q, D); key/value (B, Hkv, k, D) with H a multiple of Hkv.
    """
    b, h, q, d = query.shape
    hkv = key.shape[1]
    qg = query.reshape(b, hkv, (h // hkv) * q, d)            # heads of one KV group are contiguous
    logits = torch.matmul(qg, key.transpose(2, 3)).view(b, h, q, -1) * scaling
    if mask is not None:
        logits = logits + mask
    if sinks is not None:
        logits = torch.cat([logits, sinks.reshape(1, -1, 1, 1).expand(b, -1, q, 1).to(logits.dtype)], dim=-1)
    logits = logits - logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(logits, dim=-1, dtype=logits.dtype)
    if sinks is not None:
        probs = probs[..., :-1]
    probs = probs.to(value.dtype).reshape(b, hkv, (h // hkv) * q, -1)
    return torch.matmul(probs, value).view(b, h, q, d)


def chunked_eager_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, s_aux=None, **kwargs):
    """Drop-in for an eager attention forward; returns (output (B, S, H, D), None)."""
    if dropout and module.training:
        raise ValueError("chunked eager attention does not support attention dropout")
    sinks = s_aux if s_aux is not None else getattr(module, "sinks", None)
    b, h, s, _ = query.shape
    block = max(16, _LOGIT_BUDGET // max(1, b * h * key.shape[2]))
    outs = []
    for i in range(0, s, block):
        mask = attention_mask
        if mask is not None and mask.shape[-2] > 1:
            mask = mask[..., i:i + block, :]
        args = (query[:, :, i:i + block], key, value, mask, sinks, scaling)
        if torch.is_grad_enabled() and s > block:
            outs.append(checkpoint(_attention_block, *args, use_reentrant=False))
        else:
            outs.append(_attention_block(*args))
    out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=2)
    return out.transpose(1, 2).contiguous(), None


def _register() -> None:
    from transformers import AttentionInterface
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface

    AttentionInterface.register(ATTN_IMPLEMENTATION, chunked_eager_attention)
    AttentionMaskInterface.register(ATTN_IMPLEMENTATION, ALL_MASK_ATTENTION_FUNCTIONS["eager"])


def enable_chunked_attention(model: torch.nn.Module) -> bool:
    """Switch an eager-only model of a known type to chunked eager attention. Returns whether it did."""
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) not in CHUNKED_ATTENTION_MODEL_TYPES:
        return False
    if getattr(config, "_attn_implementation", "eager") not in ("eager", None, ATTN_IMPLEMENTATION):
        return False
    _register()
    for module in model.modules():
        cfg = getattr(module, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = ATTN_IMPLEMENTATION
    logger.info(f"Using chunked eager attention for {config.model_type}")
    return True
