"""LoRA on frozen FP8/FP4 block-quantized weights (transformers' fine-grained FP8 modules).

Checkpoints like DeepSeek-V4 ship their linears as FP8 (e4m3, 128x128 block scales) and
their routed experts as FP4 (e2m1 packed two per int8, one UE8M0 scale per 32 inputs).
transformers loads them into ``FP8Linear`` / ``FP8GroupedLinear`` / ``FP8Experts``, whose
Triton/DeepGEMM kernels have no backward, and marks the quantizer untrainable. The only
alternative was dequantizing the whole model to bf16 (~4x the memory for FP4 experts).

LoRA never needs gradients for the frozen weights, only for the activations flowing
through them: dx = dy @ W. So each quantized matmul is wrapped in an autograd function
whose forward is the original kernel and whose backward dequantizes W (one matrix at a
time, transiently) and computes dx in the gradient's dtype. Memory stays at the quantized
size; the backward costs one dequant per weight per step.

Routed experts get a grouped path instead of transformers' eager per-expert loop (a few
hundred small kernel launches per layer, three times per step under checkpointing): pairs
are sorted by expert, the forward is one fused grouped matmul per projection, and the
backward dequantizes ``_EXPERT_CHUNK`` experts at a time and runs one grouped bf16 matmul
per chunk.

PEFT also casts each adapter to its base layer's dtype when the base is floating point,
and float8 counts, so LoRA weights on an FP8 linear would be created in FP8.
``fix_lora_dtypes`` moves them back to bf16 and re-initializes them.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_FP4_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
_QUANT_LINEARS = ("FP8Linear",)
_QUANT_GROUPED = ("FP8GroupedLinear",)
_QUANT_EXPERTS = ("FP8Experts",)
_EXPERT_CHUNK = 32           # experts dequantized together in the grouped path (~1.6 GB bf16 on DSV4-Flash)
_state = {"kernel": True, "grouped_mm": True}


def _fp8_dtypes() -> tuple:
    return tuple(getattr(torch, n) for n in ("float8_e4m3fn", "float8_e5m2", "float8_e8m0fnu")
                 if hasattr(torch, n))


def is_quantized_weight(w: torch.Tensor) -> bool:
    return w.element_size() == 1


def dequantize(weight: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Block-scaled FP8 (or int8-packed FP4) weight -> dense ``dtype``.

    The block size comes from the scale grid: (rows / scale_rows, cols / scale_cols). Same
    math as transformers' ``Fp8Dequantize`` so forward and backward agree.
    """
    if weight.dtype == torch.int8 or weight.dtype == torch.uint8:
        lut = torch.tensor(_FP4_E2M1, dtype=torch.float32, device=weight.device)
        u8 = weight.contiguous().view(torch.uint8)
        w = torch.stack([lut[(u8 & 0xF).long()], lut[((u8 >> 4) & 0xF).long()]], dim=-1)
        w = w.reshape(*weight.shape[:-1], 2 * weight.shape[-1])
    else:
        w = weight.to(torch.float32)
    rows, cols = w.shape[-2:]
    if scale.dim() < 2:
        return (w * scale.to(torch.float32)).to(dtype)
    s_rows, s_cols = scale.shape[-2:]
    if rows % s_rows or cols % s_cols:
        raise ValueError(f"weight {tuple(w.shape)} not divisible by scale grid {tuple(scale.shape)}")
    s = scale.to(torch.float32)
    q = w.reshape(-1, s_rows, rows // s_rows, s_cols, cols // s_cols)
    out = q * s.reshape(-1, s_rows, s_cols)[:, :, None, :, None]
    return out.reshape(w.shape).to(dtype)


class _FrozenQuantLinear(torch.autograd.Function):
    """y = kernel(x); dx = dy @ dequant(W). No gradient for W (it is frozen)."""

    @staticmethod
    def forward(ctx, x, weight, scale, kernel):
        ctx.save_for_backward(weight, scale)
        with torch.no_grad():
            return kernel(x)

    @staticmethod
    def backward(ctx, grad_out):
        weight, scale = ctx.saved_tensors
        w = dequantize(weight, scale, grad_out.dtype)
        return grad_out @ w, None, None, None


class _FrozenQuantGrouped(torch.autograd.Function):
    """Block-diagonal grouped linear: x (..., g, in) -> y (..., g, out), per-group weights."""

    @staticmethod
    def forward(ctx, x, weight, scale, n_groups, kernel):
        ctx.save_for_backward(weight, scale)
        ctx.n_groups = n_groups
        with torch.no_grad():
            return kernel(x)

    @staticmethod
    def backward(ctx, grad_out):
        weight, scale = ctx.saved_tensors
        w = dequantize(weight, scale, grad_out.dtype)
        w = w.view(ctx.n_groups, -1, w.shape[-1])            # (g, out, in)
        return torch.einsum("...go,goi->...gi", grad_out, w), None, None, None, None


def _grouped_dense(x, weight, scale, counts, transpose):
    """Per-expert matmul over expert-sorted rows of ``x`` against dequantized expert weights.

    ``counts[e]`` rows belong to expert ``e``. ``transpose=True``: x_e @ W_e^T (forward);
    ``False``: x_e @ W_e (input gradient). Only experts with rows are dequantized, in chunks.
    """
    rows = weight.shape[-2]
    cols = weight.shape[-1] * (2 if weight.dtype in (torch.int8, torch.uint8) else 1)
    out = x.new_empty(x.shape[0], rows if transpose else cols)
    hit = [e for e, c in enumerate(counts) if c]
    starts, pos = {}, 0
    for e, c in enumerate(counts):
        starts[e] = pos
        pos += c
    for i in range(0, len(hit), _EXPERT_CHUNK):
        idx = hit[i:i + _EXPERT_CHUNK]
        sel = torch.tensor(idx, device=weight.device)
        w = dequantize(weight.index_select(0, sel), scale.index_select(0, sel), x.dtype)   # (c, rows, cols)
        lo, hi = starts[idx[0]], starts[idx[-1]] + counts[idx[-1]]
        if _state["grouped_mm"] and x.is_cuda and hasattr(torch, "_grouped_mm"):
            try:
                offs = torch.tensor([counts[e] for e in idx], device=x.device).cumsum(0).to(torch.int32)
                b = w.transpose(-2, -1) if transpose else w
                out[lo:hi] = torch._grouped_mm(x[lo:hi].contiguous(), b, offs=offs)
                continue
            except (RuntimeError, TypeError, NotImplementedError) as e:
                _state["grouped_mm"] = False
                logger.warning(f"torch._grouped_mm unavailable ({e}); using a per-expert loop")
        for j, e in enumerate(idx):
            a, b = starts[e], starts[e] + counts[e]
            out[a:b] = x[a:b] @ (w[j].T if transpose else w[j])
    return out


class _FrozenQuantExperts(torch.autograd.Function):
    """Grouped expert projection on expert-sorted rows; dx = per-expert dy @ dequant(W_e)."""

    @staticmethod
    def forward(ctx, x, weight, scale, counts, matmul):
        ctx.save_for_backward(weight, scale)
        ctx.counts = counts
        with torch.no_grad():
            return matmul(x)

    @staticmethod
    def backward(ctx, grad_out):
        weight, scale = ctx.saved_tensors
        return _grouped_dense(grad_out.contiguous(), weight, scale, ctx.counts, transpose=False), None, None, None, None


def _expert_matmul(module, weight, scale, counts, counts_t):
    """Forward matmul for one expert projection: the fused FP8/FP4 kernel on GPU, else dequant."""
    dense = lambda x: _grouped_dense(x, weight, scale, counts, transpose=True)  # noqa: E731
    if not (_state["kernel"] and weight.is_cuda):
        return dense
    try:
        from transformers.integrations.finegrained_fp8 import load_finegrained_fp8_kernel
        kernel = load_finegrained_fp8_kernel()
    except Exception as e:
        _state["kernel"] = False
        logger.warning(f"Fine-grained FP8 kernel unavailable ({e}); experts use dequant + grouped matmul")
        return dense
    tokens_per_expert = counts_t.to(torch.int32)
    offsets = torch.cumsum(tokens_per_expert, 0, dtype=torch.int32)

    def fused(x):
        try:
            return kernel.grouped_matmul(x, weight, scale, offsets=offsets, tokens_per_expert=tokens_per_expert,
                                         block_size=getattr(module, "block_size", None))
        except Exception as e:
            _state["kernel"] = False
            logger.warning(f"Fused grouped FP8/FP4 matmul failed ({e}); experts use dequant + grouped matmul")
            return dense(x)
    return fused


def _grouped_experts_forward(module, hidden_states, top_k_index, top_k_weights):
    num_tokens, hidden = hidden_states.shape
    top_k = top_k_index.size(-1)
    expert_ids, perm = torch.sort(top_k_index.reshape(-1), stable=True)
    counts_t = torch.bincount(expert_ids, minlength=module.num_experts)
    counts = counts_t.tolist()
    x = hidden_states[perm // top_k]

    if module.has_gate:
        w_up, s_up = module.gate_up_proj.detach(), module.gate_up_proj_scale_inv.detach()
    else:
        w_up, s_up = module.up_proj.detach(), module.up_proj_scale_inv.detach()
    h = _FrozenQuantExperts.apply(x, w_up, s_up, counts, _expert_matmul(module, w_up, s_up, counts, counts_t))
    h = module._apply_gate(h) if module.has_gate else module.act_fn(h)
    w_dn, s_dn = module.down_proj.detach(), module.down_proj_scale_inv.detach()
    y = _FrozenQuantExperts.apply(h, w_dn, s_dn, counts, _expert_matmul(module, w_dn, s_dn, counts, counts_t))

    y = y.float() * top_k_weights.reshape(-1)[perm].float().unsqueeze(-1)
    out = torch.zeros(num_tokens * top_k, hidden, dtype=y.dtype, device=y.device).index_copy(0, perm, y)
    return out.view(num_tokens, top_k, hidden).sum(dim=1).to(hidden_states.dtype)


def _wrap_linear(module):
    original = module.forward

    def forward(x):
        if not is_quantized_weight(module.weight) or not torch.is_grad_enabled():
            return original(x)
        return _FrozenQuantLinear.apply(x, module.weight, module.weight_scale_inv, original)

    module.forward = forward


def _wrap_grouped(module):
    original = module.forward

    def forward(x):
        if not is_quantized_weight(module.weight) or not torch.is_grad_enabled():
            return original(x)
        return _FrozenQuantGrouped.apply(x, module.weight, module.weight_scale_inv, module.n_groups, original)

    module.forward = forward


def _wrap_experts(module):
    original = module.linear

    def linear(input, weight, weight_scale_inv, activation_scale=None):
        if not is_quantized_weight(weight) or not torch.is_grad_enabled():
            return original(input, weight, weight_scale_inv, activation_scale=activation_scale)
        return _FrozenQuantLinear.apply(
            input, weight, weight_scale_inv,
            lambda t: original(t, weight, weight_scale_inv, activation_scale=activation_scale))

    module.linear = linear

    if not all(hasattr(module, a) for a in ("down_proj", "down_proj_scale_inv", "num_experts", "has_gate")):
        return
    original_forward = module.forward

    def forward(hidden_states, top_k_index, top_k_weights):
        if (not is_quantized_weight(module.down_proj) or getattr(module, "activation_scheme", "dynamic") != "dynamic"
                or getattr(module, "_is_expert_parallel", False)):
            return original_forward(hidden_states, top_k_index, top_k_weights)
        return _grouped_experts_forward(module, hidden_states, top_k_index, top_k_weights)

    module.forward = forward


def enable_quantized_backward(model: torch.nn.Module) -> int:
    """Make the frozen quantized modules of ``model`` pass activation gradients. Returns how many.

    Experts with the standard fused layout (``gate_up_proj``/``down_proj`` + scales) get
    grimoire's grouped path; anything else falls back to transformers' eager per-expert
    loop (the one that calls ``linear``), so the model's experts implementation is set to
    eager for that fallback.
    """
    count = 0
    for module in model.modules():
        if getattr(module, "_grimoire_quant_backward", False):
            continue
        names = {c.__name__ for c in type(module).__mro__}
        if names & set(_QUANT_GROUPED):
            _wrap_grouped(module)
        elif names & set(_QUANT_LINEARS):
            _wrap_linear(module)
        elif names & set(_QUANT_EXPERTS):
            _wrap_experts(module)
        else:
            continue
        module._grimoire_quant_backward = True
        count += 1
    if count:
        if hasattr(model, "set_experts_implementation"):
            try:
                model.set_experts_implementation("eager")
            except Exception as e:  # older transformers, or no experts
                logger.debug(f"set_experts_implementation('eager') failed: {e}")
        logger.info(f"Quantized backward enabled on {count} FP8/FP4 modules")
    return count


def fix_lora_dtypes(model: torch.nn.Module, dtype: torch.dtype = torch.bfloat16) -> int:
    """Move LoRA weights PEFT created in FP8 (cast to an FP8 base's dtype) to ``dtype`` and re-init."""
    fp8 = _fp8_dtypes()
    fixed = 0
    for module in model.modules():
        if not hasattr(module, "reset_lora_parameters") or not hasattr(module, "lora_A"):
            continue
        for adapter in list(module.lora_A.keys()):
            params = list(module.lora_A[adapter].parameters()) + list(module.lora_B[adapter].parameters())
            if not any(p.dtype in fp8 for p in params):
                continue
            module.lora_A[adapter].to(dtype)
            module.lora_B[adapter].to(dtype)
            cfg = (getattr(model, "peft_config", None) or {}).get(adapter)
            module.reset_lora_parameters(adapter, getattr(cfg, "init_lora_weights", True))
            fixed += 1
    if fixed:
        logger.info(f"Re-created {fixed} LoRA adapters in {dtype} (PEFT had cast them to FP8)")
    return fixed
