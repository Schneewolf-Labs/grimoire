"""Backprop through frozen FP8/FP4 modules (grimoire/quantized.py), on CPU.

transformers' FP8 kernels are GPU-only and have no backward, so the modules here stand in
for them: same class names and parameter layout, forward computed under no_grad (so
autograd can't see through it, exactly like the real kernels). Gradients are checked
against the same network built from the dequantized dense weights.
"""
import pytest
import torch
from torch import nn

from grimoire.quantized import dequantize, enable_quantized_backward, fix_lora_dtypes

FP8 = getattr(torch, "float8_e4m3fn", None)
pytestmark = pytest.mark.skipif(FP8 is None, reason="torch without float8")


def quantize_fp8(w, block=(4, 4)):
    rows, cols = w.shape
    br, bc = block
    blocks = w.reshape(rows // br, br, cols // bc, bc)
    scale = blocks.abs().amax(dim=(1, 3)) / 448.0 + 1e-8
    q = (blocks / scale[:, None, :, None]).reshape(rows, cols).to(FP8)
    return q, scale.float()


def quantize_fp4(w, group=8):
    """Nearest e2m1 per value, one power-of-two scale per `group` inputs, two nibbles per byte."""
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    rows, cols = w.shape
    g = w.reshape(rows, cols // group, group)
    scale = torch.exp2(torch.ceil(torch.log2(g.abs().amax(-1) / 6.0 + 1e-12)))
    x = g / scale[..., None]
    idx = (x.abs()[..., None] - lut).abs().argmin(-1) + 8 * (x < 0)
    idx = idx.reshape(rows, cols).to(torch.uint8)
    packed = (idx[:, 0::2] | (idx[:, 1::2] << 4)).view(torch.int8)
    return packed, scale.float()


class FP8Linear(nn.Linear):
    def __init__(self, w, block=(4, 4)):
        super().__init__(w.shape[1], w.shape[0], bias=False)
        q, s = quantize_fp8(w, block)
        self.weight = nn.Parameter(q, requires_grad=False)
        self.weight_scale_inv = nn.Parameter(s, requires_grad=False)

    def forward(self, x):
        with torch.no_grad():                       # the real kernel has no autograd
            return x @ dequantize(self.weight, self.weight_scale_inv, x.dtype).T


class FP8GroupedLinear(FP8Linear):
    def __init__(self, w, n_groups):
        super().__init__(w)
        self.n_groups = n_groups

    def forward(self, x):
        with torch.no_grad():
            w = dequantize(self.weight, self.weight_scale_inv, x.dtype).view(self.n_groups, -1, x.shape[-1])
            return torch.einsum("...gi,goi->...go", x, w)


class FP8Experts(nn.Module):
    """Two FP4 experts, each a single projection; the eager loop calls `linear`."""

    def __init__(self, ws):
        super().__init__()
        packed = [quantize_fp4(w) for w in ws]
        self.proj = nn.Parameter(torch.stack([p for p, _ in packed]), requires_grad=False)
        self.proj_scale_inv = nn.Parameter(torch.stack([s for _, s in packed]), requires_grad=False)

    def linear(self, input, weight, weight_scale_inv, activation_scale=None):
        with torch.no_grad():
            return input @ dequantize(weight, weight_scale_inv, input.dtype).T

    def forward(self, x):
        return sum(self.linear(x, self.proj[i], self.proj_scale_inv[i]) for i in range(self.proj.shape[0]))


def test_fp4_dequant_matches_transformers():
    tr = pytest.importorskip("transformers.integrations.finegrained_fp8")
    import inspect
    fn = getattr(getattr(tr, "Fp8Dequantize", None), "_dequantize_one", None)
    if fn is None or "output_dtype" not in inspect.signature(fn).parameters:
        pytest.skip("transformers without FP4-aware Fp8Dequantize (5.20+)")
    w = torch.randn(4, 16)
    packed, scale = quantize_fp4(w)
    ours = dequantize(packed, scale, torch.float32)
    theirs = tr.Fp8Dequantize(None)._dequantize_one(packed, scale, output_dtype=torch.float32)
    assert torch.equal(ours, theirs)
    assert (ours - w).abs().max() < w.abs().max() * 0.35        # quantized, but the same matrix


def test_activation_grads_match_dense_reference():
    torch.manual_seed(0)
    w1, wg, we = torch.randn(8, 8), torch.randn(8, 4), [torch.randn(8, 8), torch.randn(8, 8)]
    lin, grp, exp = FP8Linear(w1), FP8GroupedLinear(wg, n_groups=2), FP8Experts(we)
    x = torch.randn(3, 8, requires_grad=True)

    def run(x):
        h = lin(x)                                   # (3, 8)
        h = grp(h.view(3, 2, 4)).reshape(3, 8)       # two groups of 4 -> 4
        return exp(h).pow(2).sum()

    with pytest.raises(RuntimeError):                # kernels are opaque to autograd
        run(x).backward()

    assert enable_quantized_backward(nn.ModuleList([lin, grp, exp])) == 3
    run(x).backward()

    d1 = dequantize(lin.weight, lin.weight_scale_inv, torch.float32)
    dg = dequantize(grp.weight, grp.weight_scale_inv, torch.float32).view(2, 4, 4)
    de = [dequantize(exp.proj[i], exp.proj_scale_inv[i], torch.float32) for i in range(2)]
    xr = x.detach().clone().requires_grad_(True)
    h = xr @ d1.T
    h = torch.einsum("bgi,goi->bgo", h.view(3, 2, 4), dg).reshape(3, 8)
    sum(h @ d.T for d in de).pow(2).sum().backward()
    assert torch.allclose(x.grad, xr.grad, atol=1e-4, rtol=1e-4)


def test_no_grad_path_is_untouched():
    lin = FP8Linear(torch.randn(8, 8))
    enable_quantized_backward(lin)
    with torch.no_grad():
        y = lin(torch.randn(2, 8))
    assert y.shape == (2, 8) and not y.requires_grad


def test_lora_on_fp8_base_trains_in_bf16():
    peft = pytest.importorskip("peft")
    torch.manual_seed(0)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = FP8Linear(torch.randn(8, 8))

        def forward(self, x):
            return self.proj(x)

    net = Net()
    enable_quantized_backward(net)
    model = peft.get_peft_model(net, peft.LoraConfig(r=2, lora_alpha=4, target_modules=["proj"]))
    fixed = fix_lora_dtypes(model)           # 1 on peft versions that cast adapters to FP8, 0 on newer ones
    layer = model.base_model.model.proj
    dtype = layer.lora_A["default"].weight.dtype
    assert dtype in (torch.bfloat16, torch.float32) and (fixed == 0 or dtype == torch.bfloat16)
    assert layer.lora_A["default"].weight.abs().sum() > 0 and layer.lora_B["default"].weight.abs().sum() == 0
    model(torch.randn(4, 8, dtype=dtype)).float().pow(2).sum().backward()
    grads = {n for n, p in model.named_parameters() if p.grad is not None}
    assert grads and all("lora_" in n for n in grads)


class FP8GatedExperts(FP8Experts):
    """transformers' fused FP4 experts layout: gate_up_proj (E, 2I, H), down_proj (E, H, I), eager loop."""

    def __init__(self, n_experts=6, hidden=8, inter=8):
        nn.Module.__init__(self)
        self.num_experts, self.has_gate, self.block_size, self.activation_scheme = n_experts, True, None, "dynamic"
        self.act_fn = nn.functional.silu
        up = [quantize_fp4(torch.randn(2 * inter, hidden)) for _ in range(n_experts)]
        down = [quantize_fp4(torch.randn(hidden, inter)) for _ in range(n_experts)]
        self.gate_up_proj = nn.Parameter(torch.stack([p for p, _ in up]), requires_grad=False)
        self.gate_up_proj_scale_inv = nn.Parameter(torch.stack([s for _, s in up]), requires_grad=False)
        self.down_proj = nn.Parameter(torch.stack([p for p, _ in down]), requires_grad=False)
        self.down_proj_scale_inv = nn.Parameter(torch.stack([s for _, s in down]), requires_grad=False)

    def _apply_gate(self, gate_up):
        gate, up = gate_up.chunk(2, dim=-1)
        return self.act_fn(gate) * up

    def linear(self, input, weight, weight_scale_inv, activation_scale=None):
        with torch.no_grad():
            return input @ dequantize(weight, weight_scale_inv, input.dtype).T

    def forward(self, hidden_states, top_k_index, top_k_weights):
        out = torch.zeros_like(hidden_states)
        for e in range(self.num_experts):
            tok, pos = torch.where(top_k_index == e)
            if len(tok) == 0:
                continue
            h = self._apply_gate(self.linear(hidden_states[tok], self.gate_up_proj[e], self.gate_up_proj_scale_inv[e]))
            h = self.linear(h, self.down_proj[e], self.down_proj_scale_inv[e])
            out.index_add_(0, tok, h * top_k_weights[tok, pos, None])
        return out

    def dense(self, x, top_k_index, top_k_weights):
        out = torch.zeros_like(x)
        for e in range(self.num_experts):
            tok, pos = torch.where(top_k_index == e)
            gu = dequantize(self.gate_up_proj[e], self.gate_up_proj_scale_inv[e], x.dtype)
            dn = dequantize(self.down_proj[e], self.down_proj_scale_inv[e], x.dtype)
            out = out.index_add(0, tok, (self._apply_gate(x[tok] @ gu.T) @ dn.T) * top_k_weights[tok, pos, None])
        return out


def test_grouped_experts_match_eager_and_dense_grads(monkeypatch):
    import grimoire.quantized as q
    monkeypatch.setattr(q, "_EXPERT_CHUNK", 2)                  # several chunks, some with gaps
    torch.manual_seed(0)
    exp = FP8GatedExperts()
    x = torch.randn(5, 8)
    idx = torch.tensor([[0, 2], [2, 4], [4, 0], [0, 2], [5, 2]])  # experts 1 and 3 never hit
    wts = torch.softmax(torch.randn(5, 2), -1)

    eager = exp(x, idx, wts)
    assert enable_quantized_backward(exp) == 1
    xg = x.clone().requires_grad_(True)
    grouped = exp(xg, idx, wts)
    assert torch.allclose(grouped, eager, atol=1e-5, rtol=1e-4)

    grouped.pow(2).sum().backward()
    xr = x.clone().requires_grad_(True)
    exp.dense(xr, idx, wts).pow(2).sum().backward()
    assert torch.allclose(xg.grad, xr.grad, atol=1e-4, rtol=1e-4)

    with torch.no_grad():
        assert torch.allclose(exp(x, idx, wts), eager, atol=1e-5, rtol=1e-4)
