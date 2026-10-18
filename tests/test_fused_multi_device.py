"""Fused log-probs when the lm_head sits on another device than the body (multi-GPU device_map)."""
import pytest
import torch

from grimoire.losses.utils import forward_per_token_logps

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _split_model():
    from accelerate import dispatch_model
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=320, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2)
    model = LlamaForCausalLM(cfg)
    ids = torch.randint(0, 320, (2, 24))
    labels = ids.clone()
    labels[:, :10] = -100
    ref, mask = forward_per_token_logps(model.eval(), ids, torch.ones_like(ids), labels, fused=False)
    ref[mask].mean().backward()
    ref_grad = model.lm_head.weight.grad.clone()
    model.zero_grad()
    ref = ref.detach()
    device_map = {"model.embed_tokens": "cpu", "model.layers": "cpu", "model.norm": "cpu",
                  "model.rotary_emb": "cpu", "lm_head": 0}
    return dispatch_model(model, device_map=device_map), ids, labels, ref, mask, ref_grad


def test_fused_logps_match_with_head_on_other_device():
    model, ids, labels, ref, mask, _ = _split_model()
    with torch.no_grad():
        out, _ = forward_per_token_logps(model, ids, torch.ones_like(ids), labels, fused=True)
    assert torch.allclose(out.cpu()[mask], ref[mask], atol=1e-5)


def test_fused_logps_backward_with_head_on_other_device():
    # (layers on "cpu" are offloaded by accelerate, so only the head's grad is checkable here)
    model, ids, labels, _, mask, ref_grad = _split_model()
    out, _ = forward_per_token_logps(model, ids, torch.ones_like(ids), labels, fused=True)
    out[mask].mean().backward()
    assert torch.allclose(model.lm_head.weight.grad.cpu(), ref_grad, atol=1e-6)
