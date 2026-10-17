"""Chunked eager attention (grimoire/attention.py) against DeepSeek-V4's own eager path, on CPU."""
import pytest
import torch

import grimoire.attention as ga

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "DeepseekV4Config"):
    pytest.skip("transformers without deepseek_v4 (5.20+)", allow_module_level=True)


def tiny_dsv4():
    cfg = transformers.DeepseekV4Config(
        vocab_size=128, hidden_size=64, moe_intermediate_size=32, n_routed_experts=4, num_experts_per_tok=2,
        num_hidden_layers=4, num_attention_heads=8, head_dim=32, qk_rope_head_dim=8, q_lora_rank=32,
        o_lora_rank=16, sliding_window=16, index_head_dim=16, index_n_heads=4, index_topk=8, num_hash_layers=0,
        o_groups=2, compress_ratios=[0, 4, 128, 4, 0])      # sliding, CSA, HCA, CSA
    model = transformers.AutoModelForCausalLM.from_config(cfg).float().eval()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "sinks" in n:
                p.normal_()
    return model


def test_chunked_matches_eager_logits_and_grads(monkeypatch):
    torch.manual_seed(0)
    model = tiny_dsv4()
    ids = torch.randint(0, 128, (1, 300))

    def run():
        model.zero_grad()
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        return out.logits.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    logits0, grads0 = run()
    monkeypatch.setattr(ga, "_LOGIT_BUDGET", 8 * 400 * 7)     # ~7 queries per block
    blocks = []
    block_fn = ga._attention_block
    monkeypatch.setattr(ga, "_attention_block", lambda *a: (blocks.append(a[0].shape[2]), block_fn(*a))[1])
    assert ga.enable_chunked_attention(model)
    assert model.config._attn_implementation == ga.ATTN_IMPLEMENTATION
    logits1, grads1 = run()

    assert len(blocks) > 40
    assert torch.allclose(logits0, logits1, atol=1e-5)
    assert grads0.keys() == grads1.keys()
    for n in grads0:
        assert torch.allclose(grads0[n], grads1[n], atol=1e-5, rtol=1e-4), n


def test_other_models_untouched():
    cfg = transformers.LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                   num_attention_heads=4, num_key_value_heads=2)
    model = transformers.AutoModelForCausalLM.from_config(cfg, attn_implementation="eager")
    assert not ga.enable_chunked_attention(model)
    assert model.config._attn_implementation == "eager"
