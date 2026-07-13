"""Tests for the fused chunked linear+loss path (hidden states -> lm_head chunks).

The fused path computes per-token log-probs chunk-by-chunk from the final
hidden states without materializing the full [batch, seq, vocab] logits
tensor. These tests verify numerical parity (loss, metrics, gradients) with
the standard full-logits path, correct fallback for unsupported models, and
that the fused path is actually engaged when supported.
"""

import copy
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from grimoire.losses.sft import SFTLoss
from grimoire.losses.orpo import ORPOLoss
from grimoire.losses.dpo import DPOLoss
from grimoire.losses.simpo import SimPOLoss
from grimoire.losses.kto import KTOLoss
from grimoire.losses.cpo import CPOLoss
from grimoire.losses.ipo import IPOLoss
from grimoire.losses.grpo import GRPOMethod
from grimoire.losses.utils import _fused_logits_kwarg, forward_per_token_logps

from .test_losses import (
    SimpleModel,
    MockTokenizer,
    _make_preference_batch,
    _make_kto_batch,
    _make_grpo_batch,
    _length_reward_fn,
)


class RecordingLinear(nn.Linear):
    """Linear layer that records the number of input rows per call."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.call_rows = []

    def forward(self, x):
        self.call_rows.append(x.reshape(-1, x.size(-1)).size(0))
        return super().forward(x)


class HFStyleModel(nn.Module):
    """Tiny model mimicking a transformers CausalLM: supports
    ``output_hidden_states``, ``logits_to_keep``, and ``get_output_embeddings()``.
    """

    logits_kwarg = "logits_to_keep"

    def __init__(self, vocab_size=32, hidden_size=16, final_logit_softcapping=None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.mixer = nn.Linear(hidden_size, hidden_size)
        self.lm_head = RecordingLinear(hidden_size, vocab_size, bias=False)
        self.vocab_size = vocab_size
        self.config = SimpleNamespace(
            is_encoder_decoder=False,
            final_logit_softcapping=final_logit_softcapping,
        )
        self.forward_calls = []  # records the logits-trimming value per call

    def get_output_embeddings(self):
        return self.lm_head

    def _body(self, input_ids):
        h = self.embed(input_ids)
        return h + torch.tanh(self.mixer(h))

    def _head(self, h, keep):
        self.forward_calls.append(keep)
        if isinstance(keep, int) and keep > 0:
            h = h[:, -keep:, :]
        logits = self.lm_head(h)
        cap = self.config.final_logit_softcapping
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap
        return logits

    def forward(self, input_ids, attention_mask=None, use_cache=False,
                output_hidden_states=False, position_ids=None, logits_to_keep=0):
        h = self._body(input_ids)
        logits = self._head(h, logits_to_keep)
        hidden_states = (h,) if output_hidden_states else None
        return SimpleNamespace(logits=logits, hidden_states=hidden_states)


class HFStyleModelOldKwarg(HFStyleModel):
    """Same model but with the older ``num_logits_to_keep`` forward kwarg."""

    logits_kwarg = "num_logits_to_keep"

    def forward(self, input_ids, attention_mask=None, use_cache=False,
                output_hidden_states=False, position_ids=None, num_logits_to_keep=0):
        h = self._body(input_ids)
        logits = self._head(h, num_logits_to_keep)
        hidden_states = (h,) if output_hidden_states else None
        return SimpleNamespace(logits=logits, hidden_states=hidden_states)


class HFStylePeftModel(HFStyleModel):
    """HF-style model with PEFT-like adapter support for ref_model=None paths."""

    def __init__(self, vocab_size=32, hidden_size=16):
        super().__init__(vocab_size, hidden_size)
        self.adapter = nn.Linear(hidden_size, hidden_size)
        self._adapter_enabled = True

    def _body(self, input_ids):
        h = super()._body(input_ids)
        if self._adapter_enabled:
            h = h + self.adapter(h)
        return h

    @contextmanager
    def disable_adapter(self):
        self._adapter_enabled = False
        try:
            yield
        finally:
            self._adapter_enabled = True


class HFStyleGenerativeModel(HFStyleModel):
    """HF-style model with generate() support for GRPO testing."""

    def generate(self, input_ids, attention_mask=None, max_new_tokens=8,
                 temperature=1.0, do_sample=True, pad_token_id=0, use_cache=None):
        generated = input_ids
        for _ in range(max_new_tokens):
            logits = self.forward(generated).logits[:, -1, :]
            if temperature != 1.0:
                logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
        return generated


class ModuleWrapper(nn.Module):
    """DDP-style wrapper exposing the base model via .module."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def _grads(model, loss):
    model.zero_grad(set_to_none=True)
    loss.backward()
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def _assert_close(a, b, name=""):
    assert torch.allclose(a, b, rtol=1e-4, atol=1e-6), f"{name}: {a} != {b}"


def _assert_parity(loss_fused, metrics_fused, loss_unfused, metrics_unfused,
                   grads_fused=None, grads_unfused=None):
    _assert_close(loss_fused.detach(), loss_unfused.detach(), "loss")
    assert set(metrics_fused) == set(metrics_unfused)
    for k in metrics_fused:
        assert metrics_fused[k] == pytest.approx(metrics_unfused[k], rel=1e-3, abs=1e-5), k
    if grads_fused is not None:
        assert set(grads_fused) == set(grads_unfused)
        for n in grads_fused:
            _assert_close(grads_fused[n], grads_unfused[n], n)


def _run_pair(loss_cls, model, batch, ref_model=None, **kwargs):
    """Run the same (model, batch) through fused and unfused variants of a loss."""
    results = []
    for fused in (True, False):
        kw = dict(kwargs, fused=fused)
        if ref_model is not None:
            kw["ref_model"] = ref_model
        loss_fn = loss_cls(**kw)
        loss_fn._pad_token_id = 0
        loss, metrics = loss_fn(model, batch, training=True)
        grads = _grads(model, loss)
        results.append((loss, metrics, grads))
    (lf, mf, gf), (lu, mu, gu) = results
    _assert_parity(lf, mf, lu, mu, gf, gu)
    return results


class TestFusedCapability:
    def test_hf_style_model_detected(self):
        assert _fused_logits_kwarg(HFStyleModel()) == "logits_to_keep"

    def test_old_kwarg_detected(self):
        assert _fused_logits_kwarg(HFStyleModelOldKwarg()) == "num_logits_to_keep"

    def test_simple_model_not_detected(self):
        assert _fused_logits_kwarg(SimpleModel()) is None

    def test_wrapped_model_detected(self):
        assert _fused_logits_kwarg(ModuleWrapper(HFStyleModel())) == "logits_to_keep"

    def test_fused_path_engaged(self):
        """Supported model + fused=True must trim the forward logits to 1 position."""
        model = HFStyleModel()
        batch = {
            "input_ids": torch.randint(0, 32, (2, 10)),
            "attention_mask": torch.ones(2, 10, dtype=torch.long),
            "labels": torch.randint(0, 32, (2, 10)),
        }
        SFTLoss(fused=True)(model, batch)
        assert model.forward_calls == [1]

        model.forward_calls.clear()
        SFTLoss(fused=False)(model, batch)
        assert model.forward_calls == [0]

    def test_fallback_on_unsupported_model(self):
        """fused=True on a model without support silently uses the full-logits path."""
        model = SimpleModel()
        batch = {
            "input_ids": torch.randint(0, 32, (2, 10)),
            "attention_mask": torch.ones(2, 10, dtype=torch.long),
            "labels": torch.randint(0, 32, (2, 10)),
        }
        loss, _ = SFTLoss(fused=True)(model, batch)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_only_response_tokens_get_logits(self):
        """The lm_head must only ever see unmasked (response) token positions."""
        torch.manual_seed(0)
        model = HFStyleModel()
        labels = torch.randint(0, 32, (2, 10))
        labels[:, :6] = -100  # 4 response tokens per row; one lost to the shift
        batch = {
            "input_ids": torch.randint(0, 32, (2, 10)),
            "attention_mask": torch.ones(2, 10, dtype=torch.long),
            "labels": labels,
        }
        with torch.no_grad():
            SFTLoss(fused=True)(model, batch)
        # forward computed 1 trimmed position per row (2 rows), the one-time
        # parity self-check replayed the final position (2 rows), and lm_head
        # chunks covered exactly the 8 response positions
        assert sum(model.lm_head.call_rows) == 2 + 2 + 8

    def test_chunking_respects_chunk_size(self):
        torch.manual_seed(0)
        model = HFStyleModel()
        batch = {
            "input_ids": torch.randint(0, 32, (2, 10)),
            "attention_mask": torch.ones(2, 10, dtype=torch.long),
            "labels": torch.randint(0, 32, (2, 10)),
        }
        with torch.no_grad():
            SFTLoss(fused=True, fused_chunk_size=3)(model, batch)
        # first two calls: the trimmed forward and the one-time parity replay
        chunk_rows = model.lm_head.call_rows[2:]
        assert all(rows <= 3 for rows in chunk_rows)
        assert sum(chunk_rows) == 2 * 9  # every shifted position, once


class TestFusedParity:
    def test_sft(self):
        torch.manual_seed(0)
        model = HFStyleModel()
        labels = torch.randint(0, 32, (2, 10))
        labels[:, :3] = -100
        batch = {
            "input_ids": torch.randint(0, 32, (2, 10)),
            "attention_mask": torch.ones(2, 10, dtype=torch.long),
            "labels": labels,
        }
        _run_pair(SFTLoss, model, batch)

    def test_sft_with_position_ids(self):
        torch.manual_seed(0)
        model = HFStyleModel()
        batch = {
            "input_ids": torch.randint(0, 32, (2, 10)),
            "attention_mask": torch.ones(2, 10, dtype=torch.long),
            "labels": torch.randint(0, 32, (2, 10)),
            "position_ids": torch.arange(10).repeat(2, 1),
        }
        _run_pair(SFTLoss, model, batch)

    def test_orpo(self):
        torch.manual_seed(0)
        model = HFStyleModel()
        batch = _make_preference_batch(chosen_len=6, rejected_len=10)
        _run_pair(ORPOLoss, model, batch, beta=0.1, fused_chunk_size=5)

    def test_simpo(self):
        torch.manual_seed(0)
        model = HFStyleModel()
        batch = _make_preference_batch()
        _run_pair(SimPOLoss, model, batch, beta=2.0, gamma=0.5)

    def test_cpo(self):
        torch.manual_seed(0)
        model = HFStyleModel()
        batch = _make_preference_batch()
        _run_pair(CPOLoss, model, batch, beta=0.1, label_smoothing=0.1)

    def test_dpo_with_ref_model(self):
        torch.manual_seed(0)
        model = HFStyleModel()
        torch.manual_seed(99)
        ref_model = HFStyleModel()
        ref_model.eval()
        batch = _make_preference_batch()
        _run_pair(DPOLoss, model, batch, ref_model=ref_model, beta=0.1)

    def test_dpo_peft_adapter_ref(self):
        torch.manual_seed(0)
        model = HFStylePeftModel()
        batch = _make_preference_batch()
        _run_pair(DPOLoss, model, batch, beta=0.1)

    def test_ipo_with_ref_model(self):
        torch.manual_seed(0)
        model = HFStyleModel()
        torch.manual_seed(99)
        ref_model = HFStyleModel()
        ref_model.eval()
        batch = _make_preference_batch()
        _run_pair(IPOLoss, model, batch, ref_model=ref_model, beta=0.1)

    def test_kto_with_ref_model(self):
        torch.manual_seed(0)
        model = HFStyleModel()
        torch.manual_seed(99)
        ref_model = HFStyleModel()
        ref_model.eval()
        batch = _make_kto_batch()
        _run_pair(KTOLoss, model, batch, ref_model=ref_model, beta=0.1)

    def test_grpo(self):
        torch.manual_seed(0)
        model = HFStyleGenerativeModel()
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        method = GRPOMethod(
            reward_fn=_length_reward_fn,
            tokenizer=MockTokenizer(),
            num_generations=2,
            beta=0.04,
            max_new_tokens=4,
            ref_model=ref_model,
        )
        method._pad_token_id = 0
        # One rollout -> a fixed experience batch; the loss phase must produce
        # identical loss/metrics/grads whether the policy log-probs go through
        # the fused or the full-logits path.
        experience = method.rollout(model, _make_grpo_batch())
        results = []
        for fused in (True, False):
            method.fused = fused
            loss, metrics = method(model, experience, training=True)
            results.append((loss, metrics, _grads(model, loss)))
        (lf, mf, gf), (lu, mu, gu) = results
        _assert_parity(lf, mf, lu, mu, gf, gu)

    def test_old_num_logits_to_keep_kwarg(self):
        torch.manual_seed(0)
        model = HFStyleModelOldKwarg()
        batch = _make_preference_batch()
        _run_pair(ORPOLoss, model, batch, beta=0.1)
        assert 1 in model.forward_calls

    def test_wrapped_model(self):
        """Fused path works through a DDP-style .module wrapper."""
        torch.manual_seed(0)
        model = ModuleWrapper(HFStyleModel())
        batch = _make_preference_batch()
        _run_pair(SimPOLoss, model, batch, beta=2.0, gamma=0.5)
        assert 1 in model.module.forward_calls

    def test_logit_softcapping(self):
        """Models declaring final_logit_softcapping get it replayed in the fused path."""
        torch.manual_seed(0)
        model = HFStyleModel(final_logit_softcapping=1.5)
        batch = _make_preference_batch()
        _run_pair(ORPOLoss, model, batch, beta=0.1)

    def test_softcapping_actually_matters(self):
        """Sanity check: ignoring the softcap would change the loss."""
        torch.manual_seed(0)
        model = HFStyleModel(final_logit_softcapping=1.5)
        batch = _make_preference_batch()

        loss_fn = ORPOLoss(beta=0.1, fused=True)
        loss_fn._pad_token_id = 0
        loss_capped, _ = loss_fn(model, batch, training=True)

        model.config.final_logit_softcapping = None  # unfused model would differ
        loss_fn_uncapped = ORPOLoss(beta=0.1, fused=False)
        loss_fn_uncapped._pad_token_id = 0
        loss_uncapped, _ = loss_fn_uncapped(model, batch, training=True)

        assert loss_capped.item() != pytest.approx(loss_uncapped.item(), rel=1e-4)


class TestFusedEdgeCases:
    def test_all_masked_labels(self):
        """A batch with no response tokens yields zero loss, not NaN."""
        model = HFStyleModel()
        batch = {
            "input_ids": torch.randint(0, 32, (2, 10)),
            "attention_mask": torch.ones(2, 10, dtype=torch.long),
            "labels": torch.full((2, 10), -100),
        }
        loss, _ = SFTLoss(fused=True)(model, batch)
        assert loss.item() == 0.0

    def test_no_grad_skips_checkpointing(self):
        """Under torch.no_grad the fused path must not error (eval/ref passes)."""
        model = HFStyleModel()
        batch = {
            "input_ids": torch.randint(0, 32, (2, 10)),
            "attention_mask": torch.ones(2, 10, dtype=torch.long),
            "labels": torch.randint(0, 32, (2, 10)),
        }
        with torch.no_grad():
            loss, _ = SFTLoss(fused=True)(model, batch)
        assert not loss.requires_grad

    def test_gradients_flow_to_lm_head_and_body(self):
        torch.manual_seed(0)
        model = HFStyleModel()
        batch = {
            "input_ids": torch.randint(0, 32, (2, 10)),
            "attention_mask": torch.ones(2, 10, dtype=torch.long),
            "labels": torch.randint(0, 32, (2, 10)),
        }
        loss, _ = SFTLoss(fused=True, fused_chunk_size=4)(model, batch)
        loss.backward()
        assert model.lm_head.weight.grad is not None
        assert model.lm_head.weight.grad.abs().sum() > 0
        assert model.embed.weight.grad is not None
        assert model.embed.weight.grad.abs().sum() > 0

    def test_forward_per_token_logps_shapes(self):
        model = HFStyleModel()
        input_ids = torch.randint(0, 32, (3, 12))
        labels = input_ids.clone()
        labels[:, :4] = -100
        per_token, mask = forward_per_token_logps(
            model, input_ids, torch.ones_like(input_ids), labels, fused=True,
        )
        assert per_token.shape == (3, 11)
        assert mask.shape == (3, 11)
        assert (per_token[~mask] == 0).all()

    def test_cache_uses_fused_path(self):
        """cache_reference_log_probs goes through the fused path on supported models."""
        from grimoire.data.cache import cache_reference_log_probs
        from grimoire.data.preference import PreferenceCollator
        from .test_losses import _make_preference_dataset

        torch.manual_seed(0)
        ref_model = HFStyleModel()
        ref_model.eval()
        collator = PreferenceCollator(pad_token_id=0)
        dataset = _make_preference_dataset(n=4)
        dataset = cache_reference_log_probs(ref_model, dataset, collator, batch_size=2)
        assert 1 in ref_model.forward_calls
        assert isinstance(dataset[0]["ref_chosen_logps"], float)


class TestFusedShardingGate:
    """FSDP / DeepSpeed ZeRO-3 sharded parameters must disable the fused path —
    a direct lm_head call outside the wrapped forward would see sharded or
    empty weights."""

    def test_deepspeed_zero3_param_disables_fused(self):
        model = HFStyleModel()
        model.lm_head.weight.ds_id = 0  # DeepSpeed tags partitioned params
        assert _fused_logits_kwarg(model) is None

    def test_fsdp_wrapper_disables_fused(self):
        class FullyShardedDataParallel(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, *args, **kwargs):
                return self.module(*args, **kwargs)

        model = FullyShardedDataParallel(HFStyleModel())
        assert _fused_logits_kwarg(model) is None

    def test_dtensor_param_disables_fused(self):
        class DTensor(nn.Parameter):
            pass

        model = HFStyleModel()
        model.lm_head.weight = DTensor(model.lm_head.weight.data)
        assert _fused_logits_kwarg(model) is None

    def test_sharding_gate_still_trains(self):
        """The gate falls back to the full-logits path, not an error."""
        torch.manual_seed(0)
        model = HFStyleModel()
        model.lm_head.weight.ds_id = 0
        batch = {
            "input_ids": torch.randint(0, 32, (2, 10)),
            "attention_mask": torch.ones(2, 10, dtype=torch.long),
            "labels": torch.randint(0, 32, (2, 10)),
        }
        loss, _ = SFTLoss(fused=True)(model, batch)
        assert model.forward_calls == [0]  # full-logits forward
        assert not torch.isnan(loss)


class LyingModel(HFStyleModel):
    """Applies a post-head transform it does NOT declare in its config —
    the parity self-check must catch this and fall back."""

    def _head(self, h, keep):
        logits = super()._head(h, keep)
        return torch.tanh(logits / 2.0) * 2.0


class TestFusedParityCheck:
    def test_undeclared_transform_falls_back(self, caplog):
        torch.manual_seed(0)
        model = LyingModel()
        batch = {
            "input_ids": torch.randint(0, 32, (2, 10)),
            "attention_mask": torch.ones(2, 10, dtype=torch.long),
            "labels": torch.randint(0, 32, (2, 10)),
        }
        import logging as _logging
        with caplog.at_level(_logging.WARNING, logger="grimoire.losses.utils"):
            loss_fused, _ = SFTLoss(fused=True)(model, batch)
        assert any("self-check failed" in r.message for r in caplog.records)
        # First call: trimmed forward, failed check, then a full fallback forward
        assert model.forward_calls == [1, 0]

        # The fallback result matches the plain full-logits path exactly
        loss_unfused, _ = SFTLoss(fused=False)(model, batch)
        assert torch.allclose(loss_fused.detach(), loss_unfused.detach())

        # The model is permanently marked — no trimmed forwards afterwards
        model.forward_calls.clear()
        SFTLoss(fused=True)(model, batch)
        assert model.forward_calls == [0]

    def test_parity_check_runs_once(self):
        torch.manual_seed(0)
        model = HFStyleModel()
        batch = {
            "input_ids": torch.randint(0, 32, (2, 10)),
            "attention_mask": torch.ones(2, 10, dtype=torch.long),
            "labels": torch.randint(0, 32, (2, 10)),
        }
        with torch.no_grad():
            SFTLoss(fused=True)(model, batch)
            first_calls = len(model.lm_head.call_rows)
            SFTLoss(fused=True)(model, batch)
            second_calls = len(model.lm_head.call_rows) - first_calls
        # Second call skips the parity replay: one fewer lm_head call
        assert second_calls == first_calls - 1

    def test_declared_softcap_passes_check(self):
        """A declared transform is replayed, so the self-check passes and the
        fused path stays engaged."""
        torch.manual_seed(0)
        model = HFStyleModel(final_logit_softcapping=1.5)
        batch = {
            "input_ids": torch.randint(0, 32, (2, 10)),
            "attention_mask": torch.ones(2, 10, dtype=torch.long),
            "labels": torch.randint(0, 32, (2, 10)),
        }
        SFTLoss(fused=True)(model, batch)
        SFTLoss(fused=True)(model, batch)
        assert model.forward_calls == [1, 1]  # trimmed both times, no fallback


class TestFusedMixedPrecision:
    def test_bf16_hidden_fp32_head_matches_fp32(self):
        """Half-precision hidden states with an fp32 lm_head (mixed-precision
        training with fp32 master weights) must run the head in the hidden
        dtype — matching what the model's own autocast forward would produce."""
        from grimoire.losses.utils import fused_per_token_logps

        torch.manual_seed(0)
        lm_head = nn.Linear(16, 32, bias=False)  # fp32 weights
        hidden = torch.randn(2, 10, 16)
        labels = torch.randint(0, 32, (2, 10))
        labels[:, :3] = -100

        ref_logps, ref_mask = fused_per_token_logps(hidden, lm_head, labels)
        bf16_logps, bf16_mask = fused_per_token_logps(hidden.bfloat16(), lm_head, labels)

        assert bf16_logps.dtype == torch.bfloat16  # head ran in bf16, not fp32
        assert torch.equal(ref_mask, bf16_mask)
        assert torch.allclose(
            bf16_logps.float()[bf16_mask], ref_logps[ref_mask], rtol=5e-2, atol=5e-2
        )
