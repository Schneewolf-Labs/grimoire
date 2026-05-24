"""Tests for grimoire.train CLI module + grimoire.registry."""

import json
import os
import tempfile

import pytest

from grimoire import registry
from grimoire.train import (
    _convert_dtypes_in_kwargs,
    _deep_merge,
    _parse_overrides,
    _resolve_dtype,
    build_peft_config,
    build_training_config,
    load_dataset_from_spec,
    load_yaml,
)


# ---- registry ----

class TestRegistry:
    def test_get_loss_class_known(self):
        cls = registry.get_loss_class("sft")
        assert cls.__name__ == "SFTLoss"

    def test_get_loss_class_orpo(self):
        cls = registry.get_loss_class("orpo")
        assert cls.__name__ == "ORPOLoss"

    def test_get_loss_class_full_spec(self):
        cls = registry.get_loss_class("grimoire.losses.dpo:DPOLoss")
        assert cls.__name__ == "DPOLoss"

    def test_get_loss_class_unknown(self):
        with pytest.raises(KeyError, match="unknown loss"):
            registry.get_loss_class("frobnicate_v9")

    def test_get_tokenize_fn_known(self):
        fn = registry.get_tokenize_fn("sft")
        assert fn.__name__ == "tokenize_sft"

    def test_get_tokenize_fn_preference(self):
        fn = registry.get_tokenize_fn("preference")
        assert fn.__name__ == "tokenize_preference"

    def test_get_tokenize_fn_unknown(self):
        with pytest.raises(KeyError, match="unknown tokenize"):
            registry.get_tokenize_fn("not_a_tokenizer")


# ---- dtype resolution ----

class TestDtypeResolution:
    def test_resolve_bf16(self):
        import torch
        assert _resolve_dtype("bfloat16") is torch.bfloat16
        assert _resolve_dtype("bf16") is torch.bfloat16

    def test_resolve_fp16(self):
        import torch
        assert _resolve_dtype("float16") is torch.float16
        assert _resolve_dtype("fp16") is torch.float16
        assert _resolve_dtype("half") is torch.float16

    def test_resolve_passthrough(self):
        assert _resolve_dtype("not_a_dtype") == "not_a_dtype"
        assert _resolve_dtype(42) == 42

    def test_convert_kwargs(self):
        import torch
        out = _convert_dtypes_in_kwargs({"torch_dtype": "bf16", "max_length": 2048})
        assert out["torch_dtype"] is torch.bfloat16
        assert out["max_length"] == 2048


# ---- CLI helpers ----

class TestParseOverrides:
    def test_simple_string(self):
        assert _parse_overrides(["foo=bar"]) == {"foo": "bar"}

    def test_json_int(self):
        assert _parse_overrides(["training.num_epochs=4"]) == {"training": {"num_epochs": 4}}

    def test_json_list(self):
        result = _parse_overrides(['peft.target=["q","v"]'])
        assert result == {"peft": {"target": ["q", "v"]}}

    def test_json_bool(self):
        assert _parse_overrides(["training.bf16=true"]) == {"training": {"bf16": True}}

    def test_nested_three_deep(self):
        assert _parse_overrides(["a.b.c=1"]) == {"a": {"b": {"c": 1}}}

    def test_multiple(self):
        assert _parse_overrides(["a=1", "b=2"]) == {"a": 1, "b": 2}

    def test_missing_value(self):
        with pytest.raises(SystemExit):
            _parse_overrides(["nokey"])


class TestDeepMerge:
    def test_simple(self):
        assert _deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_override(self):
        assert _deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_nested(self):
        result = _deep_merge(
            {"training": {"lr": 1e-4, "epochs": 8}},
            {"training": {"epochs": 4}},
        )
        assert result == {"training": {"lr": 1e-4, "epochs": 4}}

    def test_base_unchanged(self):
        base = {"a": {"b": 1}}
        _deep_merge(base, {"a": {"b": 2}})
        assert base == {"a": {"b": 1}}


class TestLoadYaml:
    def test_round_trip(self):
        pytest.importorskip("yaml")
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("model:\n  pretrained: foo\n")
            path = f.name
        try:
            cfg = load_yaml(path)
            assert cfg["model"]["pretrained"] == "foo"
        finally:
            os.unlink(path)


# ---- build_* helpers ----

class TestBuildPeftConfig:
    def test_none_returns_none(self):
        assert build_peft_config(None) is None
        assert build_peft_config({}) is None

    def test_lora(self):
        cfg = build_peft_config({
            "type": "lora", "r": 16, "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"],
        })
        assert cfg.r == 16
        assert cfg.lora_alpha == 32
        assert cfg.init_lora_weights == "gaussian"  # implicit default

    def test_lora_explicit_init(self):
        cfg = build_peft_config({
            "type": "lora", "r": 8, "lora_alpha": 16,
            "target_modules": ["q_proj"], "init_lora_weights": True,
        })
        assert cfg.init_lora_weights is True

    def test_unsupported_type(self):
        with pytest.raises(ValueError, match="unsupported peft type"):
            build_peft_config({"type": "ia3"})


class TestBuildTrainingConfig:
    def test_override(self):
        cfg = build_training_config({"num_epochs": 12, "batch_size": 4})
        assert cfg.num_epochs == 12
        assert cfg.batch_size == 4


class TestLoadDatasetFromSpec:
    def test_jsonl(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"prompt": "x", "response": "y"}) + "\n")
            f.write(json.dumps({"prompt": "a", "response": "b"}) + "\n")
            path = f.name
        try:
            ds = load_dataset_from_spec({"jsonl": path})
            assert len(ds) == 2
            assert "prompt" in ds.column_names
            assert "response" in ds.column_names
        finally:
            os.unlink(path)

    def test_jsonl_max_samples(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            for i in range(5):
                f.write(json.dumps({"prompt": f"x{i}", "response": "y"}) + "\n")
            path = f.name
        try:
            ds = load_dataset_from_spec({"jsonl": path, "max_samples": 2})
            assert len(ds) == 2
        finally:
            os.unlink(path)

    def test_no_source(self):
        with pytest.raises(ValueError, match="must include one of"):
            load_dataset_from_spec({})
