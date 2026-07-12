"""
CLI entry point for Grimoire training.

Usage:
    python -m grimoire.train --config configs/my_run.yaml

The YAML config drives everything: model, tokenizer, dataset, tokenize
function, loss, optional PEFT/quantization, and TrainingConfig fields.
Designed so Merlina (or any other orchestrator) can spawn training runs
with the same shape as Atelier's CLI (mirror schema where possible).

Schema (all sections optional unless noted):

    model:
      pretrained: "meta-llama/Llama-3.2-3B"  # REQUIRED
      tokenizer: null                        # default to pretrained
      model_kwargs:                          # AutoModelForCausalLM.from_pretrained kwargs
        torch_dtype: "bfloat16"
        attn_implementation: "flash_attention_2"
      tokenizer_kwargs:                      # AutoTokenizer.from_pretrained kwargs
        trust_remote_code: false

    quantization:                            # optional — bitsandbytes config
      load_in_4bit: true
      bnb_4bit_compute_dtype: "bfloat16"
      bnb_4bit_quant_type: "nf4"
      bnb_4bit_use_double_quant: true

    dataset:                                 # REQUIRED
      name: "user/dataset"                   # or {jsonl: ...} or {path: ...}
      split: "train"
      max_samples: null
      tokenize:                              # REQUIRED
        type: "sft"                          # see registry.TOKENIZERS
        max_length: 2048
        # any other tokenize kwargs are passed through:
        prompt_field: "prompt"
        response_field: "response"

    loss:
      type: "sft"                            # see registry.LOSSES
      args:
        # type-specific args (beta for preference losses, etc.)
        beta: 0.1

    peft:                                    # optional
      type: "lora"
      r: 32
      lora_alpha: 64
      target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
      lora_dropout: 0.05
      init_lora_weights: "gaussian"

    training:                                # all TrainingConfig fields supported
      output_dir: "./output"
      num_epochs: 3
      batch_size: 1
      learning_rate: 2.0e-5
      optimizer: "adafactor"
      mixed_precision: "bf16"
      gradient_checkpointing: true
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_yaml(path: str) -> dict:
    """Load a YAML config. Tries PyYAML then OmegaConf — neither is a hard dep."""
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except ImportError:
        try:
            from omegaconf import OmegaConf
            return OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        except ImportError as e:
            raise ImportError(
                "YAML config loading needs `pyyaml` or `omegaconf` installed."
            ) from e


_DTYPE_MAP = {
    "float32": "float32", "fp32": "float32",
    "float16": "float16", "fp16": "float16", "half": "float16",
    "bfloat16": "bfloat16", "bf16": "bfloat16",
}


def _resolve_dtype(v):
    """Map string dtype names → torch dtype objects; pass through everything else."""
    if isinstance(v, str) and v.lower() in _DTYPE_MAP:
        import torch
        return getattr(torch, _DTYPE_MAP[v.lower()])
    return v


def _convert_dtypes_in_kwargs(kwargs: dict) -> dict:
    """Convert any string dtype values in a kwargs dict to torch.dtype."""
    out = {}
    for k, v in kwargs.items():
        out[k] = _resolve_dtype(v)
    return out


def build_quantization_config(spec: dict | None):
    """Build a BitsAndBytesConfig from the YAML 'quantization' section."""
    if not spec:
        return None
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(**_convert_dtypes_in_kwargs(spec))


def load_model_and_tokenizer(spec: dict, quantization_config=None):
    """Load the model + tokenizer from the YAML 'model' section."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if "pretrained" not in spec:
        raise ValueError("config.model requires 'pretrained'")

    tok_path = spec.get("tokenizer") or spec["pretrained"]
    tokenizer = AutoTokenizer.from_pretrained(tok_path, **(spec.get("tokenizer_kwargs") or {}))

    model_kwargs = _convert_dtypes_in_kwargs(spec.get("model_kwargs") or {})
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    model = AutoModelForCausalLM.from_pretrained(spec["pretrained"], **model_kwargs)
    return model, tokenizer


def load_dataset_from_spec(spec: dict):
    """Load a HF dataset from the YAML 'dataset' section.

    Accepts: {name, split}, {jsonl}, or {path}. Tokenization is applied
    separately via tokenize_dataset() so the dataset is returned as-is.
    """
    from datasets import Dataset, load_dataset, load_from_disk

    max_samples = spec.get("max_samples")
    if "jsonl" in spec:
        rows = []
        with open(os.path.expanduser(spec["jsonl"])) as f:
            for line in f:
                rows.append(json.loads(line))
        ds = Dataset.from_list(rows)
    elif "path" in spec:
        ds = load_from_disk(os.path.expanduser(spec["path"]))
    elif "name" in spec:
        ds = load_dataset(spec["name"], split=spec.get("split", "train"))
    else:
        raise ValueError("dataset section must include one of: jsonl, path, name")

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def tokenize_dataset(dataset, tokenizer, tokenize_spec: dict):
    """Apply a tokenize function (sft/preference/kto) to the dataset."""
    from .registry import get_tokenize_fn

    kwargs = dict(tokenize_spec)
    tok_type = kwargs.pop("type", "sft")
    tok_fn = get_tokenize_fn(tok_type)

    # tokenize functions take (example, tokenizer, **kwargs) per-row
    def _apply(example):
        return tok_fn(example, tokenizer, **kwargs)

    cols_to_remove = dataset.column_names
    return dataset.map(_apply, remove_columns=cols_to_remove)


def build_peft_config(spec: dict | None):
    """Build a PEFT config from the YAML 'peft' section. None → no PEFT."""
    if not spec:
        return None
    kwargs = dict(spec)
    peft_type = kwargs.pop("type", "lora").lower()
    if peft_type == "lora":
        from peft import LoraConfig
        # init_lora_weights="gaussian" is the common preference-tuning default;
        # set it implicitly unless the config overrides.
        kwargs.setdefault("init_lora_weights", "gaussian")
        return LoraConfig(**kwargs)
    raise ValueError(f"unsupported peft type: {peft_type}")


def build_training_config(spec: dict | None):
    """Map the YAML 'training' section onto TrainingConfig."""
    from .config import TrainingConfig
    return TrainingConfig(**(spec or {}))


def run_from_config(cfg: dict) -> None:
    """Drive a full training run from a parsed YAML config dict."""
    from .registry import get_loss_class
    from .trainer import GrimoireTrainer

    quant_cfg = build_quantization_config(cfg.get("quantization"))

    model_cfg = cfg.get("model") or {}
    model, tokenizer = load_model_and_tokenizer(model_cfg, quantization_config=quant_cfg)

    dataset_spec = cfg.get("dataset")
    if not dataset_spec:
        raise ValueError("config requires a 'dataset' section")
    tokenize_spec = dataset_spec.get("tokenize")
    if not tokenize_spec:
        raise ValueError("config.dataset requires a 'tokenize' section")

    raw_dataset = load_dataset_from_spec(dataset_spec)
    logger.info("dataset: %d rows (pre-tokenization)", len(raw_dataset))
    train_dataset = tokenize_dataset(raw_dataset, tokenizer, tokenize_spec)
    logger.info("dataset: %d rows (post-tokenization)", len(train_dataset))

    loss_cfg = cfg.get("loss") or {"type": "sft"}
    loss_cls = get_loss_class(loss_cfg["type"])
    loss_fn = loss_cls(**(loss_cfg.get("args") or {}))

    training = build_training_config(cfg.get("training"))
    Path(training.output_dir).mkdir(parents=True, exist_ok=True)

    trainer = GrimoireTrainer(
        model=model,
        tokenizer=tokenizer,
        config=training,
        loss_fn=loss_fn,
        train_dataset=train_dataset,
        peft_config=build_peft_config(cfg.get("peft")),
    )
    trainer.train()

    final_dir = cfg.get("save_to") or os.path.join(training.output_dir, "final")
    trainer.save_model(final_dir)
    logger.info("saved to %s", final_dir)


def _parse_overrides(items: list[str]) -> dict[str, Any]:
    """Parse `--set key.sub=value` overrides into a nested dict.

    Values are JSON-decoded when possible so `--set training.num_epochs=4`
    yields an int and `--set peft.target_modules='[\"q_proj\"]'` yields a list.
    """
    out: dict[str, Any] = {}
    for item in items:
        key, _, value = item.partition("=")
        if not key or not value:
            raise SystemExit(f"--set expects key=value, got: {item}")
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        parts = key.split(".")
        cur = out
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = parsed
    return out


def _deep_merge(base: dict, over: dict) -> dict:
    """Recursively merge over into base (returns a new dict)."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="grimoire.train",
                                     description="Run a Grimoire training job from a YAML config.")
    parser.add_argument("--config", required=True, help="path to YAML config file")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="override a config key (repeatable). Values JSON-decoded.")
    parser.add_argument("--log-level", default="INFO",
                        help="logging level (default INFO)")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_yaml(args.config)
    if args.overrides:
        cfg = _deep_merge(cfg, _parse_overrides(args.overrides))

    run_from_config(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
