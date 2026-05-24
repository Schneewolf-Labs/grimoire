"""String → class registry for losses and tokenizers.

Lets YAML configs reference losses / tokenize functions by short name
instead of fully-qualified Python paths. Used by grimoire.train CLI
but also fine to import programmatically.

Mirrors atelier.registry — keep these in lock-step so Merlina sees a
uniform interface across diffusion (Atelier) and LLM (Grimoire) training.
"""
import importlib
from typing import Any


LOSSES: dict[str, str] = {
    "sft":    "grimoire.losses.sft:SFTLoss",
    "orpo":   "grimoire.losses.orpo:ORPOLoss",
    "dpo":    "grimoire.losses.dpo:DPOLoss",
    "simpo":  "grimoire.losses.simpo:SimPOLoss",
    "kto":    "grimoire.losses.kto:KTOLoss",
    "cpo":    "grimoire.losses.cpo:CPOLoss",
    "ipo":    "grimoire.losses.ipo:IPOLoss",
    "grpo":   "grimoire.losses.grpo:GRPOLoss",
    "reward": "grimoire.losses.reward:RewardModelLoss",
}


TOKENIZERS: dict[str, str] = {
    "sft":        "grimoire.data.sft:tokenize_sft",
    "preference": "grimoire.data.preference:tokenize_preference",
    "kto":        "grimoire.data.kto:tokenize_kto",
    "grpo":       "grimoire.data.grpo:tokenize_grpo",
}


def _resolve(spec: str) -> Any:
    """'pkg.mod:Class' → the actual class / callable object."""
    module_path, _, name = spec.partition(":")
    if not name:
        raise ValueError(f"registry spec '{spec}' missing ':Name'")
    return getattr(importlib.import_module(module_path), name)


def get_loss_class(name: str):
    """Resolve loss short name to class. Accepts 'pkg.mod:Class' too."""
    spec = LOSSES.get(name, name)
    return _resolve(spec) if ":" in spec else _err("loss", name, LOSSES)


def get_tokenize_fn(name: str):
    """Resolve tokenize short name to function. Accepts 'pkg.mod:fn' too."""
    spec = TOKENIZERS.get(name, name)
    return _resolve(spec) if ":" in spec else _err("tokenize", name, TOKENIZERS)


def _err(kind: str, name: str, registry: dict[str, str]):
    known = ", ".join(sorted(registry))
    raise KeyError(f"unknown {kind} '{name}'. Known: {known}. "
                   f"Or pass a full 'package.module:Name' spec.")
