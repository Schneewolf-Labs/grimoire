"""Shared benchmarking utilities for Grimoire vs TRL vs Unsloth.

Keeps the three scripts per task as comparable as possible: same model,
same dataset slice, same hyperparameters, same timing / memory
accounting. Each script writes a JSON row to ``benchmarks/results/``.
"""
import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
DEFAULT_SFT_DATASET = "HuggingFaceH4/ultrachat_200k"
DEFAULT_SFT_SPLIT = "train_sft"
DEFAULT_DPO_DATASET = "HuggingFaceH4/ultrafeedback_binarized"
DEFAULT_DPO_SPLIT = "train_prefs"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def common_args(task: str) -> argparse.ArgumentParser:
    """Argument parser shared by every benchmark script.

    task: "sft" or "dpo" — only changes which dataset default is picked.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--framework", required=True, choices=["grimoire", "trl", "unsloth"])
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument(
        "--dataset",
        default=DEFAULT_SFT_DATASET if task == "sft" else DEFAULT_DPO_DATASET,
    )
    p.add_argument(
        "--split",
        default=DEFAULT_SFT_SPLIT if task == "sft" else DEFAULT_DPO_SPLIT,
    )
    p.add_argument("--num-samples", type=int, default=1024)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--fp16", action="store_true", default=False)
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--lora", action="store_true", default=False)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--output-dir", default="./benchmarks/_out")
    p.add_argument("--results-file", default=None)
    p.add_argument("--tag", default=None, help="Free-form tag stored in results.")
    return p


@dataclass
class BenchmarkResult:
    """One row per benchmark run."""

    framework: str
    task: str
    model: str
    dataset: str
    num_samples: int
    batch_size: int
    grad_accum: int
    max_length: int
    lora: bool
    precision: str
    num_gpus: int
    wall_clock_seconds: float
    train_samples_per_second: float
    train_tokens_per_second: float
    peak_vram_gb: float
    final_train_loss: float
    num_train_steps: int
    total_train_tokens: int
    framework_version: str
    torch_version: str
    gpu_name: str
    tag: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class Timer:
    """CUDA-aware wall-clock timer."""

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.t0


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def gpu_name() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_name(0)


def precision_str(args: argparse.Namespace) -> str:
    if args.fp16:
        return "fp16"
    if args.bf16:
        return "bf16"
    return "fp32"


def count_label_tokens(dataset, label_key: str = "labels") -> int:
    """Count non-masked label tokens across a tokenized dataset."""
    total = 0
    for ex in dataset:
        labels = ex[label_key]
        total += sum(1 for t in labels if t != -100)
    return total


def count_preference_tokens(dataset) -> int:
    total = 0
    for ex in dataset:
        total += sum(1 for t in ex["chosen_labels"] if t != -100)
        total += sum(1 for t in ex["rejected_labels"] if t != -100)
    return total


def save_result(result: BenchmarkResult, results_file: Optional[str]) -> Path:
    """Append result to a JSONL file. Default path: results/<framework>_<task>.jsonl."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if results_file is None:
        path = RESULTS_DIR / f"{result.framework}_{result.task}.jsonl"
    else:
        path = Path(results_file)
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(asdict(result)) + "\n")
    return path


def load_results(paths: Optional[List[Path]] = None) -> List[Dict[str, Any]]:
    """Read every JSONL row under results/ (or a given list)."""
    if paths is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        paths = sorted(RESULTS_DIR.glob("*.jsonl"))
    rows: List[Dict[str, Any]] = []
    for p in paths:
        with p.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def detect_num_gpus() -> int:
    if not torch.cuda.is_available():
        return 0
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "0"))
    world = int(os.environ.get("WORLD_SIZE", "0"))
    return max(local_world, world, torch.cuda.device_count())


def framework_version(name: str) -> str:
    try:
        if name == "grimoire":
            import grimoire
            return grimoire.__version__
        if name == "trl":
            import trl
            return trl.__version__
        if name == "unsloth":
            import unsloth
            return getattr(unsloth, "__version__", "unknown")
    except Exception as e:
        return f"unavailable ({e.__class__.__name__})"
    return "unknown"


def load_sft_samples(dataset_name: str, split: str, num_samples: int, seed: int):
    """Return a list of {"prompt": str, "response": str} dicts.

    Handles the two common schema flavors:
      - ultrachat_200k: "messages" = [{"role": ..., "content": ...}, ...]
      - raw "prompt" / "response" columns
    """
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split=f"{split}[:{num_samples}]")
    ds = ds.shuffle(seed=seed)

    def to_pair(ex):
        if "messages" in ex and ex["messages"]:
            msgs = ex["messages"]
            if len(msgs) >= 2 and msgs[-1]["role"] == "assistant":
                prompt = "".join(
                    f"<|{m['role']}|>\n{m['content']}\n" for m in msgs[:-1]
                ) + "<|assistant|>\n"
                return {"prompt": prompt, "response": msgs[-1]["content"]}
        if "prompt" in ex and "response" in ex:
            return {"prompt": ex["prompt"], "response": ex["response"]}
        raise ValueError(f"Unrecognized SFT schema; keys: {list(ex.keys())}")

    return ds.map(to_pair, remove_columns=ds.column_names)


def load_preference_samples(dataset_name: str, split: str, num_samples: int, seed: int):
    """Return a list of {"prompt", "chosen", "rejected"} dicts."""
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split=f"{split}[:{num_samples}]")
    ds = ds.shuffle(seed=seed)

    def to_triple(ex):
        prompt = ex.get("prompt", "")
        chosen = ex["chosen"]
        rejected = ex["rejected"]
        if isinstance(chosen, list):
            chosen = chosen[-1]["content"] if chosen else ""
        if isinstance(rejected, list):
            rejected = rejected[-1]["content"] if rejected else ""
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

    return ds.map(to_triple, remove_columns=ds.column_names)
