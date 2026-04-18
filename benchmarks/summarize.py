"""Print a markdown table of every result under benchmarks/results/*.jsonl."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_results


COLUMNS = [
    ("framework", "framework"),
    ("task", "task"),
    ("num_gpus", "gpus"),
    ("lora", "lora"),
    ("batch_size", "bs"),
    ("max_length", "seqlen"),
    ("wall_clock_seconds", "wall (s)"),
    ("train_tokens_per_second", "tok/s"),
    ("peak_vram_gb", "vram (gb)"),
    ("final_train_loss", "final loss"),
]


def fmt(v):
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def main():
    rows = load_results()
    if not rows:
        print("No results yet. Run a benchmark first.")
        return

    header = "| " + " | ".join(name for _, name in COLUMNS) + " |"
    sep = "|" + "|".join("---" for _ in COLUMNS) + "|"
    print(header)
    print(sep)
    rows.sort(key=lambda r: (r.get("task", ""), r.get("framework", "")))
    for r in rows:
        cells = [fmt(r.get(key, "")) for key, _ in COLUMNS]
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
