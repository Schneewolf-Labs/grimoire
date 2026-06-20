# Benchmarks

Apples-to-apples comparison of Grimoire against TRL and Unsloth on matched
SFT and DPO workloads. Every framework trains the **same model** on the
**same dataset slice** with the **same hyperparameters**; each run appends
one JSONL row per framework to `benchmarks/results/`.

## Layout

```
benchmarks/
├── common.py             # shared arg parser, timer, VRAM + token counters, result schema
├── requirements.txt      # extra deps on top of `pip install -e .`
├── run_all.sh            # orchestrator — runs the full sweep, skips missing frameworks
├── summarize.py          # prints a markdown table of all results/*.jsonl rows
├── sft/
│   ├── grimoire_sft.py
│   ├── trl_sft.py
│   └── unsloth_sft.py
├── dpo/
│   ├── grimoire_dpo.py
│   ├── trl_dpo.py
│   └── unsloth_dpo.py
└── results/              # JSONL output, one row per run
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . -r benchmarks/requirements.txt
# Unsloth is CUDA-version specific — install per their instructions:
# https://github.com/unslothai/unsloth
```

## Run

Single framework / single task:

```bash
python benchmarks/sft/grimoire_sft.py --framework grimoire --num-samples 512
python benchmarks/sft/trl_sft.py       --framework trl      --num-samples 512
python benchmarks/sft/unsloth_sft.py   --framework unsloth  --num-samples 512 --lora
```

Full sweep (all frameworks, SFT + DPO):

```bash
./benchmarks/run_all.sh
```

Task-only or framework-only sweeps:

```bash
./benchmarks/run_all.sh sft            # SFT across all frameworks
./benchmarks/run_all.sh dpo grimoire   # just Grimoire DPO
```

Multi-GPU for Grimoire / TRL (Unsloth is single-GPU only):

```bash
accelerate launch benchmarks/sft/grimoire_sft.py --framework grimoire
accelerate launch benchmarks/sft/trl_sft.py      --framework trl
```

## What gets measured

Every script records the same `BenchmarkResult` schema (see `common.py`):

| field                        | meaning                                                    |
|------------------------------|------------------------------------------------------------|
| `wall_clock_seconds`         | CUDA-synchronized time around `.train()`                   |
| `train_samples_per_second`   | training samples processed per second                      |
| `train_tokens_per_second`    | label / response tokens per second                         |
| `peak_vram_gb`               | `torch.cuda.max_memory_allocated()` after training         |
| `final_train_loss`           | loss at the last optimizer step                            |
| `num_train_steps`            | optimizer steps actually taken                             |
| `framework_version`          | grimoire / trl / unsloth version string                    |
| `torch_version`, `gpu_name`  | environment context                                        |
| `tag`                        | free-form label (`--tag lora-rank-16` etc.)                |

## Ground rules for a fair comparison

1. **Same base model.** Default is `Qwen/Qwen2.5-0.5B` — small enough for
   a single consumer GPU, big enough that kernel differences matter.
2. **Same data slice.** `--num-samples N` takes the first `N` examples
   (shuffled with a fixed seed) from the same dataset.
3. **Same hyperparameters.** Batch size, grad-accum, LR, warmup, seqlen,
   precision, gradient checkpointing are all shared CLI flags.
4. **Same GPU.** Run the sweep on one machine before comparing rows.
5. **Unsloth is single-GPU only.** Keep `num_gpus=1` when comparing all
   three; for multi-GPU scaling compare Grimoire vs TRL separately.
6. **LoRA vs full fine-tune.** Unsloth's advantage is largest with LoRA;
   be explicit about which column you report.

## Known caveats

- Token counting differs slightly: Grimoire and Unsloth/TRL-DPO count
  label tokens with `-100` masking; TRL-SFT treats the whole text as
  labels. Prefer comparing `wall_clock_seconds` + `peak_vram_gb` as the
  primary metrics, and treat `tokens_per_second` as indicative.
- `final_train_loss` across frameworks is only meaningful within the
  same task — DPO loss != SFT loss.
- Unsloth patches `transformers` at import time; always import it before
  `transformers` / `trl` (the scripts already do this).
- The orchestrator swallows individual framework failures so one
  broken install doesn't kill the sweep — check stderr for skips.

## Adding another framework

1. Drop a new script under `sft/` or `dpo/` that calls `common_args(...)`
   and writes a `BenchmarkResult` via `save_result(...)`.
2. Add a line in `run_all.sh`.
3. Re-run `python benchmarks/summarize.py` to pick it up automatically.
