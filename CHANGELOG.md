# Changelog

All notable changes to Grimoire will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Removed
- **GRPO** (`GRPOLoss`, `GRPOCollator`, `tokenize_grpo`, and the `grpo` loss/tokenizer registry entries). Online RL — generating completions and scoring them against a reward function or environment — is a fundamentally different concern from Grimoire's offline, static-dataset losses: it needs `model.generate()`, a reward callable, reference-model KL, and it constrains the distributed strategy (no ZeRO-3). It now lives in a dedicated RL library. **Breaking:** import GRPO from that library instead. The offline losses (SFT, ORPO, DPO, SimPO, KTO, CPO, IPO, Reward Model) are unaffected.

## [1.2.0] - 2026-05-20

### Added
- **`eval_batch_size`** config field (`Optional[int]`, defaults to `None` → falls back to `batch_size`). Decouples evaluation batch size from training; eval has no optimizer state and grad-checkpointing is disabled during `evaluate()`, so it can typically run much larger batches than training. Backwards-compatible: setting `None` preserves prior behavior.

### Fixed
- **Sample-weighted eval averaging**: `eval/loss` now aggregates by sample count rather than batch count, so the reported number is invariant to `eval_batch_size`. Previously a ragged final batch (under `drop_last=False`) would slightly bias the mean; runs with different eval batch sizes were not directly comparable.

## [1.1.1] - 2026-05-18

### Added
- **`tqdm` progress bar in `evaluate()`** with `desc="Evaluating"`, `disable` on non-main processes, `leave=False`. Long held-out evals (especially in VLM / large-corpus settings) are no longer silent; the postfix surfaces the running loss.

## [1.0.0] - 2026-03-15

### Added
- `GrimoireTrainer` — single training loop with pluggable loss functions
- `TrainingConfig` dataclass for all training hyperparameters
- `TrainerCallback` base class for custom hooks
- Loss functions: SFT, ORPO, DPO, SimPO, KTO, CPO, IPO, GRPO
- Data collators and tokenization for SFT, preference, KTO, and GRPO formats
- `cache_reference_log_probs()` utility for offline reference model computation
- Multi-GPU, DeepSpeed, and FSDP support via `accelerate`
- LoRA support via `peft`
- Gradient checkpointing with `use_reentrant=False`
- Optional bitsandbytes quantization and wandb logging
