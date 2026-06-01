# Changelog

All notable changes to Grimoire will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **`GRPOTrainer`** (`grimoire.GRPOTrainer`) — a dedicated online-RL trainer (subclass of `GrimoireTrainer`) that owns the GRPO rollout loop: samples `num_generations` completions per prompt via HuggingFace `.generate()`, scores them through one synchronous `reward_fn`, computes group-relative advantages, runs current/old/reference forward passes, and emits `reward/mean`, `reward/std`, `kl`, `clip_frac`, `ratio_mean`, and `completion_length/mean` to callbacks (plus `trainer.sample_completions` for streaming prompt/completion/reward rows). The reference policy is free under LoRA (disabled adapter); a frozen deepcopy is made only when `peft_config is None` and `beta > 0`.
- **`GRPOConfig`** (`grimoire.GRPOConfig`) — `TrainingConfig` subclass adding rollout knobs (`num_generations`, `generation_batch_size`, `max_completion_length`, `temperature`, `top_p`, `top_k`), `num_iterations` (PPO inner epochs), and a `use_vllm` flag (raises `NotImplementedError` for now; HF `.generate()` is the supported backend).
- **`RewardFn`** Protocol (`grimoire.rewards`) — the synchronous reward contract; grimoire defines only the Protocol, concrete rewards live in the caller. Non-finite rewards are clamped to `0.0` with a warning.
- **`tokenize_prompt`** + **`PromptCollator`** (`grimoire.data`) — prompt-only tokenization with optional chat-template rendering and column passthrough; `PromptCollator` LEFT-pads so the prompt/completion boundary stays uniform after generation and forwards extra columns under `batch["columns"]`.
- **`GrimoireTrainer._train_batch(batch)`** hook — the base `train()` loop now delegates the per-batch optimization step to this method, which `GRPOTrainer` overrides for stateful rollouts.

### Changed
- **BREAKING — `GRPOLoss` is now a PURE function.** It no longer owns generation/reward/collation. The signature is now keyword-only tensors (`logprobs`, `old_logprobs`, `ref_logprobs`, `advantages`, `completion_mask`) returning a `GRPOLossOutput` (`.loss`, `.kl`, `.clip_frac`, `.ratio_mean`), and it gains `loss_type` (`"grpo"`/`"dr_grpo"`) and `scale_rewards` options. Drive GRPO through `GRPOTrainer` + `reward_fn` instead of `GrimoireTrainer`. Removed `GRPOCollator`/`tokenize_grpo` (use `PromptCollator`/`tokenize_prompt`); the `grpo` tokenizer registry entry now points at `tokenize_prompt`.

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
