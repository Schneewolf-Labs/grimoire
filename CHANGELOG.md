# Changelog

All notable changes to Grimoire will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [2.0.0] - 2026-07-13

### Added
- **Online-method `rollout` hook.** The trainer now calls `loss_fn.rollout(model, batch)` before the loss on each step whenever the method defines it — turning a prompt-only batch into a scored *experience* batch (generate → reward → advantages). Offline losses don't define `rollout`, so their path is byte-for-byte unchanged. This makes online RL a first-class, explicit phase instead of something smuggled inside a loss's `__call__`, while sharing the entire training loop / checkpointing / multi-GPU machinery. A startup guard rejects online methods under DeepSpeed ZeRO-3 (generation needs whole weights) with a clear error.
- **Fused chunked linear+loss path** (`fused=True`, on by default in every logits-based loss: SFT, ORPO, DPO, SimPO, KTO, CPO, IPO, GRPO). The model forward runs with `logits_to_keep=1` + `output_hidden_states=True`, and per-token log-probs are computed chunk-by-chunk from the final hidden states through the `lm_head` under activation checkpointing — the full `[batch, seq, vocab]` logits tensor is never materialized, in either the forward or backward pass. On large-vocab models (128k+) this tensor dominates preference-training memory (it's built for chosen+rejected in one pass and again for the reference model), so the fused path allows substantially larger batches. Only response tokens are pushed through the `lm_head`; prompt and padding positions never get logits at all. Reference-model passes and `cache_reference_log_probs()` use the same path. Models declaring post-head logit transforms (`final_logit_softcapping`, `logit_scale`, `logits_scaling`) have them replayed. Falls back silently to the full-logits path for models without `logits_to_keep`/`get_output_embeddings()` support — numerics are identical either way.
- `forward_per_token_logps()`, `fused_per_token_logps()`, `per_token_logps_from_logits()`, and `masked_avg_logps()` helpers in `grimoire.losses.utils` for custom losses that want the same memory profile.
- Fused-path guard rails: a first-batch parity self-check compares the replayed `lm_head` (+ declared post-head transform) against the model's own trimmed logits from the same forward and permanently falls back with a warning on mismatch, so an unknown architecture can be slower but never silently wrong; sharded-parameter setups (FSDP, DeepSpeed ZeRO-3, DTensor) are detected and use the full-logits path, since the `lm_head` cannot be called outside the wrapped forward; head chunks run under autocast when hidden states are half-precision and the head weights are fp32, matching mixed-precision numerics and tensor-core speed; the forward's retained per-layer hidden states are released before the chunked computation begins.

### Changed
- **`GRPOLoss` → `GRPOMethod`, restructured into two phases.** GRPO no longer does generation inside a loss `__call__`. The online work (generate, decode, reward, group-relative advantages, frozen-reference pass) moved into `rollout(model, batch)`, which the trainer calls before the loss; `__call__` is now a pure loss over the resulting experience batch, structurally like the offline losses. **Breaking:** import `GRPOMethod` (not `GRPOLoss`) from `grimoire` / `grimoire.losses`; the `grpo` registry key is unchanged. Behavior and loss values are unchanged.
- Reference-model losses (DPO, IPO, KTO, GRPO) now run the frozen reference forward BEFORE the policy forward. The reference pass carries no gradients, so computing it first means its activations never coexist with the policy's retained autograd graph — peak memory drops by roughly the reference pass's footprint. Loss values are unchanged (the two passes are independent).
- GRPO passes `use_cache=True` explicitly to `model.generate()`. The trainer sets `model.config.use_cache = False` for training, and if generation ever inherited that, every new token would recompute the full prefix (quadratic decoding).

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
