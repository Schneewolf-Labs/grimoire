# Changelog

All notable changes to Grimoire will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
