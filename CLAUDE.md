# Grimoire

Simple, multi-GPU LLM fine-tuning library. Training engine for Merlina.

## Philosophy

One training loop, pluggable loss functions. Adding a new training method means writing a loss function, not a new trainer. A minimal YAML-driven CLI (`python -m grimoire.train`) for orchestrators like Merlina — no plugins, no unnecessary abstractions.

## Stack

- `accelerate` for multi-GPU / DeepSpeed / FSDP (NOT transformers.Trainer)
- `peft` for LoRA
- `torch` for everything else

## Structure

```
grimoire/
├── __init__.py        # Public API
├── config.py          # TrainingConfig dataclass
├── trainer.py         # GrimoireTrainer — the training loop
├── callbacks.py       # TrainerCallback base class
├── registry.py        # String → class registry for the YAML CLI
├── train.py           # CLI entry point (python -m grimoire.train --config run.yaml)
├── losses/
│   ├── sft.py         # SFT loss (NLL on target tokens)
│   ├── orpo.py        # ORPO loss (SFT + odds ratio)
│   ├── dpo.py         # DPO loss (reference model + preference)
│   ├── simpo.py       # SimPO loss (reference-free + reward margin)
│   ├── kto.py         # KTO loss (unpaired binary feedback + reference model)
│   ├── cpo.py         # CPO loss (reference-free + SFT + contrastive preference)
│   ├── ipo.py         # IPO loss (squared loss variant of DPO + reference model)
│   ├── online.py      # OnlineMethod base — shared rollout machinery (generate/score/reference)
│   ├── grpo.py        # GRPOMethod — online RL (group-relative REINFORCE + Dr. GRPO/DAPO options)
│   ├── rloo.py        # RLOOMethod — online RL (leave-one-out baseline, no std scaling)
│   ├── online_dpo.py  # OnlineDPOMethod — on-policy best/worst pairs + DPO loss
│   ├── raft.py        # RAFTMethod — best-of-N rejection sampling + SFT loss
│   └── reward.py      # Reward model loss (Bradley-Terry pairwise ranking)
└── data/
    ├── common.py      # encode_prompt_response() — exact prompt masking
    ├── sft.py         # SFT collator + packed collator + tokenization
    ├── preference.py  # Preference collator + tokenization (ORPO/DPO/SimPO/CPO/IPO)
    ├── kto.py         # KTO collator + tokenization (unpaired feedback)
    ├── grpo.py        # Prompt-only collator + tokenization (all online methods)
    └── cache.py       # cache_reference_log_probs() utility
```

## Key Design Decisions

- Uses `accelerate.Accelerator` directly for full control over the training loop
- Loss functions are callables: `loss, metrics = loss_fn(model, batch, training=True)`
- Loss functions own their data collators via `create_collator(pad_token_id)`
- **Online methods add one optional hook, `rollout(model, batch)`.** The trainer calls it before the loss whenever `loss_fn` defines it, turning a prompt batch into a scored *experience* batch (generate → reward → experience); the loss itself stays a pure function of that batch. Offline losses don't define `rollout`, so their path is unchanged. This is what lets the online methods (GRPO/RLOO/Online DPO/RAFT) live here without any loss secretly doing generation.
- Multi-GPU, DeepSpeed, FSDP work out of the box via `accelerate config`
- Gradient checkpointing with `use_reentrant=False` for DDP/FSDP compatibility
- Single concatenated forward pass for ORPO/DPO/SimPO/CPO/IPO (chosen + rejected in one call)
- Average log probabilities for ORPO/DPO/SimPO/KTO/CPO/IPO stability across varying response lengths
- DPO uses a frozen reference model passed to the loss function (caller manages lifecycle)
- SimPO is reference-free like ORPO but uses only a margin-based preference loss (no NLL term)
- KTO uses unpaired binary feedback with a frozen reference model (no chosen/rejected pairs needed)
- CPO is reference-free like ORPO but uses a contrastive preference term instead of odds ratio (theoretically cleaner)
- IPO replaces DPO's log-sigmoid with squared loss to prevent overfitting on noisy preference data
- The online methods (GRPO/RLOO/Online DPO/RAFT) share the `OnlineMethod` base (`losses/online.py`): generation from LEFT-padded prompts (unwrapped model), post-EOS masking, `reward_fn` scoring, and frozen-reference log-probs — all under `no_grad`. Each method is a two-phase `rollout()` + pure `__call__()`. They all call `model.generate()`, which needs full weights — the trainer refuses ZeRO-3 for any loss that defines `rollout` (a runtime guard); ZeRO-2 or lower, or FSDP, is fine. They all train from prompt-only data (`tokenize_grpo` / `GRPOCollator`, registry alias `prompt`). Per-row metadata for correctness rewards: `tokenize_grpo(metadata_fields=[...])` copies dataset columns through untokenized, `GRPOCollator` gathers them into `batch["metadata"]` (a list of dicts, never tensorized), and `_generate_and_score` calls the reward as `reward_fn(prompts, completions, metadata)` with entry i aligned to prompt i // G — batches without metadata keep the 2-arg call
- `GRPOMethod`: `rollout()` normalizes rewards within groups of G into advantages; `__call__()` computes the REINFORCE loss from that experience batch, structurally like the offline losses. One policy update per generation step (mu=1), so the importance ratio is 1 and clipping is inactive (kept for parity with the paper). Optional KL penalty (k3 estimator) against a frozen reference: `ref_model` if given, else the base model via `disable_adapter()` for PEFT. Bias-fix options: `scale_rewards=False` drops the per-group std division (Dr. GRPO), `loss_type="dr_grpo"` aggregates token log-probs as sum/max_new_tokens instead of the mean (length-bias fix), `dynamic_sampling=True` masks zero-variance groups out of the loss average (DAPO) — the mask multiplies the per-sequence objective so the loss stays graph-connected for DDP gradient sync even when every group is degenerate
- `RLOOMethod` subclasses `GRPOMethod`, overriding only the advantage computation: leave-one-out baseline `A_i = r_i - mean(r_j, j != i)`, no std scaling (unbiased, avoids GRPO's difficulty bias). Requires `num_generations >= 2`
- `OnlineDPOMethod`: `rollout()` takes the best/worst-reward completions of each group as (chosen, rejected), stacks them in the concatenated preference layout, and runs the frozen reference over the pair batch; `__call__()` is the standard DPO loss. Reward-tied pairs (best == worst, e.g. an all-tied group) carry no preference signal and are masked out of the loss (`pair_mask`, surfaced as the `tied_pairs` metric; the mask keeps the loss graph-connected when a whole batch is tied). Requires a reference policy (`ref_model` or PEFT `disable_adapter()`) — `rollout()` raises otherwise. Metrics prefix the DPO implicit rewards with `implicit_` to avoid colliding with `reward_fn` stats
- `RAFTMethod`: `rollout()` keeps only the argmax-reward completion per prompt; `__call__()` is plain SFT NLL on the winners. No reference model, no KL — the simplest online method. Optional `min_reward` floor: winners scoring below it are masked out of the loss (`keep_mask`, surfaced as the `filtered_winners` metric) so the policy never imitates the best of a bad group
- RewardModelLoss trains a reward model with Bradley-Terry pairwise ranking (reuses preference data format)
- NEFTune adds uniform noise to embeddings during SFT for improved chat quality (set `neftune_alpha` in config)
- PackedSFTCollator bins multiple sequences into single rows to minimize padding waste (requires flash attention 2)
- Liger Kernel (`use_liger=True`) patches RMSNorm/RoPE/SwiGLU/GeGLU with fused Triton kernels for ~20% speedup and ~60% less activation VRAM; CE kernels stay disabled since losses are computed externally from logits; stacks with bitsandbytes 4-bit + LoRA for low-VRAM QLoRA training
- Selective kbit upcast for QLoRA (`kbit_upcast` in TrainingConfig, default `"norms"`): bitsandbytes never quantizes `embed_tokens`/`lm_head`, and peft's `prepare_model_for_kbit_training` upcasts all of those non-quantized params to fp32 (~4 GB extra on 128k+-vocab models). The trainer instead upcasts only 1-D params (norm weights, biases) to fp32 and keeps embeddings/`lm_head` in their loaded half precision; `"all"` restores peft's behavior, `"none"` skips the upcast
- Fused chunked linear+loss path (`fused=True`, default on for SFT/ORPO/DPO/SimPO/KTO/CPO/IPO/GRPO): the model forward runs with `logits_to_keep=1` so it never builds full logits, then per-token log-probs are computed chunk-by-chunk from final hidden states through the `lm_head` under activation checkpointing — the `[batch, seq, vocab]` logits tensor (the dominant memory cost of preference training on 128k+ vocab models) is never materialized in forward or backward. Only response tokens get logits; ref-model passes and `cache_reference_log_probs()` use the same path; post-head transforms (Gemma softcapping, Cohere `logit_scale`, Granite `logits_scaling`) are replayed; head chunks run under autocast when hidden states are half-precision (matches mixed-precision numerics/speed); a first-batch parity self-check compares the replayed head against the model's own trimmed logits and permanently falls back on mismatch (never silently wrong); silently falls back on models without `logits_to_keep` support and on sharded-parameter setups (FSDP / DeepSpeed ZeRO-3)

## Usage

```python
from grimoire import GrimoireTrainer, TrainingConfig
from grimoire.losses import SFTLoss, ORPOLoss, DPOLoss, SimPOLoss, KTOLoss, CPOLoss, IPOLoss, GRPOMethod, RLOOMethod, OnlineDPOMethod, RAFTMethod, RewardModelLoss
from grimoire.data import tokenize_sft, tokenize_preference, tokenize_kto, tokenize_grpo, PackedSFTCollator

config = TrainingConfig(
    output_dir="./output",
    num_epochs=3,
    batch_size=4,
    learning_rate=2e-5,
)

# SFT
trainer = GrimoireTrainer(
    model=model, tokenizer=tokenizer, config=config,
    loss_fn=SFTLoss(), train_dataset=dataset,
)
trainer.train()
trainer.save_model("./my-model")

# ORPO — same trainer, different loss
trainer = GrimoireTrainer(
    model=model, tokenizer=tokenizer, config=config,
    loss_fn=ORPOLoss(beta=0.1), train_dataset=pref_dataset,
)
trainer.train()

# DPO — requires a frozen reference model
import copy
ref_model = copy.deepcopy(model)
ref_model.eval()
trainer = GrimoireTrainer(
    model=model, tokenizer=tokenizer, config=config,
    loss_fn=DPOLoss(ref_model=ref_model, beta=0.1), train_dataset=pref_dataset,
)
trainer.train()

# SimPO — reference-free, no NLL term, just margin-based preference
trainer = GrimoireTrainer(
    model=model, tokenizer=tokenizer, config=config,
    loss_fn=SimPOLoss(beta=2.0, gamma=0.5), train_dataset=pref_dataset,
)
trainer.train()

# KTO — unpaired binary feedback, requires reference model
import copy
ref_model = copy.deepcopy(model)
ref_model.eval()
trainer = GrimoireTrainer(
    model=model, tokenizer=tokenizer, config=config,
    loss_fn=KTOLoss(ref_model=ref_model, beta=0.1), train_dataset=kto_dataset,
)
trainer.train()

# CPO — reference-free, SFT + contrastive preference
trainer = GrimoireTrainer(
    model=model, tokenizer=tokenizer, config=config,
    loss_fn=CPOLoss(beta=0.1), train_dataset=pref_dataset,
)
trainer.train()

# IPO — like DPO but with squared loss (robust to noisy preferences)
import copy
ref_model = copy.deepcopy(model)
ref_model.eval()
trainer = GrimoireTrainer(
    model=model, tokenizer=tokenizer, config=config,
    loss_fn=IPOLoss(ref_model=ref_model, beta=0.1), train_dataset=pref_dataset,
)
trainer.train()

# Online methods (GRPO/RLOO/Online DPO/RAFT) — online RL with a reward
# function (no pre-labeled data needed). Each defines rollout(); the trainer
# calls it each step to generate and score completions before the loss.
# Nothing else about the call changes. All share the same prompt-only data.
# metadata_fields (optional) carries dataset columns through to the reward,
# which is then called as (prompts, completions, metadata) — use it for
# correctness rewards (expected outputs, test specs).
grpo_dataset = dataset.map(
    lambda x: tokenize_grpo(x, tokenizer, max_prompt_length=512),
    remove_columns=dataset.column_names,
)

# GRPO — group-normalized REINFORCE
trainer = GrimoireTrainer(
    model=model, tokenizer=tokenizer, config=config,
    loss_fn=GRPOMethod(
        reward_fn=my_reward_fn,  # callable: (prompts, completions) -> list[float]
        tokenizer=tokenizer,
        num_generations=4,
        beta=0.04,
        epsilon=0.2,
        # ref_model=ref_model,       # frozen reference for the KL penalty;
        # omit for PEFT models (base weights via disable_adapter())
        # scale_rewards=False,       # Dr. GRPO: no std division
        # loss_type="dr_grpo",       # Dr. GRPO: length-bias fix
        # dynamic_sampling=True,     # DAPO: skip zero-variance groups
    ),
    train_dataset=grpo_dataset,
)
trainer.train()

# RLOO — same rollout as GRPO, leave-one-out baseline, no std scaling
trainer = GrimoireTrainer(
    model=model, tokenizer=tokenizer, config=config,
    loss_fn=RLOOMethod(
        reward_fn=my_reward_fn, tokenizer=tokenizer, num_generations=4,
    ),
    train_dataset=grpo_dataset,
)
trainer.train()

# Online DPO — generate best/worst pairs on-policy, apply the DPO loss.
# Requires a reference policy (ref_model, or PEFT disable_adapter()).
trainer = GrimoireTrainer(
    model=model, tokenizer=tokenizer, config=config,
    loss_fn=OnlineDPOMethod(
        reward_fn=my_reward_fn, tokenizer=tokenizer,
        num_generations=2, beta=0.1,
        # ref_model=ref_model,  # omit for PEFT models
    ),
    train_dataset=grpo_dataset,
)
trainer.train()

# RAFT — best-of-N rejection sampling: SFT on the highest-reward completion.
# The simplest online method: no reference model, no KL.
trainer = GrimoireTrainer(
    model=model, tokenizer=tokenizer, config=config,
    loss_fn=RAFTMethod(
        reward_fn=my_reward_fn, tokenizer=tokenizer, num_generations=4,
    ),
    train_dataset=grpo_dataset,
)
trainer.train()

# Reward Model — train a reward model on preference data
from transformers import AutoModelForSequenceClassification
reward_model = AutoModelForSequenceClassification.from_pretrained("model-name", num_labels=1)
trainer = GrimoireTrainer(
    model=reward_model, tokenizer=tokenizer, config=config,
    loss_fn=RewardModelLoss(margin=0.0), train_dataset=pref_dataset,
)
trainer.train()

# SFT with NEFTune — noisy embeddings for better chat quality
config = TrainingConfig(
    output_dir="./output",
    neftune_alpha=5.0,  # noise scale (5-15 typical)
    # ... other params
)
trainer = GrimoireTrainer(
    model=model, tokenizer=tokenizer, config=config,
    loss_fn=SFTLoss(), train_dataset=dataset,
)
trainer.train()

# SFT with sample packing — pack short sequences to minimize padding waste
from grimoire.data import PackedSFTCollator
packed_collator = PackedSFTCollator(pad_token_id=tokenizer.pad_token_id, max_length=2048)
trainer = GrimoireTrainer(
    model=model, tokenizer=tokenizer, config=config,
    loss_fn=SFTLoss(), train_dataset=dataset,
    data_collator=packed_collator,  # overrides default collator
)
trainer.train()
```

## Commands

```bash
pip install -e .                    # Install in dev mode
pip install -e ".[quantization]"    # With bitsandbytes
pip install -e ".[logging]"         # With wandb
accelerate config                   # Configure multi-GPU / DeepSpeed
accelerate launch script.py         # Run distributed training
pytest                              # Run tests
```

## Multi-GPU / DeepSpeed

No code changes needed. Configure via accelerate:

```bash
# Interactive setup
accelerate config

# Or launch directly
accelerate launch --multi_gpu --num_processes 4 train.py
accelerate launch --use_deepspeed --deepspeed_config ds_config.json train.py
```

## ORPO Loss Formula

```
L_ORPO = L_SFT(chosen) + beta * L_OR

L_SFT  = CrossEntropy on chosen response tokens (prompt masked)
L_OR   = -mean(log(sigmoid(log_odds_ratio)))

log_odds_ratio = log(P_c/(1-P_c)) - log(P_r/(1-P_r))
               = (log_P_c - log_P_r) - (log1p(-exp(log_P_c)) - log1p(-exp(log_P_r)))
```

## DPO Loss Formula

```
L_DPO = -mean((1-eps)*log(sigmoid(x)) + eps*log(sigmoid(-x)))

x          = beta * (log(pi/pi_ref)(chosen) - log(pi/pi_ref)(rejected))
eps        = label_smoothing (default 0.0, set >0 for conservative regularization)
pi         = policy model (being trained)
pi_ref     = reference model (frozen copy of initial weights)
log(pi/pi_ref)(y) = avg_logp_pi(y|x) - avg_logp_ref(y|x)
```

## SimPO Loss Formula

```
L_SimPO = -mean(log(sigmoid(beta * (avg_logp_chosen - avg_logp_rejected - gamma))))

beta   = scaling factor (default 2.0, higher than DPO since no reference baseline)
gamma  = target reward margin (default 0.5, enforces minimum gap between chosen/rejected)
```

## KTO Loss Formula

```
L_KTO = mean(desirable_losses) + mean(undesirable_losses)

desirable_loss   = lambda_d * (1 - sigmoid(beta * (log_ratio - KL_ref)))
undesirable_loss = lambda_u * (1 - sigmoid(beta * (KL_ref - log_ratio)))

log_ratio = avg_logp_policy(y|x) - avg_logp_ref(y|x)
KL_ref    = clamp(mean(log_ratio), min=0)  (estimated from batch)

beta      = scaling factor (default 0.1)
lambda_d  = desirable weight (default 1.0)
lambda_u  = undesirable weight (default 1.0, higher = loss aversion)
```

## CPO Loss Formula

```
L_CPO = L_SFT(chosen) + beta * L_preference

L_SFT        = CrossEntropy on chosen response tokens (prompt masked)
L_preference = -mean((1-eps)*log(sigmoid(x)) + eps*log(sigmoid(-x)))
x             = beta * (avg_logp_chosen - avg_logp_rejected)
eps           = label_smoothing (default 0.0, set >0 for conservative regularization)
```

## IPO Loss Formula

```
L_IPO = mean((log(pi/pi_ref)(chosen) - log(pi/pi_ref)(rejected) - 1/(2*beta))^2)

pi         = policy model (being trained)
pi_ref     = reference model (frozen copy of initial weights)
beta       = scaling factor (default 0.1, controls target margin 1/(2*beta))
```

## GRPO Formula

The `rollout` phase produces `advantages` and `ref_logps`; the loss phase (below)
consumes them. `r` (rewards) and the group normalization happen in `rollout`.

```
L_GRPO = -mean(advantages * min(ratio, clipped_ratio)) + beta * KL

ratio         = pi(y|x) / pi_old(y|x)   (== 1: single update per generation step, mu=1)
clipped_ratio = clamp(ratio, 1 - epsilon, 1 + epsilon)
advantages    = (r - mean(r_group)) / std(r_group)  (normalized within group of G)
KL            = mean(exp(d) - d - 1),  d = log_pi_ref(y|x) - log_pi(y|x)  (k3 estimator)

pi         = policy model (being trained)
pi_old     = generation policy (same model, same weights — ratio kept for paper parity)
pi_ref     = frozen reference: ref_model if given, else base model via disable_adapter();
             if neither is available the KL term is skipped with a warning
r          = reward scores from reward_fn
G          = num_generations (completions per prompt)
beta       = KL penalty (default 0.04)
epsilon    = clip ratio (default 0.2)
```

Options: `scale_rewards=False` drops the `/std` (Dr. GRPO); `loss_type="dr_grpo"`
replaces per-sequence mean log-probs with `sum / max_new_tokens` in the policy
term (length-bias fix); `dynamic_sampling=True` masks zero-variance groups out
of the loss average (DAPO).

## RLOO Formula

Same rollout and loss structure as GRPO; only the advantages differ:

```
A_i = r_i - mean(r_j, j != i)   (leave-one-out baseline over the group of G, no std scaling)
```

Requires G >= 2. The baseline never includes the completion it baselines (unbiased).

## Online DPO Formula

The `rollout` phase generates G completions per prompt, scores them with
reward_fn, and takes best/worst as (chosen, rejected); the loss phase is the
standard DPO loss over those on-policy pairs:

```
L_OnlineDPO = L_DPO   (see DPO Loss Formula; pairs re-sampled from the policy each step)

chosen   = argmax_g r(y_g),  rejected = argmin_g r(y_g)
```

Requires a reference policy: ref_model if given, else the base model via
disable_adapter() for PEFT; rollout raises if neither is available.

## RAFT Formula

The `rollout` phase generates G completions per prompt and keeps only the
highest-reward one; the loss phase is plain SFT on the winner:

```
L_RAFT = CrossEntropy on the argmax-reward completion's tokens (prompt masked)
```

No reference model, no KL term — reward enters only through the argmax.

## Reward Model Loss Formula

```
L_RM = -mean(log(sigmoid(r_chosen - r_rejected - margin)))

r_chosen   = scalar reward for chosen sequence
r_rejected = scalar reward for rejected sequence
margin     = minimum reward gap to enforce (default 0.0)
```

## Relationship to Merlina

Grimoire is a standalone library that Merlina imports. Merlina handles:
- API endpoints, job queue, WebSocket updates
- Dataset loading, formatting, chat templates
- Model loading, LoRA config
- Hub upload

Grimoire handles:
- The training loop (offline losses, plus online methods via the `rollout` hook)
- Loss computation (SFT, ORPO, DPO, SimPO, KTO, CPO, IPO, Reward Model) and the online methods (GRPO, RLOO, Online DPO, RAFT)
- Data collation and tokenization
- Checkpointing and logging
- Multi-GPU orchestration

## CI Requirements

Before considering any work done, you MUST ensure:
1. `ruff check .` passes with no errors
2. `pytest` passes with no failures

These match the GitHub Actions workflow in `.github/workflows/tests.yml`.

## Testing

```bash
pytest                              # All tests
pytest tests/test_losses.py         # Loss computation tests
pytest tests/test_trainer.py        # Trainer tests
```
