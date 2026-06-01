# Grimoire

Simple, multi-GPU LLM fine-tuning library. Training engine for Merlina.

## Philosophy

One training loop, pluggable loss functions. Adding a new training method means writing a loss function, not a new trainer. No CLI, no plugins, no unnecessary abstractions.

## Stack

- `accelerate` for multi-GPU / DeepSpeed / FSDP (NOT transformers.Trainer)
- `peft` for LoRA
- `torch` for everything else

## Structure

```
grimoire/
├── __init__.py        # Public API
├── config.py          # TrainingConfig dataclass + GRPOConfig subclass
├── trainer.py         # GrimoireTrainer — the training loop
├── grpo_trainer.py    # GRPOTrainer — online RL loop (rollout + reward + advantages)
├── rewards.py         # RewardFn Protocol (the reward membrane; no concrete impls)
├── callbacks.py       # TrainerCallback base class
├── losses/
│   ├── sft.py         # SFT loss (NLL on target tokens)
│   ├── orpo.py        # ORPO loss (SFT + odds ratio)
│   ├── dpo.py         # DPO loss (reference model + preference)
│   ├── simpo.py       # SimPO loss (reference-free + reward margin)
│   ├── kto.py         # KTO loss (unpaired binary feedback + reference model)
│   ├── cpo.py         # CPO loss (reference-free + SFT + contrastive preference)
│   ├── ipo.py         # IPO loss (squared loss variant of DPO + reference model)
│   ├── grpo.py        # GRPO loss (PURE tensor->scalar; GRPOTrainer feeds it)
│   └── reward.py      # Reward model loss (Bradley-Terry pairwise ranking)
└── data/
    ├── sft.py         # SFT collator + packed collator + tokenization
    ├── preference.py  # Preference collator + tokenization (ORPO/DPO/SimPO/CPO/IPO)
    ├── kto.py         # KTO collator + tokenization (unpaired feedback)
    ├── grpo.py        # PromptCollator (left-pad) + tokenize_prompt (prompt-only)
    └── cache.py       # cache_reference_log_probs() utility
```

## Key Design Decisions

- Uses `accelerate.Accelerator` directly for full control over the training loop
- Loss functions are callables: `loss, metrics = loss_fn(model, batch, training=True)`
- Loss functions own their data collators via `create_collator(pad_token_id)`
- Multi-GPU, DeepSpeed, FSDP work out of the box via `accelerate config`
- Gradient checkpointing with `use_reentrant=False` for DDP/FSDP compatibility
- Single concatenated forward pass for ORPO/DPO/SimPO/CPO/IPO (chosen + rejected in one call)
- Average log probabilities for ORPO/DPO/SimPO/KTO/CPO/IPO stability across varying response lengths
- DPO uses a frozen reference model passed to the loss function (caller manages lifecycle)
- SimPO is reference-free like ORPO but uses only a margin-based preference loss (no NLL term)
- KTO uses unpaired binary feedback with a frozen reference model (no chosen/rejected pairs needed)
- CPO is reference-free like ORPO but uses a contrastive preference term instead of odds ratio (theoretically cleaner)
- IPO replaces DPO's log-sigmoid with squared loss to prevent overfitting on noisy preference data
- GRPO is the one exception to "one trainer, pluggable loss": online RL is stateful, so `GRPOTrainer` (a `GrimoireTrainer` subclass) owns rollout generation, reward scoring, advantage computation, and reference/old-policy forwards, while `GRPOLoss` stays a PURE `tensor->scalar` function. The base `GrimoireTrainer.train()` calls a `_train_batch(batch)` hook that `GRPOTrainer` overrides
- GRPO reward enters as ONE synchronous `RewardFn` callable (a Protocol in `grimoire.rewards`); grimoire never knows whether it's a regex, a network judge, or sandboxed code — Merlina builds/contains that behind the sync interface. Non-finite rewards are clamped to 0.0 with a warning
- GRPO reference policy is FREE under LoRA (disable the adapter -> base = reference); a frozen deepcopy is made only when `peft_config is None` and `beta > 0`. `beta == 0` skips reference logprobs entirely. Requires ZeRO-2 or lower (generation needs full weights)
- GRPO datasets are prompt-only; `PromptCollator` LEFT-pads so the prompt/completion boundary is uniform after `generate()`, and passes extra columns through `batch["columns"]` to the reward fn
- RewardModelLoss trains a reward model with Bradley-Terry pairwise ranking (reuses preference data format)
- NEFTune adds uniform noise to embeddings during SFT for improved chat quality (set `neftune_alpha` in config)
- PackedSFTCollator bins multiple sequences into single rows to minimize padding waste (requires flash attention 2)
- Liger Kernel (`use_liger=True`) patches RMSNorm/RoPE/SwiGLU/GeGLU with fused Triton kernels for ~20% speedup and ~60% less activation VRAM; CE kernels stay disabled since losses are computed externally from logits; stacks with bitsandbytes 4-bit + LoRA for low-VRAM QLoRA training

## Usage

```python
from grimoire import GrimoireTrainer, GRPOTrainer, TrainingConfig, GRPOConfig
from grimoire.losses import SFTLoss, ORPOLoss, DPOLoss, SimPOLoss, KTOLoss, CPOLoss, IPOLoss, GRPOLoss, RewardModelLoss
from grimoire.data import tokenize_sft, tokenize_preference, tokenize_kto, tokenize_prompt, PackedSFTCollator

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

# GRPO — online RL with a reward function (no pre-labeled data needed).
# Uses a dedicated GRPOTrainer + GRPOConfig; the loss is a PURE function and
# the reward enters as one synchronous callable (the trainer owns the rollout).
grpo_dataset = dataset.map(
    lambda x: tokenize_prompt(x, tokenizer, max_prompt_length=512),
    # NOTE: do NOT remove_columns — the reward fn receives them via **columns
)

def my_reward_fn(prompts, completions, **columns):
    # prompts: length B, completions: length B*num_generations (group-major),
    # columns: each list aligned to completions. Return one float per completion.
    return [float(len(c)) for c in completions]

grpo_config = GRPOConfig(
    output_dir="./output",
    num_generations=8,        # G — completions sampled per prompt
    max_completion_length=256,
    temperature=1.0,
    num_iterations=1,         # PPO inner epochs; 1 => old policy == sampling policy
    # ... plus any TrainingConfig field (batch_size, learning_rate, ...)
)
trainer = GRPOTrainer(
    model=model, tokenizer=tokenizer, config=grpo_config,
    loss_fn=GRPOLoss(beta=0.04, epsilon=0.2, loss_type="grpo", scale_rewards=True),
    reward_fn=my_reward_fn,
    train_dataset=grpo_dataset,
    peft_config=lora_config,  # LoRA => reference = disabled adapter (no second model)
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

## GRPO Loss Formula

`GRPOLoss` is a PURE function of precomputed per-token logprobs + per-sequence
advantages (the trainer produces all inputs):

```
L_GRPO = aggregate_t( -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A) + beta * KL )

ratio  = exp(logprobs - old_logprobs)             (per completion token)
A      = advantages                               (per sequence, broadcast over tokens)
KL     = exp(ref - logprobs) - (ref - logprobs) - 1   (k3 estimator, non-negative)

advantages (trainer-side, group of G):
    grpo     (scale_rewards=True):  A = (r - mean(r_group)) / (std(r_group) + 1e-4)
    dr_grpo  (scale_rewards=False): A = r - mean(r_group)            (mean-center only)

aggregate_t:
    grpo     -> per-sequence length normalization, then mean over sequences
    dr_grpo  -> divide by a constant (padded completion width); no length norm

pi/logprobs     = policy model (being trained)
pi_old          = sampling policy; == logprobs.detach() when num_iterations == 1
ref             = reference policy (disabled LoRA adapter, or frozen copy); None iff beta == 0
G               = num_generations (completions per prompt)
beta            = KL penalty (default 0.04; 0 disables KL + reference forward)
epsilon         = clip range (default 0.2; only bites when num_iterations > 1)
```

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
- The training loop
- Loss computation (SFT, ORPO, DPO, SimPO, KTO, CPO, IPO, GRPO, Reward Model)
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
