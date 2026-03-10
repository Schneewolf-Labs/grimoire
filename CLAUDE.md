# Grimoire

Simple, multi-GPU LLM fine-tuning library. Training engine for Merlina.

## Philosophy

One training loop, pluggable loss functions. Adding a new training method means writing a loss function, not a new trainer. No CLI, no plugins, no unnecessary abstractions.

## Stack

- `accelerate` for multi-GPU / DeepSpeed / FSDP (NOT transformers.Trainer)
- `peft` for LoRA
- `transformers` for models and tokenizers
- `torch` for everything else

## Structure

```
grimoire/
├── __init__.py        # Public API
├── config.py          # TrainingConfig dataclass
├── trainer.py         # GrimoireTrainer — the training loop
├── callbacks.py       # TrainerCallback base class
├── losses/
│   ├── sft.py         # SFT loss (NLL on target tokens)
│   ├── orpo.py        # ORPO loss (SFT + odds ratio)
│   ├── dpo.py         # DPO loss (reference model + preference)
│   ├── simpo.py       # SimPO loss (reference-free + reward margin)
│   ├── kto.py         # KTO loss (unpaired binary feedback + reference model)
│   ├── cpo.py         # CPO loss (reference-free + SFT + contrastive preference)
│   └── ipo.py         # IPO loss (squared loss variant of DPO + reference model)
└── data/
    ├── sft.py         # SFT collator + tokenization
    ├── preference.py  # Preference collator + tokenization (ORPO/DPO/SimPO/CPO/IPO)
    └── kto.py         # KTO collator + tokenization (unpaired feedback)
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

## Usage

```python
from grimoire import GrimoireTrainer, TrainingConfig
from grimoire.losses import SFTLoss, ORPOLoss, DPOLoss, SimPOLoss, KTOLoss, CPOLoss, IPOLoss
from grimoire.data import tokenize_sft, tokenize_preference, tokenize_kto

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
L_DPO = -mean(log(sigmoid(beta * (log(pi/pi_ref)(chosen) - log(pi/pi_ref)(rejected)))))

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
L_preference = -mean(log(sigmoid(beta * (avg_logp_chosen - avg_logp_rejected))))
```

## IPO Loss Formula

```
L_IPO = mean((log(pi/pi_ref)(chosen) - log(pi/pi_ref)(rejected) - 1/(2*beta))^2)

pi         = policy model (being trained)
pi_ref     = reference model (frozen copy of initial weights)
beta       = scaling factor (default 0.1, controls target margin 1/(2*beta))
```

## Relationship to Merlina

Grimoire is a standalone library that Merlina imports. Merlina handles:
- API endpoints, job queue, WebSocket updates
- Dataset loading, formatting, chat templates
- Model loading, LoRA config
- Hub upload

Grimoire handles:
- The training loop
- Loss computation (SFT, ORPO, DPO, SimPO, KTO, CPO, IPO)
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
