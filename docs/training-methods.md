# Choosing a Training Method

Grimoire supports 7 training methods. This guide helps you pick the right one.

## Start here: What data do you have?

- **Prompt + completion examples** (no preference pairs) → [**SFT**](#sft)
- **Thumbs-up / thumbs-down per response** (unpaired feedback) → [**KTO**](#kto)
- **Chosen + rejected response pairs** → see [preference methods](#preference-methods) below

## SFT

Supervised fine-tuning. The model learns to generate completions given prompts. Use this to teach a base model a new task, style, or domain.

```python
from grimoire.losses import SFTLoss
from grimoire.data import tokenize_sft

dataset = dataset.map(
    lambda x: tokenize_sft(x, tokenizer, max_length=2048,
                           prompt_field="prompt", response_field="response"),
    remove_columns=dataset.column_names,
)

trainer = GrimoireTrainer(
    model=model, tokenizer=tokenizer, config=config,
    loss_fn=SFTLoss(), train_dataset=dataset,
)
trainer.train()
```

## Preference Methods

All preference methods require a dataset with `prompt`, `chosen`, and `rejected` columns. They share the same tokenization:

```python
from grimoire.data import tokenize_preference

dataset = dataset.map(
    lambda x: tokenize_preference(x, tokenizer, max_length=2048),
    remove_columns=dataset.column_names,
)
```

### Do you have enough GPU memory for two copies of the model?

**No** (single model only) → pick a reference-free method:

| Method | When to use |
|--------|-------------|
| [ORPO](#orpo) | Good default. Combines SFT + preference in one loss. Best when the model still needs to learn the task (not just preferences). |
| [SimPO](#simpo) | Model already knows the task (e.g., after SFT). Simpler than ORPO — no SFT term, just margin-based preference. |
| [CPO](#cpo) | Like ORPO but uses contrastive preference instead of odds ratio. Try if ORPO isn't converging. |

**Yes** (can load a frozen reference model) → pick a reference-based method:

| Method | When to use |
|--------|-------------|
| [DPO](#dpo) | The standard. Well-studied, reliable. Start here if you can afford the memory. |
| [IPO](#ipo) | Use instead of DPO when preference labels are noisy or crowd-sourced. Squared loss prevents overfitting to mislabeled pairs. |

### ORPO

Odds Ratio Preference Optimization. Combines SFT loss on the chosen response with an odds ratio preference term. No reference model needed.

- **Best for:** Aligning a base model in a single training run (SFT + alignment together)
- **Memory:** Low (one model)
- **Key param:** `beta` (default 0.1) — weight of the preference term relative to SFT

```python
from grimoire.losses import ORPOLoss

trainer = GrimoireTrainer(
    model=model, tokenizer=tokenizer, config=config,
    loss_fn=ORPOLoss(beta=0.1), train_dataset=dataset,
)
```

**Loss formula:**
```
L_ORPO = L_SFT(chosen) + beta * L_OR
L_OR   = -mean(log(sigmoid(log_odds_ratio)))
```

### DPO

Direct Preference Optimization. The standard preference alignment method. Requires a frozen copy of the model as a reference.

- **Best for:** Aligning an already-capable model (after SFT) with reliable preference data
- **Memory:** High (two copies of the model)
- **Key param:** `beta` (default 0.1) — controls how far the policy can drift from the reference

```python
import copy
from grimoire.losses import DPOLoss

ref_model = copy.deepcopy(model)
ref_model.eval()

trainer = GrimoireTrainer(
    model=model, tokenizer=tokenizer, config=config,
    loss_fn=DPOLoss(ref_model=ref_model, beta=0.1), train_dataset=dataset,
)
```

**Loss formula:**
```
L_DPO = -mean(log(sigmoid(beta * (log(pi/pi_ref)(chosen) - log(pi/pi_ref)(rejected)))))
```

### SimPO

Simple Preference Optimization. Reference-free like ORPO, but without an SFT term — purely margin-based preference alignment.

- **Best for:** Preference alignment after the model already knows the task (post-SFT)
- **Memory:** Low (one model)
- **Key params:** `beta` (default 2.0) — scaling factor; `gamma` (default 0.5) — minimum reward margin between chosen and rejected

```python
from grimoire.losses import SimPOLoss

trainer = GrimoireTrainer(
    model=model, tokenizer=tokenizer, config=config,
    loss_fn=SimPOLoss(beta=2.0, gamma=0.5), train_dataset=dataset,
)
```

**Loss formula:**
```
L_SimPO = -mean(log(sigmoid(beta * (avg_logp_chosen - avg_logp_rejected - gamma))))
```

### CPO

Contrastive Preference Optimization. Reference-free like ORPO, but uses a contrastive preference term instead of odds ratio.

- **Best for:** Alternative to ORPO if it isn't converging well; theoretically cleaner gradient signal
- **Memory:** Low (one model)
- **Key param:** `beta` (default 0.1) — weight of the preference term

```python
from grimoire.losses import CPOLoss

trainer = GrimoireTrainer(
    model=model, tokenizer=tokenizer, config=config,
    loss_fn=CPOLoss(beta=0.1), train_dataset=dataset,
)
```

**Loss formula:**
```
L_CPO = L_SFT(chosen) + beta * -mean(log(sigmoid(beta * (avg_logp_chosen - avg_logp_rejected))))
```

### IPO

Identity Preference Optimization. Like DPO but replaces log-sigmoid with squared loss, making it robust to noisy preference labels.

- **Best for:** Preference alignment when labels are crowd-sourced, noisy, or you suspect mislabeled pairs
- **Memory:** High (two copies of the model)
- **Key param:** `beta` (default 0.1) — controls target margin `1/(2*beta)`

```python
import copy
from grimoire.losses import IPOLoss

ref_model = copy.deepcopy(model)
ref_model.eval()

trainer = GrimoireTrainer(
    model=model, tokenizer=tokenizer, config=config,
    loss_fn=IPOLoss(ref_model=ref_model, beta=0.1), train_dataset=dataset,
)
```

**Loss formula:**
```
L_IPO = mean((log(pi/pi_ref)(chosen) - log(pi/pi_ref)(rejected) - 1/(2*beta))^2)
```

## KTO

Kahneman-Tversky Optimization. The only method that works with **unpaired** feedback — each example is independently labeled good or bad.

- **Best for:** Binary user feedback (likes/dislikes) where collecting paired preferences is impractical
- **Memory:** High (two copies of the model)
- **Key params:** `beta` (default 0.1) — scaling factor; `lambda_d` / `lambda_u` (default 1.0) — weights for desirable/undesirable examples (increase `lambda_u` for loss aversion)

```python
import copy
from grimoire.losses import KTOLoss
from grimoire.data import tokenize_kto

ref_model = copy.deepcopy(model)
ref_model.eval()

dataset = dataset.map(
    lambda x: tokenize_kto(x, tokenizer, max_length=2048),
    remove_columns=dataset.column_names,
)

trainer = GrimoireTrainer(
    model=model, tokenizer=tokenizer, config=config,
    loss_fn=KTOLoss(ref_model=ref_model, beta=0.1), train_dataset=dataset,
)
```

**Loss formula:**
```
L_KTO = mean(desirable_losses) + mean(undesirable_losses)

desirable_loss   = lambda_d * (1 - sigmoid(beta * (log_ratio - KL_ref)))
undesirable_loss = lambda_u * (1 - sigmoid(beta * (KL_ref - log_ratio)))
```

## Quick Reference

| Method | Data Format | Ref Model | Memory | Best For |
|--------|-------------|-----------|--------|----------|
| SFT | Completions | No | Low | Teaching a task from scratch |
| ORPO | Paired | No | Low | SFT + alignment in one pass |
| SimPO | Paired | No | Low | Alignment after SFT (margin-based) |
| CPO | Paired | No | Low | Alignment after SFT (contrastive) |
| DPO | Paired | Yes | High | Standard preference alignment |
| IPO | Paired | Yes | High | Noisy preference data |
| KTO | Unpaired | Yes | High | Binary feedback (no pairs) |

## Typical Training Pipelines

1. **Base model → instruction follower:** SFT
2. **Base model → aligned in one step:** ORPO or CPO
3. **SFT model → aligned:** DPO, SimPO, or IPO
4. **SFT model → aligned from user feedback:** KTO
