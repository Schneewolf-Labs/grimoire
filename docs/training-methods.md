# Choosing a Training Method

Grimoire supports 11 training methods. This guide helps you pick the right one.

## Start here: What data do you have?

- **Prompt + completion examples** (no preference pairs) → [**SFT**](#sft)
- **Thumbs-up / thumbs-down per response** (unpaired feedback) → [**KTO**](#kto)
- **Chosen + rejected response pairs** → see [preference methods](#preference-methods) below
- **Prompts + a reward function** (generate and score on-the-fly) → see [online methods](#online-methods): [**GRPO**](#grpo), [**RLOO**](#rloo), [**Online DPO**](#online-dpo), [**RAFT**](#raft)

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

**Memory tip — caching reference log probs:** Since the reference model is frozen, its log probs never change. You can precompute them once, store them in the dataset, and delete the reference model before training. This halves memory during training:

```python
from grimoire.data import cache_reference_log_probs

loss_fn = DPOLoss(ref_model=ref_model, beta=0.1)
collator = loss_fn.create_collator(tokenizer.pad_token_id)
dataset = cache_reference_log_probs(ref_model, dataset, collator)

del ref_model
import torch; torch.cuda.empty_cache()

# DPOLoss will use the cached values automatically — no ref_model needed
trainer = GrimoireTrainer(
    model=model, tokenizer=tokenizer, config=config,
    loss_fn=DPOLoss(beta=0.1), train_dataset=dataset,
)
```

This also works with `IPOLoss` and `KTOLoss`.

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

## Online Methods

The online methods need no pre-labeled responses — the model learns from its own generations, scored by a reward function you write. They all share the same setup:

- **Data:** prompts only, tokenized with `tokenize_grpo` (registry alias: `prompt`)
- **Reward function:** a callable `(prompts, completions) → list[float]`
- **Two phases:** each method exposes a `rollout(model, batch)` that the trainer calls before the loss each step — that's where generation, reward scoring, and experience-building happen. Its `__call__` is then a pure loss over the resulting experience batch. You don't call `rollout` yourself; passing the method as `loss_fn` is all that's needed.
- **Constraint:** ZeRO-2 or lower (or FSDP), not ZeRO-3 — `model.generate()` needs full weight access; the trainer enforces this

```python
from grimoire.data import tokenize_grpo

# Dataset needs only prompts — no responses required
dataset = dataset.map(
    lambda x: tokenize_grpo(x, tokenizer, max_prompt_length=512),
    remove_columns=dataset.column_names,
)

def reward_fn(prompts, completions):
    # Return a score for each (prompt, completion) pair
    return [score_completion(p, c) for p, c in zip(prompts, completions)]
```

**Per-row metadata (correctness rewards):** rewards often need more than the
prompt text — an expected output, a test spec. Name dataset columns in
`tokenize_grpo(metadata_fields=[...])` and they are carried through the
collator to the reward, which is then called with a third argument: one
metadata dict per completion, aligned to its prompt. Batches without metadata
keep the 2-arg call, so existing rewards are unaffected.

```python
dataset = dataset.map(
    lambda x: tokenize_grpo(x, tokenizer, max_prompt_length=512,
                            metadata_fields=["expected_stdout"]),
    remove_columns=dataset.column_names,
)

def reward_fn(prompts, completions, metadata):
    return [1.0 if run(c) == m["expected_stdout"] else 0.0
            for c, m in zip(completions, metadata)]
```

Which one?

| Method | When to use |
|--------|-------------|
| [RAFT](#raft) | Start here. Simplest possible online method (best-of-N + SFT), almost nothing to misconfigure. |
| [GRPO](#grpo) | The standard for verifiable rewards (math, code). Group-normalized REINFORCE. |
| [RLOO](#rloo) | Like GRPO but with an unbiased leave-one-out baseline and no std scaling — try it when GRPO is noisy on near-uniform rewards. |
| [Online DPO](#online-dpo) | You trust preference-style learning (DPO) but want the pairs to track the current policy instead of a static dataset. |

### GRPO

Group Relative Policy Optimization. Generates G completions per prompt, scores them, normalizes rewards within each group into advantages, and optimizes a clipped REINFORCE objective.

- **Best for:** Tasks with a verifiable reward signal (math, code, structured output) where writing a scorer is easier than collecting preference pairs
- **Memory:** Very high (generation + two forward passes per batch)
- **Key params:** `reward_fn`; `num_generations` (default 4) — completions per prompt; `beta` (default 0.04) — KL penalty; `epsilon` (default 0.2) — clip ratio
- **Bias-fix options:** `scale_rewards=False` (Dr. GRPO — drop the std division), `loss_type="dr_grpo"` (length-bias fix), `dynamic_sampling=True` (DAPO — exclude zero-variance groups from the loss average)

```python
from grimoire.losses import GRPOMethod

trainer = GrimoireTrainer(
    model=model, tokenizer=tokenizer, config=config,
    loss_fn=GRPOMethod(
        reward_fn=reward_fn,
        tokenizer=tokenizer,
        num_generations=4,
        beta=0.04,
        epsilon=0.2,
        max_new_tokens=512,
    ),
    train_dataset=dataset,
)
trainer.train()
```

**Loss formula:**
```
L_GRPO = -mean(advantages * min(ratio, clipped_ratio)) + beta * KL

ratio         = pi(y|x) / pi_old(y|x)
clipped_ratio = clamp(ratio, 1-epsilon, 1+epsilon)
advantages    = (r - mean(r_group)) / std(r_group)   # normalized within group of G
KL            = mean(log_pi_old(y|x) - log_pi(y|x))
```

### RLOO

REINFORCE Leave-One-Out. Same rollout as GRPO, but each completion's baseline is the mean reward of the *other* G-1 completions in its group, with no std normalization. This is plain REINFORCE with an unbiased baseline — the std division GRPO applies can amplify noise when a group's rewards barely differ; RLOO sidesteps that entirely.

- **Best for:** The same tasks as GRPO, with cleaner statistics — a good default when rewards are near-uniform within groups
- **Memory:** Same as GRPO
- **Key params:** `reward_fn`; `num_generations` (default 4, must be ≥ 2); `beta` (default 0.04) — KL penalty

```python
from grimoire.losses import RLOOMethod

trainer = GrimoireTrainer(
    model=model, tokenizer=tokenizer, config=config,
    loss_fn=RLOOMethod(
        reward_fn=reward_fn,
        tokenizer=tokenizer,
        num_generations=4,
    ),
    train_dataset=dataset,
)
```

**Advantage formula:**
```
A_i = r_i - mean(r_j, j != i)     # no std scaling
```

### Online DPO

Generates G completions per prompt (default 2), scores them, takes the highest- and lowest-reward completion as the (chosen, rejected) pair, and applies the standard DPO loss. Offline DPO's weakness is that its static pairs go off-distribution as the policy moves; here the pairs are re-sampled from the current policy every step.

- **Best for:** Preference-style alignment when you have a reward model or judge instead of a preference dataset
- **Memory:** Very high (generation + reference + policy forwards)
- **Key params:** `reward_fn`; `num_generations` (default 2, higher = best-of-G vs worst-of-G contrast); `beta` (default 0.1)
- **Requires a reference policy:** `ref_model`, or a PEFT model (base weights via `disable_adapter()`)

```python
from grimoire.losses import OnlineDPOMethod

trainer = GrimoireTrainer(
    model=model, tokenizer=tokenizer, config=config,
    loss_fn=OnlineDPOMethod(
        reward_fn=reward_fn,
        tokenizer=tokenizer,
        num_generations=2,
        beta=0.1,
        # ref_model=ref_model,  # omit for PEFT models
    ),
    train_dataset=dataset,
)
```

**Loss formula:** identical to [DPO](#dpo), over pairs generated on-policy each step.

### RAFT

Reward-rAnked FineTuning (best-of-N rejection sampling). Generates G completions per prompt, keeps only the highest-reward one, and applies plain SFT loss to it. The reward only ever enters through the argmax, so there's no advantage estimation, no KL, no reference model — the simplest, hardest-to-misconfigure online method.

- **Best for:** A first online experiment, or when GRPO-style methods are unstable on your reward
- **Memory:** High (generation + one forward pass per batch — no reference model)
- **Key params:** `reward_fn`; `num_generations` (default 4) — higher = stronger selection pressure

```python
from grimoire.losses import RAFTMethod

trainer = GrimoireTrainer(
    model=model, tokenizer=tokenizer, config=config,
    loss_fn=RAFTMethod(
        reward_fn=reward_fn,
        tokenizer=tokenizer,
        num_generations=4,
    ),
    train_dataset=dataset,
)
```

**Loss formula:**
```
L_RAFT = L_SFT(argmax_reward completion per prompt)
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
| GRPO | Prompts only | No | Very high | Verifiable reward signal (math, code) |
| RLOO | Prompts only | No | Very high | GRPO alternative, unbiased baseline |
| Online DPO | Prompts only | Yes | Very high | On-policy preference pairs from a reward fn |
| RAFT | Prompts only | No | High | Simplest online method (best-of-N + SFT) |

## Typical Training Pipelines

1. **Base model → instruction follower:** SFT
2. **Base model → aligned in one step:** ORPO or CPO
3. **SFT model → aligned:** DPO, SimPO, or IPO
4. **SFT model → aligned from user feedback:** KTO
5. **SFT model → aligned with a reward function:** RAFT first, then GRPO or RLOO; Online DPO if you prefer preference-style updates
