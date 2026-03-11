# Grimoire vs TRL: Implementation Comparison

Detailed comparison of Grimoire's loss implementations against HuggingFace TRL.

## Architecture

| Aspect | Grimoire | TRL |
|--------|----------|-----|
| **Design** | Pure loss callables, shared `GrimoireTrainer` | Each method = full `Trainer` subclass (inherits `transformers.Trainer`) |
| **Training loop** | One loop + `accelerate.Accelerator` directly | `transformers.Trainer` with overridden `compute_loss` |
| **Lines per loss** | ~60-130 | ~500-2000+ per trainer class |
| **Adding a method** | Write a loss function | Write an entire Trainer subclass |

## Log Probability Computation

### Aggregation: Average vs Sum

**Grimoire** — always uses **average** log probs (length-normalized):

```python
return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1)
```

**TRL** — varies by loss type:

| Loss | TRL Default | Grimoire |
|------|-------------|----------|
| DPO | **Sum** | Average |
| ORPO | Average | Average |
| SimPO | Average (via CPO) | Average |
| IPO | Average (via DPO/CPO) | Average |
| CPO (sigmoid) | **Sum** | Average |
| KTO | Configurable | Average |

The DPO difference is meaningful. TRL's sum matches the original paper; Grimoire's average provides better stability across varying response lengths.

### Memory Efficiency

**Grimoire** — `gather + logsumexp` (avoids materializing full vocab tensor):

```python
gathered_logits = torch.gather(shift_logits, dim=2, index=safe_labels.unsqueeze(2)).squeeze(2)
per_token_logps = gathered_logits - torch.logsumexp(shift_logits, dim=-1)
```

**TRL** — `log_softmax + gather` (materializes `[batch, seq, vocab]`):

```python
per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
```

Newer TRL versions use `selective_log_softmax` which may fuse this, but the older/default path materializes the full tensor.

## DPO

| Aspect | Grimoire | TRL |
|--------|----------|-----|
| **Core formula** | `-logsigmoid(β(π_ratio - ref_ratio))` | Same |
| **Log probs** | Average | Sum |
| **Ref model** | Caller-managed frozen copy | 3 strategies: explicit model, PEFT adapter disable, precomputed cache |
| **Label smoothing** | No | Yes (via `robust` loss type) |
| **Loss variants** | Sigmoid only | 15+ (sigmoid, hinge, ipo, exo_pair, nca_pair, robust, sppo_hard, aot, apo_zero, apo_down, discopop, etc.) |
| **F-divergences** | No | reverse_kl, forward_kl, js_divergence, alpha_divergence |
| **WPO weighting** | No | Optional per-sample weighting |
| **Liger kernel** | No | Fused linear DPO loss |

## ORPO

| Aspect | Grimoire | TRL |
|--------|----------|-----|
| **Formula** | `NLL(chosen) + β * -mean(logsigmoid(log_odds))` | Same (sign convention differs slightly) |
| **Odds ratio** | `(logp_c - logp_r) - (log1p(-exp(logp_c)) - log1p(-exp(logp_r)))` | Same |
| **Numerical stability** | **Clamps logps to `max=-1e-4`** to prevent `log1p(-1) = -inf` | No clamping (uses `log1mexp` utility) |
| **Log probs** | Average | Average |
| **TRL status** | — | Deprecated, moved to `trl.experimental` |

Grimoire's clamping is a practical stability improvement. When `exp(logp) → 1` (i.e., the model is very confident), `log1p(-exp(logp))` produces `-inf`. Grimoire guards against this.

## SimPO

| Aspect | Grimoire | TRL |
|--------|----------|-----|
| **Implementation** | Standalone `SimPOLoss` class | `loss_type="simpo"` inside CPO trainer |
| **Formula** | `-logsigmoid(β(avg_chosen - avg_rejected - γ))` | Same, with optional label smoothing |
| **Defaults** | `β=2.0, γ=0.5` | Uses CPO's beta + `simpo_gamma` |
| **Label smoothing** | No | `-(1-α)logsigmoid(βΔ) - α·logsigmoid(-βΔ)` |

## KTO

| Aspect | Grimoire | TRL |
|--------|----------|-----|
| **Core formula** | `λ_d(1-σ(β(ratio-KL))) + λ_u(1-σ(β(KL-ratio)))` | Same |
| **KL estimation** | `mean(policy_logps - ref_logps).clamp(min=0)` from current batch | Explicit mismatched pairs via dataset offset + cross-process gather |
| **Loss averaging** | Separate means, divided by `n_terms` if both present | `torch.cat` all losses, single mean |
| **Data format** | Single batch with `kto_label` bool tensor | Similar, per-example binary labels |
| **TRL status** | — | Deprecated, moved to `trl.experimental` |

TRL's KL estimation is more sophisticated (creates mismatched prompt-completion pairs and gathers across distributed processes). Grimoire uses a simpler batch-level estimate.

## CPO

| Aspect | Grimoire | TRL |
|--------|----------|-----|
| **Formula** | `NLL(chosen) + β * -logsigmoid(β(avg_chosen - avg_rejected))` | `mean(preference_loss) + α * NLL` |
| **SFT weight** | Fixed: `β` scales the preference term | Configurable `cpo_alpha` (0 disables NLL entirely) |
| **Loss variants** | Sigmoid only | sigmoid, hinge, IPO, SimPO |
| **AlphaPO** | No | Probability-based reward transformation |
| **TRL status** | — | Deprecated, moved to `trl.experimental` |

## IPO

| Aspect | Grimoire | TRL |
|--------|----------|-----|
| **Implementation** | Standalone `IPOLoss` class | `loss_type="ipo"` in DPO or CPO trainer |
| **Formula** | `((π_ratio - ref_ratio) - 1/(2β))²` | Same |
| **Ref model** | Required (with ref model, like DPO) | In DPO: required. In CPO: reference-free |

## Concatenated Forward Pass

Both use the same strategy: concatenate chosen + rejected into one batch, single forward pass, split by `len_chosen`.

**Grimoire**: Manual `_pad_dim1` + `torch.cat`, split by index slicing
**TRL**: `concatenated_inputs` / `DataCollatorForPreference`, split via `.chunk(2)` or index slicing

## Features TRL Has That Grimoire Doesn't

- **Label smoothing** across DPO/CPO loss types
- **15+ DPO loss variants** (hinge, exo_pair, nca_pair, robust, sppo_hard, etc.)
- **F-divergence alternatives** (forward KL, JS divergence, alpha divergence)
- **WPO per-sample weighting**
- **Length-diff mode** (`ld_alpha`) for shared prefix token handling
- **AlphaPO** probability-based reward transformation in CPO
- **Precomputed reference log probs** cached in dataset
- **PEFT adapter-based reference** (disable adapter instead of deepcopy)
- **Multi-loss combination** (multiple loss types simultaneously with weights)
- **Liger kernel fusion** for memory-efficient DPO
- **Vision-language model support**

## Features Grimoire Has That TRL Doesn't

- **ORPO numerical stability** via logp clamping
- **Memory-efficient log prob computation** (gather + logsumexp vs log_softmax + gather)
- **Consistent average log probs** across all methods
- **Simplicity** — each loss is a self-contained ~100-line callable, not a 1000+ line trainer

## Summary

Grimoire prioritizes simplicity and correctness. TRL prioritizes breadth and configurability. Grimoire's implementations are mathematically equivalent to TRL's core formulas (with the DPO sum-vs-average exception) but strip away the extensive feature surface. The trade-off is clear: Grimoire is easier to understand, modify, and debug, while TRL supports more experimental variants and production features out of the box.
