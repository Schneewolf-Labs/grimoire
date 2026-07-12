# Loss Formulas

Side-by-side comparison of the math behind each training method.

## SFT

Standard next-token prediction. Prompt tokens are masked with `-100` so only the response contributes to loss.

```
L_SFT = CrossEntropy(logits, labels)    # labels have prompt tokens masked
```

## ORPO

[arXiv:2403.07691](https://arxiv.org/abs/2403.07691)

SFT loss on chosen + odds ratio preference term. No reference model.

```
L_ORPO = L_SFT(chosen) + beta * L_OR

L_SFT  = CrossEntropy on chosen response tokens (prompt masked)
L_OR   = -mean(log(sigmoid(log_odds_ratio)))

log_odds_ratio = log(P_c / (1 - P_c)) - log(P_r / (1 - P_r))
               = (log_P_c - log_P_r) - (log1p(-exp(log_P_c)) - log1p(-exp(log_P_r)))
```

- `P_c`, `P_r` = average log probability of chosen/rejected responses
- `beta` = weight of preference term (default 0.1)
- `log1p` formulation avoids numerical issues near P=1

## DPO

[arXiv:2305.18290](https://arxiv.org/abs/2305.18290)

Log-sigmoid on the gap between policy and reference log-ratios. Requires frozen reference model.

```
L_DPO = -mean(log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected))))

log_ratio(y) = avg_logp_pi(y|x) - avg_logp_ref(y|x)
```

- `pi` = policy model (being trained)
- `pi_ref` = reference model (frozen copy)
- `beta` = temperature controlling divergence from reference (default 0.1)

**Implicit rewards:** `r(y|x) = beta * (log pi(y|x) - log pi_ref(y|x))`

## SimPO

[arXiv:2405.14734](https://arxiv.org/abs/2405.14734)

Like DPO but reference-free — uses average log probability as an implicit reward, with a margin.

```
L_SimPO = -mean(log(sigmoid(beta * (avg_logp_chosen - avg_logp_rejected - gamma))))
```

- `beta` = scaling factor (default 2.0, higher than DPO since no reference baseline)
- `gamma` = target reward margin (default 0.5)

## KTO

[arXiv:2402.01306](https://arxiv.org/abs/2402.01306)

Unpaired binary feedback. Each example is independently good or bad. Requires frozen reference model.

```
L_KTO = mean(desirable_losses) + mean(undesirable_losses)

desirable_loss   = lambda_d * (1 - sigmoid(beta * (log_ratio - KL_ref)))
undesirable_loss = lambda_u * (1 - sigmoid(beta * (KL_ref - log_ratio)))

log_ratio = avg_logp_policy(y|x) - avg_logp_ref(y|x)
KL_ref    = clamp(mean(log_ratio), min=0)
```

- `beta` = scaling factor (default 0.1)
- `lambda_d` = desirable weight (default 1.0)
- `lambda_u` = undesirable weight (default 1.0, increase for loss aversion)
- `KL_ref` is estimated from the batch

## CPO

[arXiv:2312.02143](https://arxiv.org/abs/2312.02143)

SFT + contrastive preference. Reference-free, like ORPO but with a theoretically cleaner preference term.

```
L_CPO = L_SFT(chosen) + beta * L_preference

L_SFT        = CrossEntropy on chosen response tokens (prompt masked)
L_preference = -mean(log(sigmoid(beta * (avg_logp_chosen - avg_logp_rejected))))
```

- `beta` = weight of preference term (default 0.1)

**ORPO vs CPO:** ORPO uses odds ratio `log(P/(1-P))`, CPO uses raw log probabilities directly. CPO's gradient signal is simpler.

## IPO

[arXiv:2310.12036](https://arxiv.org/abs/2310.12036)

Squared loss variant of DPO. Prevents overfitting on noisy preference data. Requires frozen reference model.

```
L_IPO = mean((log_ratio_chosen - log_ratio_rejected - 1/(2*beta))^2)

log_ratio(y) = avg_logp_pi(y|x) - avg_logp_ref(y|x)
```

- `beta` = controls target margin `1/(2*beta)` (default 0.1, so margin = 5.0)

**DPO vs IPO:** DPO's log-sigmoid can saturate, causing the model to overfit to mislabeled preferences. IPO's squared loss keeps pushing toward the target margin without saturating.

## Comparison

| Method | Loss Type | Reference Model | Key Innovation |
|--------|-----------|-----------------|----------------|
| SFT | Cross-entropy | No | Standard next-token prediction |
| ORPO | Cross-entropy + log-sigmoid | No | Odds ratio as preference signal |
| DPO | Log-sigmoid | Yes | Policy vs reference log-ratio |
| SimPO | Log-sigmoid | No | Margin-based, no reference |
| KTO | Sigmoid | Yes | Unpaired binary feedback |
| CPO | Cross-entropy + log-sigmoid | No | Contrastive (simpler than odds ratio) |
| IPO | Squared | Yes | Robust to noisy labels |

## Implementation Details

All preference methods in Grimoire share these patterns:

- **Single forward pass:** Chosen and rejected sequences are concatenated into one batch, run through the model once, then split. This is faster and required for FSDP compatibility.
- **Average log probabilities:** Per-token log probs are averaged over response length, making the loss invariant to response length differences.
- **Prompt masking:** Prompt tokens are set to `-100` in labels and excluded from log probability computation via `loss_mask`.
