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

## GRPO

[arXiv:2402.03300](https://arxiv.org/abs/2402.03300)

Group Relative Policy Optimization — an **online** method (`GRPOMethod`), not a plain loss. Its `rollout` phase generates G completions per prompt, scores them with a reward function, and normalizes rewards within each group into advantages; the loss phase below then optimizes a clipped REINFORCE objective over that experience. No pre-labeled data needed.

```
L_GRPO = -mean(advantages * min(ratio, clipped_ratio)) + beta * KL

ratio         = pi(y|x) / pi_old(y|x)
clipped_ratio = clamp(ratio, 1 - epsilon, 1 + epsilon)
advantages    = (r - mean(r_group)) / std(r_group)
KL            = mean(log_pi_old(y|x) - log_pi(y|x))
```

- `pi` = policy model (being trained)
- `pi_old` = generation policy (same model, frozen snapshot from this step's generation)
- `r` = reward scores from `reward_fn`
- `G` = `num_generations` — completions sampled per prompt
- `beta` = KL penalty weight (default 0.04)
- `epsilon` = clip ratio (default 0.2)

**How it works:** For each prompt, `G` completions are sampled, scored by `reward_fn`, and normalized relative to the group mean and std. The clipped ratio prevents large policy updates (same mechanism as PPO), while the KL term keeps the policy from drifting too far from the generation distribution.

**Bias-fix options** (from [Dr. GRPO, arXiv:2503.20783](https://arxiv.org/abs/2503.20783) and [DAPO, arXiv:2503.14476](https://arxiv.org/abs/2503.14476)):

- `scale_rewards=False` — drop the `/ std(r_group)` division. The std normalization up-weights groups whose rewards barely differ (difficulty bias).
- `loss_type="dr_grpo"` — aggregate per-token log-probs as `sum / max_new_tokens` instead of per-sequence mean, removing the length bias that makes long wrong answers cheaper than short ones.
- `dynamic_sampling=True` — exclude zero-variance groups (every completion got the same reward, so every advantage is 0) from the loss average so they don't dilute the gradient.

## RLOO

[arXiv:2402.14740](https://arxiv.org/abs/2402.14740)

REINFORCE Leave-One-Out (`RLOOMethod`) — an **online** method with the same rollout as GRPO, but a leave-one-out advantage baseline and no std normalization. Plain REINFORCE with an unbiased baseline.

```
L_RLOO = -mean(A_i * log pi(y_i|x)) + beta * KL      (structurally as L_GRPO, ratio == 1)

A_i = r_i - mean(r_j, j != i)     # baseline = mean of the OTHER G-1 completions
```

- The baseline never includes the completion it baselines, so it's unbiased
- No std division — avoids amplifying noise on near-uniform-reward groups
- `num_generations` must be ≥ 2; KL term as in GRPO (pass `beta=0` to disable)

## Online DPO

[arXiv:2402.04792](https://arxiv.org/abs/2402.04792)

Online DPO (`OnlineDPOMethod`) — an **online** method whose rollout generates G completions per prompt, scores them with `reward_fn`, and takes the best/worst as the (chosen, rejected) pair. The loss is then exactly the DPO loss:

```
L_OnlineDPO = -mean(log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected))))

log_ratio(y) = avg_logp_pi(y|x) - avg_logp_ref(y|x)
chosen       = argmax_g r(y_g),  rejected = argmin_g r(y_g)
```

- Same math as offline DPO; the difference is that pairs are re-sampled from the current policy every step, so the preference signal never goes off-distribution
- Requires a reference policy (`ref_model` or PEFT `disable_adapter()`)

## RAFT

[arXiv:2304.06767](https://arxiv.org/abs/2304.06767)

Reward-rAnked FineTuning (`RAFTMethod`) — an **online** method implementing best-of-N rejection sampling. The rollout generates G completions per prompt, keeps only the highest-reward one, and the loss is plain SFT on the winner:

```
L_RAFT = CrossEntropy on the argmax-reward completion's tokens (prompt masked)
```

- Reward enters only through the argmax — no advantages, no KL, no reference model
- The simplest online method; a strong baseline before GRPO/RLOO

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
| GRPO | Clipped REINFORCE | No | Group-relative reward normalization |
| RLOO | REINFORCE | No | Unbiased leave-one-out baseline |
| Online DPO | Log-sigmoid | Yes | On-policy preference pairs from a reward fn |
| RAFT | Cross-entropy | No | Best-of-N rejection sampling |

## Implementation Details

All preference methods in Grimoire share these patterns:

- **Single forward pass:** Chosen and rejected sequences are concatenated into one batch, run through the model once, then split. This is faster and required for FSDP compatibility.
- **Average log probabilities:** Per-token log probs are averaged over response length, making the loss invariant to response length differences.
- **Prompt masking:** Prompt tokens are set to `-100` in labels and excluded from log probability computation via `loss_mask`.
