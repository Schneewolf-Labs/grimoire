from .grpo import GRPOMethod
from .utils import DEFAULT_FUSED_CHUNK_SIZE


class RLOOMethod(GRPOMethod):
    """RLOO (REINFORCE Leave-One-Out) — an online RL method.

    Paper: "Back to Basics: Revisiting REINFORCE-Style Optimization for
           Learning from Human Feedback in LLMs" arXiv:2402.14740

    Identical rollout to GRPO — G completions per prompt scored by
    ``reward_fn`` — but each completion's baseline is the mean reward of the
    OTHER G-1 completions in its group, with no std normalization:

        A_i = r_i - mean(r_j, j != i)

    The leave-one-out baseline is unbiased (a completion never baselines
    itself) and skipping the std division avoids GRPO's difficulty bias,
    where near-uniform-reward groups get their tiny advantages amplified by
    a small denominator. This is plain REINFORCE with a good baseline —
    "normal RL", no clipping tricks doing any work (the inherited ratio/clip
    machinery is inert at one update per generation step, exactly as in GRPO).

    The optional KL penalty is inherited from GRPO as a loss term (k3
    estimator against ``ref_model`` or the PEFT base model); the RLOO paper
    folds KL into the reward instead, which is equivalent in spirit. Pass
    beta=0 to disable.
    """

    def __init__(
        self,
        reward_fn,
        tokenizer,
        num_generations=4,
        beta=0.04,
        max_new_tokens=512,
        temperature=1.0,
        label_pad_token_id=-100,
        ref_model=None,
        fused=True,
        fused_chunk_size=DEFAULT_FUSED_CHUNK_SIZE,
    ):
        if num_generations < 2:
            raise ValueError(
                "RLOO needs num_generations >= 2 — the leave-one-out baseline "
                "is the mean of the other G-1 completions in the group."
            )
        super().__init__(
            reward_fn=reward_fn,
            tokenizer=tokenizer,
            num_generations=num_generations,
            beta=beta,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            label_pad_token_id=label_pad_token_id,
            ref_model=ref_model,
            fused=fused,
            fused_chunk_size=fused_chunk_size,
        )

    def _compute_advantages(self, rewards, B, G):
        """Leave-one-out baseline: A_i = r_i - mean(r_j, j != i), unscaled."""
        rewards_grouped = rewards.view(B, G)
        baseline = (rewards_grouped.sum(dim=1, keepdim=True) - rewards_grouped) / (G - 1)
        return (rewards_grouped - baseline).view(B * G)
