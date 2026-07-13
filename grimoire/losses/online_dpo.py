import torch
import torch.nn.functional as F

from .online import OnlineMethod
from .utils import DEFAULT_FUSED_CHUNK_SIZE


class OnlineDPOMethod(OnlineMethod):
    """Online DPO — on-policy preference pairs ranked by a reward function.

    Paper: "Direct Language Model Alignment from Online AI Feedback"
           arXiv:2402.04792

    Two-phase online method (see ``OnlineMethod``):

    1. ``rollout(model, batch)`` — generates G completions per prompt, scores
       them with ``reward_fn``, and takes the highest- and lowest-reward
       completion of each group as the (chosen, rejected) pair. The frozen
       reference is scored over the pair batch in the same no-grad phase.
    2. ``__call__(model, experience)`` — the standard DPO loss on that batch,
       the same math as ``DPOLoss``. The only difference from offline DPO is
       where the pairs come from: the current policy, re-sampled every step,
       instead of a static dataset — so the preference signal tracks the
       policy as it moves instead of going off-distribution.

    Loss = -mean((1-eps)*log(sigmoid(x)) + eps*log(sigmoid(-x)))
    x    = beta * ((logp_chosen - logp_rejected) - (ref_chosen - ref_rejected))

    Like DPO, this REQUIRES a reference policy: ``ref_model`` if given, else
    the base model via ``disable_adapter()`` for PEFT models. rollout() raises
    if neither is available.

    ``num_generations`` defaults to 2 (one pair from one sample pair); higher
    values sample best-of-G / worst-of-G, giving a stronger contrast per pair
    at more generation cost.
    """

    def __init__(
        self,
        reward_fn,
        tokenizer,
        num_generations=2,
        beta=0.1,
        label_smoothing=0.0,
        max_new_tokens=512,
        temperature=1.0,
        label_pad_token_id=-100,
        ref_model=None,
        fused=True,
        fused_chunk_size=DEFAULT_FUSED_CHUNK_SIZE,
    ):
        if num_generations < 2:
            raise ValueError(
                "Online DPO needs num_generations >= 2 to form a "
                "(chosen, rejected) pair per prompt."
            )
        super().__init__(
            reward_fn=reward_fn,
            tokenizer=tokenizer,
            num_generations=num_generations,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            label_pad_token_id=label_pad_token_id,
            ref_model=ref_model,
            fused=fused,
            fused_chunk_size=fused_chunk_size,
        )
        self.beta = beta
        self.label_smoothing = label_smoothing

    # --- online phase ----------------------------------------------------- #

    @torch.no_grad()
    def rollout(self, model, batch):
        """Turn a prompt-only batch into an on-policy preference batch.

        Returns ``input_ids`` / ``attention_mask`` / ``labels`` with the B
        chosen sequences stacked above the B rejected sequences (the same
        concatenated layout the offline preference losses forward in one
        pass), plus ``ref_chosen_logps`` / ``ref_rejected_logps``, a per-pair
        ``pair_mask``, and ``rollout_metrics``.

        Groups where every completion scored the same produce a degenerate
        pair — best and worst are reward-ties (possibly the very same row),
        so the "preference" is arbitrary and training on it is pure noise.
        Those pairs are masked out of the loss (``pair_mask`` 0); the
        ``tied_pairs`` metric reports the masked fraction.
        """
        B = batch["input_ids"].size(0)
        G = self.num_generations

        gen = self._generate_and_score(model, batch)
        rewards_grouped = gen["rewards"].view(B, G)

        # Best-of-G is chosen, worst-of-G is rejected. Rows in gen are grouped
        # by prompt (i*G..(i+1)*G-1), so offset the within-group argmax/argmin.
        row_offset = torch.arange(B, device=rewards_grouped.device) * G
        chosen_idx = row_offset + rewards_grouped.argmax(dim=1)
        rejected_idx = row_offset + rewards_grouped.argmin(dim=1)
        order = torch.cat([chosen_idx, rejected_idx])  # [2B], chosen block first

        input_ids = gen["input_ids"][order]
        attention_mask = gen["attention_mask"][order]
        labels = gen["labels"][order]
        del gen

        # Frozen reference over the pair batch (still no grad). Required: the
        # DPO objective has no reference-free form.
        ref_logps = self._reference_logps(
            model, input_ids, attention_mask, labels, required=True
        )

        chosen_rewards = rewards_grouped.max(dim=1).values
        rejected_rewards = rewards_grouped.min(dim=1).values
        # A reward-tied pair expresses no preference — mask it out of the loss
        # rather than training toward an arbitrary ordering.
        pair_mask = (chosen_rewards > rejected_rewards).float()  # [B]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "ref_chosen_logps": ref_logps[:B],
            "ref_rejected_logps": ref_logps[B:],
            "pair_mask": pair_mask,
            "rollout_metrics": {
                "rewards_mean": rewards_grouped.mean().item(),
                "rewards_std": rewards_grouped.std().item(),
                "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
                "tied_pairs": 1.0 - pair_mask.mean().item(),
                "completion_length": float(input_ids.size(1) - batch["input_ids"].size(1)),
            },
        }

    # --- loss phase (pure) ------------------------------------------------ #

    def __call__(self, model, batch, training=True):
        if not training:
            # Eval batches are prompt-only (the trainer runs no rollout there).
            return torch.zeros((), device=batch["input_ids"].device), {}

        len_chosen = batch["input_ids"].size(0) // 2
        ref_chosen_logps = batch["ref_chosen_logps"]
        ref_rejected_logps = batch["ref_rejected_logps"]

        # Policy log-probs (WITH grad): one forward over the concatenated pairs.
        all_logps = self._avg_logps(
            model, batch["input_ids"], batch["attention_mask"], batch["labels"]
        )
        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        # DPO loss: -log sigmoid(beta * (pi_logratio - ref_logratio))
        # With label smoothing: -(1-eps)*logsigmoid(x) - eps*logsigmoid(-x)
        # Reward-tied pairs carry no preference signal; average over the
        # informative pairs only (pair_mask keeps the graph intact when a
        # whole batch is tied).
        pi_logratios = chosen_logps - rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits_diff = self.beta * (pi_logratios - ref_logratios)
        pair_mask = batch["pair_mask"]
        per_pair_loss = -(
            (1 - self.label_smoothing) * F.logsigmoid(logits_diff)
            + self.label_smoothing * F.logsigmoid(-logits_diff)
        )
        loss = (per_pair_loss * pair_mask).sum() / pair_mask.sum().clamp(min=1)

        # Implicit rewards: beta * (log pi(y|x) - log pi_ref(y|x)). Prefixed
        # "implicit" — the plain reward_* metrics are reward_fn scores from
        # the rollout phase.
        implicit_chosen = self.beta * (chosen_logps - ref_chosen_logps).detach()
        implicit_rejected = self.beta * (rejected_logps - ref_rejected_logps).detach()

        metrics = dict(batch["rollout_metrics"])
        metrics.update(
            implicit_chosen_reward=implicit_chosen.mean().item(),
            implicit_rejected_reward=implicit_rejected.mean().item(),
            implicit_reward_margin=(implicit_chosen - implicit_rejected).mean().item(),
            implicit_reward_accuracy=(implicit_chosen > implicit_rejected).float().mean().item(),
        )
        return loss, metrics
