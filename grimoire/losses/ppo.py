"""PPO (Proximal Policy Optimization) loss for RLHF.

Paper: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
       arXiv:1707.06347

Generates completions, scores them with a reward function, estimates advantages
using a learned value head, and optimizes with the clipped surrogate objective.

The value head is attached to the model during __init__ so its parameters are
included in the optimizer when the trainer is created afterward.

Loss = policy_loss + vf_coef * value_loss - entropy_coef * entropy + beta * KL

Note: Within grimoire's single-pass training loop, this operates as single-epoch
PPO (K=1). The clipping mechanism is present but trivially satisfied on the first
epoch. All other PPO components (value baseline, advantage estimation, KL penalty)
function normally.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.prompt import PromptCollator


class PPOLoss:
    """PPO loss with learned value baseline for RLHF.

    Attaches a value head to the model during construction. Create this loss
    function BEFORE creating GrimoireTrainer so the value head parameters are
    included in the optimizer.

    Args:
        model: The policy model — a value head will be attached to it.
        ref_model: Frozen reference model for KL penalty.
        reward_fn: Callable (prompts: list[str], completions: list[str]) -> list[float].
        tokenizer: HuggingFace tokenizer for decoding generated tokens.
        beta: KL penalty coefficient.
        clip_eps: PPO clipping epsilon for the surrogate objective.
        vf_coef: Value function loss coefficient.
        entropy_coef: Entropy bonus coefficient.
        max_new_tokens: Maximum number of tokens to generate per completion.
        temperature: Sampling temperature for generation.
    """

    def __init__(
        self,
        model,
        ref_model,
        reward_fn,
        tokenizer,
        beta=0.1,
        clip_eps=0.2,
        vf_coef=0.1,
        entropy_coef=0.01,
        max_new_tokens=256,
        temperature=1.0,
    ):
        self.ref_model = ref_model
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        self.beta = beta
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._pad_token_id = 0

        # Attach value head to model so it's included in the optimizer
        hidden_size = model.config.hidden_size
        value_head = nn.Linear(hidden_size, 1)
        value_head.weight.data.zero_()
        value_head.bias.data.zero_()
        model.value_head = value_head

    def __call__(self, model, batch, training=True):
        if not training:
            return self._eval_forward(model, batch)
        return self._train_forward(model, batch)

    def create_collator(self, pad_token_id):
        self._pad_token_id = pad_token_id
        return PromptCollator(pad_token_id=pad_token_id)

    def _eval_forward(self, model, batch):
        return torch.tensor(0.0, device=batch["input_ids"].device), {}

    def _train_forward(self, model, batch):
        prompt_ids = batch["input_ids"]       # (B, prompt_len) — left-padded
        prompt_mask = batch["attention_mask"]  # (B, prompt_len)
        prompt_len = prompt_ids.shape[1]

        # Step 1: Generate completions (no grad, eval mode)
        full_ids, full_mask = self._generate(model, prompt_ids, prompt_mask)

        # Step 2: Compute rewards
        rewards = self._compute_rewards(full_ids, prompt_ids, prompt_len)

        # Step 3: Build completion mask
        target_ids = full_ids[:, 1:]
        comp_mask = torch.zeros(target_ids.shape, device=full_ids.device)
        if prompt_len - 1 < comp_mask.shape[1]:
            comp_mask[:, prompt_len - 1:] = 1.0
        comp_mask = comp_mask * (target_ids != self._pad_token_id).float()
        comp_lengths = comp_mask.sum(dim=1).clamp(min=1)

        # Step 4: Old log probs, values, ref log probs (no grad)
        with torch.no_grad():
            old_logps, old_values, _ = self._forward_with_values(
                model, full_ids, full_mask
            )
            ref_logps = self._forward_logps(self.ref_model, full_ids, full_mask)

        old_avg_logps = (old_logps * comp_mask).sum(dim=1) / comp_lengths
        ref_avg_logps = (ref_logps * comp_mask).sum(dim=1) / comp_lengths
        old_avg_values = (old_values * comp_mask).sum(dim=1) / comp_lengths

        # KL-penalized rewards and advantages
        kl_per_seq = old_avg_logps - ref_avg_logps
        penalized_rewards = rewards - self.beta * kl_per_seq
        advantages = (penalized_rewards - old_avg_values).detach()

        if advantages.shape[0] > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Step 5: New forward pass (with gradients)
        new_logps, new_values, entropy_per_token = self._forward_with_values(
            model, full_ids, full_mask
        )

        new_avg_logps = (new_logps * comp_mask).sum(dim=1) / comp_lengths
        new_avg_values = (new_values * comp_mask).sum(dim=1) / comp_lengths

        # Step 6: PPO clipped surrogate loss
        ratio = torch.exp(new_avg_logps - old_avg_logps.detach())
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value function loss
        value_loss = F.mse_loss(new_avg_values, penalized_rewards.detach())

        # Entropy bonus
        avg_entropy = (entropy_per_token * comp_mask).sum(dim=1) / comp_lengths
        entropy_bonus = avg_entropy.mean()

        loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy_bonus

        metrics = {
            "policy_loss": policy_loss.detach().item(),
            "value_loss": value_loss.detach().item(),
            "entropy": entropy_bonus.detach().item(),
            "kl": kl_per_seq.mean().item(),
            "mean_reward": rewards.mean().item(),
            "mean_advantage": advantages.mean().item(),
            "mean_value": old_avg_values.mean().item(),
            "clip_fraction": ((ratio - 1.0).abs() > self.clip_eps).float().mean().item(),
            "mean_completion_len": comp_lengths.mean().item(),
        }

        return loss, metrics

    def _generate(self, model, prompt_ids, prompt_mask):
        """Generate one completion per prompt."""
        was_training = model.training
        model.eval()
        with torch.no_grad():
            full_ids = model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self._pad_token_id,
            )
        if was_training:
            model.train()

        full_mask = (full_ids != self._pad_token_id).long()
        full_mask[:, :prompt_ids.shape[1]] = prompt_mask

        return full_ids, full_mask

    def _compute_rewards(self, full_ids, prompt_ids, prompt_len):
        """Decode completions and compute rewards."""
        completion_ids = full_ids[:, prompt_len:]
        prompt_texts = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
        completion_texts = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        rewards = self.reward_fn(prompt_texts, completion_texts)
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, dtype=torch.float32)
        return rewards.to(full_ids.device)

    def _forward_with_values(self, model, input_ids, attention_mask):
        """Forward pass returning log probs, value estimates, and per-token entropy.

        Returns:
            per_token_logps: (B, seq_len-1)
            values: (B, seq_len-1) value estimates from value head
            entropy: (B, seq_len-1) per-token entropy
        """
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask,
            use_cache=False, output_hidden_states=True,
        )
        logits = outputs.logits
        last_hidden = outputs.hidden_states[-1]

        # Per-token log probs (shifted)
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        target_ids = input_ids[:, 1:]
        per_token_logps = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

        # Entropy
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)  # (B, seq_len-1)

        # Value head
        value_head = model.value_head
        values = value_head(last_hidden[:, :-1, :]).squeeze(-1)  # (B, seq_len-1)

        return per_token_logps, values, entropy

    def _forward_logps(self, model, input_ids, attention_mask):
        """Forward pass returning only log probs (for reference model)."""
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
        )
        log_probs = F.log_softmax(outputs.logits[:, :-1, :], dim=-1)
        target_ids = input_ids[:, 1:]
        return log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
