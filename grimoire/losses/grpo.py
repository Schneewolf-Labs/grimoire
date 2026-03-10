"""GRPO (Group Relative Policy Optimization) loss.

Paper: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
       arXiv:2402.03300

Generates multiple completions per prompt, computes group-relative advantages
(no value network needed), and applies a KL penalty against a frozen reference model.

Loss = -mean(advantage * avg_logp_policy) + beta * KL(policy || ref)

Advantages are computed per-group: A_i = (r_i - mean(r)) / (std(r) + eps)
"""

import torch
import torch.nn.functional as F

from ..data.prompt import PromptCollator


class GRPOLoss:
    """Group Relative Policy Optimization loss.

    Online RL method that generates multiple completions per prompt and uses
    group-relative advantages as the baseline (no value network needed).

    Args:
        ref_model: Frozen reference model for KL penalty.
        reward_fn: Callable (prompts: list[str], completions: list[str]) -> list[float].
        tokenizer: HuggingFace tokenizer for decoding generated tokens.
        num_generations: Number of completions to sample per prompt (G).
        beta: KL penalty coefficient.
        max_new_tokens: Maximum number of tokens to generate per completion.
        temperature: Sampling temperature for generation.
    """

    def __init__(
        self,
        ref_model,
        reward_fn,
        tokenizer,
        num_generations=4,
        beta=0.04,
        max_new_tokens=256,
        temperature=1.0,
    ):
        self.ref_model = ref_model
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        self.num_generations = num_generations
        self.beta = beta
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._pad_token_id = 0

    def __call__(self, model, batch, training=True):
        if not training:
            return self._eval_forward(model, batch)
        return self._train_forward(model, batch)

    def create_collator(self, pad_token_id):
        self._pad_token_id = pad_token_id
        return PromptCollator(pad_token_id=pad_token_id)

    def _eval_forward(self, model, batch):
        """Eval generates completions and returns mean reward as the loss (lower = better)."""
        return torch.tensor(0.0, device=batch["input_ids"].device), {}

    def _train_forward(self, model, batch):
        prompt_ids = batch["input_ids"]       # (B, prompt_len) — left-padded
        prompt_mask = batch["attention_mask"]  # (B, prompt_len)
        batch_size = prompt_ids.shape[0]
        prompt_len = prompt_ids.shape[1]
        G = self.num_generations

        # Step 1: Generate G completions per prompt
        full_ids, full_mask = self._generate(model, prompt_ids, prompt_mask)
        # full_ids: (B*G, prompt_len + gen_len)

        # Step 2: Compute rewards
        rewards = self._compute_rewards(full_ids, prompt_ids, prompt_len, batch_size, G)
        # rewards: (B*G,)

        # Step 3: Compute group-relative advantages
        rewards_grouped = rewards.view(batch_size, G)
        mean_r = rewards_grouped.mean(dim=1, keepdim=True)
        std_r = rewards_grouped.std(dim=1, keepdim=True)
        advantages = ((rewards_grouped - mean_r) / (std_r + 1e-8)).view(-1)  # (B*G,)

        # Step 4: Compute per-token log probabilities under policy (with gradients)
        policy_logps, comp_mask = self._get_completion_logps(
            model, full_ids, full_mask, prompt_len
        )

        # Step 5: Compute per-token log probabilities under reference (no gradients)
        with torch.no_grad():
            ref_logps, _ = self._get_completion_logps(
                self.ref_model, full_ids, full_mask, prompt_len
            )

        # Step 6: Compute loss
        comp_lengths = comp_mask.sum(dim=1).clamp(min=1)

        # Average log prob per completion (length-normalized)
        avg_policy_logps = (policy_logps * comp_mask).sum(dim=1) / comp_lengths
        avg_ref_logps = (ref_logps * comp_mask).sum(dim=1) / comp_lengths

        # Policy gradient: -advantage * log_prob
        pg_loss = -(advantages.detach() * avg_policy_logps).mean()

        # KL penalty: avg(log_pi - log_ref), positive when policy diverges from ref
        kl = (avg_policy_logps - avg_ref_logps).mean()

        loss = pg_loss + self.beta * kl

        metrics = {
            "pg_loss": pg_loss.detach().item(),
            "kl": kl.detach().item(),
            "mean_reward": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            "mean_advantage": advantages.mean().item(),
            "mean_completion_len": comp_lengths.mean().item(),
        }

        return loss, metrics

    def _generate(self, model, prompt_ids, prompt_mask):
        """Generate G completions per prompt.

        Returns:
            full_ids: (B*G, prompt_len + max_gen_len) — prompt + generated tokens
            full_mask: (B*G, prompt_len + max_gen_len) — attention mask
        """
        G = self.num_generations

        # Repeat each prompt G times
        expanded_ids = prompt_ids.repeat_interleave(G, dim=0)    # (B*G, prompt_len)
        expanded_mask = prompt_mask.repeat_interleave(G, dim=0)  # (B*G, prompt_len)

        was_training = model.training
        model.eval()
        with torch.no_grad():
            full_ids = model.generate(
                input_ids=expanded_ids,
                attention_mask=expanded_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self._pad_token_id,
            )
        if was_training:
            model.train()

        # Build attention mask for full sequence (1 for real tokens, 0 for padding)
        full_mask = (full_ids != self._pad_token_id).long()
        # Preserve original prompt mask (left-pad tokens should stay masked)
        full_mask[:, :expanded_ids.shape[1]] = expanded_mask

        return full_ids, full_mask

    def _compute_rewards(self, full_ids, prompt_ids, prompt_len, batch_size, G):
        """Decode completions and compute rewards via reward_fn."""
        # Extract completion tokens (everything after prompt)
        completion_ids = full_ids[:, prompt_len:]

        # Decode prompts and completions
        # Use the original (non-repeated) prompts, then repeat for matching
        prompt_texts = self.tokenizer.batch_decode(
            prompt_ids, skip_special_tokens=True
        )
        # Repeat each prompt G times to match completions
        expanded_prompts = [p for p in prompt_texts for _ in range(G)]

        completion_texts = self.tokenizer.batch_decode(
            completion_ids, skip_special_tokens=True
        )

        rewards = self.reward_fn(expanded_prompts, completion_texts)
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, dtype=torch.float32)
        return rewards.to(full_ids.device)

    def _get_completion_logps(self, model, input_ids, attention_mask, prompt_len):
        """Compute per-token log probabilities for completion tokens only.

        Returns:
            per_token_logps: (B*G, seq_len-1) log probs at each position
            completion_mask: (B*G, seq_len-1) mask for completion tokens only
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits  # (B*G, seq_len, vocab)

        # Shift for next-token prediction: logits[t] predicts token[t+1]
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        target_ids = input_ids[:, 1:]

        # Gather log probs for actual tokens
        per_token_logps = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

        # Mask: only completion tokens (positions >= prompt_len-1 in shifted sequence)
        completion_mask = torch.zeros_like(per_token_logps)
        if prompt_len - 1 < completion_mask.shape[1]:
            completion_mask[:, prompt_len - 1:] = 1.0

        # Also mask padding tokens in the completion
        pad_mask = (target_ids != self._pad_token_id).float()
        completion_mask = completion_mask * pad_mask

        return per_token_logps, completion_mask
