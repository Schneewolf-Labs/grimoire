import torch

from ..data.grpo import GRPOCollator
from .utils import get_batch_logps


class GRPOLoss:
    """GRPO (Group Relative Policy Optimization) loss.

    Paper: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
           arXiv:2402.03300

    Generates G completions per prompt, scores them with a reward function,
    computes group-relative advantages, and optimizes with a clipped
    REINFORCE-style objective (like PPO but without a value function).

    Loss = -mean(advantages * min(ratio, clipped_ratio)) + beta * KL

    where ratio = pi(y|x) / pi_old(y|x), advantages are normalized within
    each group of G completions, and KL is estimated from the generation policy.

    Note: GRPO requires ZeRO-2 or lower (or FSDP), not ZeRO-3, because
    model.generate() needs full weight access.
    """

    def __init__(
        self,
        reward_fn,
        tokenizer,
        num_generations=4,
        beta=0.04,
        epsilon=0.2,
        max_new_tokens=512,
        temperature=1.0,
        label_pad_token_id=-100,
    ):
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        self.num_generations = num_generations
        self.beta = beta
        self.epsilon = epsilon
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.label_pad_token_id = label_pad_token_id
        self._pad_token_id = 0

    def __call__(self, model, batch, training=True):
        if not training:
            return self._eval_forward(model, batch)
        return self._train_forward(model, batch)

    def create_collator(self, pad_token_id):
        self._pad_token_id = pad_token_id
        return GRPOCollator(pad_token_id=pad_token_id)

    def _train_forward(self, model, batch):
        input_ids = batch["input_ids"]  # [B, prompt_len]
        attention_mask = batch["attention_mask"]  # [B, prompt_len]
        B = input_ids.size(0)
        G = self.num_generations

        # 1. Generate G completions per prompt (no grad)
        repeated_ids = input_ids.repeat_interleave(G, dim=0)  # [B*G, prompt_len]
        repeated_mask = attention_mask.repeat_interleave(G, dim=0)  # [B*G, prompt_len]

        with torch.no_grad():
            model.eval()
            generated = model.generate(
                input_ids=repeated_ids,
                attention_mask=repeated_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self._pad_token_id,
            )  # [B*G, prompt_len + completion_len]
            model.train()

        # Build labels: mask prompt tokens with label_pad_token_id
        prompt_len = input_ids.size(1)
        gen_labels = generated.clone()
        gen_labels[:, :prompt_len] = self.label_pad_token_id

        # Build attention mask for generated sequences
        gen_attention_mask = (generated != self._pad_token_id).long()

        # 2. Score completions with reward function
        # Decode prompts and completions for reward_fn
        prompt_texts = self.tokenizer.batch_decode(
            repeated_ids, skip_special_tokens=True,
        )
        completion_texts = self.tokenizer.batch_decode(
            generated[:, prompt_len:], skip_special_tokens=True,
        )
        del repeated_ids, repeated_mask  # Free [B*G, prompt_len] tensors

        rewards = self.reward_fn(prompt_texts, completion_texts)  # list[float] or tensor
        del prompt_texts, completion_texts  # Free decoded strings
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, dtype=torch.float32, device=input_ids.device)
        elif rewards.device != input_ids.device:
            rewards = rewards.to(input_ids.device)  # [B*G]

        # 3. Group-relative advantages
        rewards_grouped = rewards.view(B, G)
        group_mean = rewards_grouped.mean(dim=1, keepdim=True)
        group_std = rewards_grouped.std(dim=1, keepdim=True).clamp(min=1e-8)
        advantages = ((rewards_grouped - group_mean) / group_std).view(B * G)  # [B*G]
        del rewards_grouped, group_mean, group_std

        # Capture sequence length before tensors are freed
        completion_length = gen_labels.size(1) - prompt_len

        # 4. Old log-probs (from generation policy, no grad)
        with torch.no_grad():
            old_logits = model(
                input_ids=generated,
                attention_mask=gen_attention_mask,
                use_cache=False,
            ).logits
            old_logps = get_batch_logps(old_logits, gen_labels, self.label_pad_token_id)  # [B*G]
            del old_logits

        # 5. Policy log-probs (WITH grad)
        logits = model(
            input_ids=generated,
            attention_mask=gen_attention_mask,
            use_cache=False,
        ).logits
        del generated, gen_attention_mask  # Free [B*G, seq_len] tensors
        logps = get_batch_logps(logits, gen_labels, self.label_pad_token_id)  # [B*G]
        del logits, gen_labels

        # 6. Clipped REINFORCE loss
        ratio = torch.exp(logps - old_logps.detach())
        clipped = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        # Use per-token style: advantages select which bound matters
        policy_loss = -torch.mean(
            torch.min(ratio * advantages.detach(), clipped * advantages.detach())
        )

        # 7. KL penalty (against old/generation policy)
        kl = (old_logps.detach() - logps).mean()

        loss = policy_loss
        if self.beta > 0:
            loss = loss + self.beta * kl

        metrics = {
            "rewards_mean": rewards.mean().item(),
            "rewards_std": rewards.std().item(),
            "advantages_mean": advantages.mean().item(),
            "kl": kl.detach().item(),
            "policy_loss": policy_loss.detach().item(),
            "ratio_mean": ratio.detach().mean().item(),
            "completion_length": completion_length,
        }

        return loss, metrics

    def _eval_forward(self, model, batch):
        """Eval not meaningful for GRPO (no labels). Return zero loss."""
        loss = torch.tensor(0.0, device=batch["input_ids"].device)
        return loss, {}

    def _get_batch_logps(self, logits, labels):
        """Average log probability per sequence over response tokens only."""
        return get_batch_logps(logits, labels, self.label_pad_token_id)
