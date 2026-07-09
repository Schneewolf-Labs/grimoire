import logging

import torch

from ..data.grpo import GRPOCollator
from .utils import (
    DEFAULT_FUSED_CHUNK_SIZE,
    _disable_grad_checkpointing,
    forward_per_token_logps,
    masked_avg_logps,
)

logger = logging.getLogger(__name__)


class GRPOLoss:
    """GRPO (Group Relative Policy Optimization) loss.

    Paper: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
           arXiv:2402.03300

    Generates G completions per prompt, scores them with a reward function,
    computes group-relative advantages, and optimizes with a REINFORCE-style
    objective (like PPO but without a value function).

    Loss = -mean(advantages * min(ratio, clipped_ratio)) + beta * KL

    where ratio = pi(y|x) / pi_old(y|x) and advantages are normalized within
    each group of G completions. Grimoire performs a single policy update per
    generation step (mu=1 in the paper), so pi_old == pi and the ratio is
    identically 1 — the clipping never activates and the objective reduces to
    REINFORCE with group-normalized advantages. The ratio form is kept for
    parity with the paper's objective.

    The KL penalty is estimated against a frozen reference policy using the
    non-negative k3 estimator: exp(ref - pi) - (ref - pi) - 1. The reference
    is ``ref_model`` if provided, else the base model via ``disable_adapter()``
    for PEFT models. If neither is available, the KL term is skipped (with a
    warning) — pass beta=0 to silence it.

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
        ref_model=None,
        fused=True,
        fused_chunk_size=DEFAULT_FUSED_CHUNK_SIZE,
    ):
        if ref_model is not None and ref_model.training:
            raise ValueError("ref_model must be in eval mode (call ref_model.eval() first)")
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        self.num_generations = num_generations
        self.beta = beta
        self.epsilon = epsilon
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.label_pad_token_id = label_pad_token_id
        self.ref_model = ref_model
        self.fused = fused
        self.fused_chunk_size = fused_chunk_size
        self._pad_token_id = 0
        self._warned_no_ref = False

    def _avg_logps(self, model, input_ids, attention_mask, labels):
        per_token_logps, loss_mask = forward_per_token_logps(
            model, input_ids, attention_mask, labels, self.label_pad_token_id,
            fused=self.fused, fused_chunk_size=self.fused_chunk_size,
        )
        return masked_avg_logps(per_token_logps, loss_mask)

    def __call__(self, model, batch, training=True):
        if not training:
            return self._eval_forward(model, batch)
        return self._train_forward(model, batch)

    def create_collator(self, pad_token_id):
        self._pad_token_id = pad_token_id
        return GRPOCollator(pad_token_id=pad_token_id)

    def _train_forward(self, model, batch):
        input_ids = batch["input_ids"]  # [B, prompt_len], left-padded
        attention_mask = batch["attention_mask"]  # [B, prompt_len]
        B = input_ids.size(0)
        G = self.num_generations

        # 1. Generate G completions per prompt (no grad)
        repeated_ids = input_ids.repeat_interleave(G, dim=0)  # [B*G, prompt_len]
        repeated_mask = attention_mask.repeat_interleave(G, dim=0)  # [B*G, prompt_len]

        with _disable_grad_checkpointing(model), torch.no_grad():
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

        prompt_len = input_ids.size(1)
        completion_ids = generated[:, prompt_len:]
        completion_length = completion_ids.size(1)

        # Mask completion tokens after the first EOS (generate() pads there,
        # and pad_token_id may collide with real token IDs, e.g. pad == eos).
        completion_mask = self._completion_mask(completion_ids)

        # Attention mask: collator's left-pad mask for the prompt, EOS-derived
        # mask for the completion
        gen_attention_mask = torch.cat([repeated_mask, completion_mask], dim=1)

        # Labels: mask the prompt region and padding after EOS
        gen_labels = generated.clone()
        gen_labels[:, :prompt_len] = self.label_pad_token_id
        gen_labels[:, prompt_len:] = torch.where(
            completion_mask.bool(), completion_ids, self.label_pad_token_id
        )

        # 2. Score completions with reward function
        prompt_texts = self.tokenizer.batch_decode(
            repeated_ids, skip_special_tokens=True,
        )
        completion_texts = self.tokenizer.batch_decode(
            completion_ids, skip_special_tokens=True,
        )
        del repeated_ids, repeated_mask, completion_ids

        rewards = self.reward_fn(prompt_texts, completion_texts)  # list[float] or tensor
        del prompt_texts, completion_texts
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

        # 4. Policy log-probs (WITH grad)
        logps = self._avg_logps(model, generated, gen_attention_mask, gen_labels)  # [B*G]

        # Generation policy log-probs: the model is only updated once per
        # generation step, so pi_old == pi — no extra forward pass needed.
        old_logps = logps.detach()

        # 5. Reference log-probs for the KL penalty (no grad)
        ref_logps = None
        if self.beta > 0:
            ref_logps = self._reference_logps(model, generated, gen_attention_mask, gen_labels)
        del generated, gen_attention_mask, gen_labels

        # 6. Clipped REINFORCE loss (ratio == 1, see class docstring)
        ratio = torch.exp(logps - old_logps)
        clipped = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        policy_loss = -torch.mean(
            torch.min(ratio * advantages.detach(), clipped * advantages.detach())
        )

        # 7. KL penalty against the reference policy (k3 estimator)
        if ref_logps is not None:
            log_ratio = ref_logps - logps
            kl = torch.exp(log_ratio) - log_ratio - 1.0
            kl = kl.mean()
            loss = policy_loss + self.beta * kl
        else:
            kl = torch.zeros((), device=logps.device)
            loss = policy_loss

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

    def _completion_mask(self, completion_ids):
        """Mask of real completion tokens: 1 up to and including the first EOS."""
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is None:
            return torch.ones_like(completion_ids)

        comp_len = completion_ids.size(1)
        is_eos = completion_ids == eos_id
        has_eos = is_eos.any(dim=1)
        first_eos = torch.where(
            has_eos,
            is_eos.int().argmax(dim=1),
            torch.full_like(has_eos, comp_len, dtype=torch.long),
        )
        positions = torch.arange(comp_len, device=completion_ids.device)
        return (positions.unsqueeze(0) <= first_eos.unsqueeze(1)).long()

    def _reference_logps(self, model, input_ids, attention_mask, labels):
        """Average log-probs under the frozen reference policy, or None if unavailable."""
        with torch.no_grad():
            if self.ref_model is not None:
                return self._avg_logps(self.ref_model, input_ids, attention_mask, labels)
            elif hasattr(model, "disable_adapter"):
                with _disable_grad_checkpointing(model), model.disable_adapter():
                    return self._avg_logps(model, input_ids, attention_mask, labels)
            else:
                if not self._warned_no_ref:
                    logger.warning(
                        "GRPOLoss: beta > 0 but no reference policy is available "
                        "(no ref_model and the model has no disable_adapter()). "
                        "Skipping the KL penalty — pass ref_model or set beta=0."
                    )
                    self._warned_no_ref = True
                return None

    def _eval_forward(self, model, batch):
        """Eval not meaningful for GRPO (no labels). Return zero loss."""
        loss = torch.tensor(0.0, device=batch["input_ids"].device)
        return loss, {}
