import copy
import logging
from contextlib import nullcontext

import torch

from .config import GRPOConfig
from .data.grpo import PromptCollator
from .losses.utils import _per_token_logps, _disable_grad_checkpointing
from .trainer import GrimoireTrainer

logger = logging.getLogger(__name__)


class GRPOTrainer(GrimoireTrainer):
    """On-policy GRPO trainer. Same lifecycle/callbacks/checkpointing as GrimoireTrainer.

    Unlike the pure-loss training methods, GRPO is inherently online and stateful:
    the trainer owns rollout generation, reward scoring, advantage computation, and
    the reference/old-policy forward passes. The loss (``GRPOLoss``) stays a pure
    tensor->scalar function. The reward enters as one synchronous ``reward_fn``
    callable — the trainer never knows what is behind it.

    Per optimization step (one prompt batch):
      1. Sample ``num_generations`` completions per prompt via ``model.generate``.
      2. Decode and call ``reward_fn(prompts, completions, **columns)`` -> floats.
      3. Group-normalize rewards -> advantages (per ``loss_fn.scale_rewards``).
      4. Forward passes for current (+ old if num_iterations>1, + ref if beta>0)
         per-token log-probs. Reference log-probs come FREE from disabling the
         LoRA adapter when PEFT is used — only a separate ref copy is made when
         ``peft_config is None`` and ``beta > 0``.
      5. ``loss_fn(...)`` -> backward -> step (repeated ``num_iterations`` times).
      6. Emit metrics to callbacks (reward/mean, reward/std, kl, clip_frac,
         ratio_mean, completion_length/mean) and expose ``self.sample_completions``
         (a few prompt/completion/reward rows) for callbacks to stream/log.

    Reproducibility honors ``config.seed`` for sampling (set by the base trainer).
    """

    def __init__(
        self,
        *,
        model,
        tokenizer,
        config: GRPOConfig,
        loss_fn,
        reward_fn,
        train_dataset,
        eval_dataset=None,
        peft_config=None,
        callbacks=None,
        data_collator=None,
    ):
        if config.use_vllm:
            raise NotImplementedError(
                "The vLLM generation backend is not implemented yet. "
                "Set use_vllm=False to generate with HuggingFace .generate()."
            )

        self.reward_fn = reward_fn
        self.sample_completions = []

        # Reference policy. With LoRA we get it free by disabling the adapter
        # (base weights = reference), so no second model is needed. Only fall
        # back to a frozen deepcopy when there is no adapter and beta > 0.
        self._needs_ref = getattr(loss_fn, "beta", 0.0) > 0
        ref_to_prepare = None
        if self._needs_ref and peft_config is None:
            ref_to_prepare = copy.deepcopy(model)
            ref_to_prepare.eval()
            for p in ref_to_prepare.parameters():
                p.requires_grad_(False)

        if data_collator is None:
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            data_collator = PromptCollator(pad_token_id=pad_id)

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            config=config,
            loss_fn=loss_fn,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            peft_config=peft_config,
            callbacks=callbacks,
        )

        self.ref_model = None
        if ref_to_prepare is not None:
            self.ref_model = self.accelerator.prepare_model(ref_to_prepare, evaluation_mode=True)

    # ---- Training step override ----

    def _train_batch(self, batch):
        rollout = self._rollout(batch)
        del batch

        loss = None
        out = None
        for _ in range(max(1, self.config.num_iterations)):
            with self.accelerator.accumulate(self.model):
                logprobs = self._policy_logps(self.model, rollout, with_grad=True)
                old_logprobs = rollout["old_logprobs"]
                if old_logprobs is None:
                    # num_iterations == 1: old policy == current policy.
                    old_logprobs = logprobs.detach()

                out = self.loss_fn(
                    logprobs=logprobs,
                    old_logprobs=old_logprobs,
                    ref_logprobs=rollout["ref_logprobs"],
                    advantages=rollout["advantages"],
                    completion_mask=rollout["completion_mask"],
                )
                loss = out.loss

                self.accelerator.backward(loss)
                if self.config.max_grad_norm and self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                if not self.accelerator.optimizer_step_was_skipped:
                    self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

        metrics = {
            "reward/mean": rollout["reward_mean"],
            "reward/std": rollout["reward_std"],
            "kl": out.kl,
            "clip_frac": out.clip_frac,
            "ratio_mean": out.ratio_mean,
            "completion_length/mean": rollout["completion_length_mean"],
        }
        return loss, metrics

    def evaluate(self):
        """GRPO has no static eval loss (rewards are online). No-op."""
        self._log_info("GRPO has no static eval loss — skipping evaluate().")
        return {}

    # ---- Rollout ----

    def _rollout(self, batch):
        config = self.config
        G = config.num_generations
        device = self.accelerator.device

        prompt_ids = batch["input_ids"].to(device)
        prompt_mask = batch["attention_mask"].to(device)
        columns = batch.get("columns", {})
        B = prompt_ids.size(0)
        prompt_len = prompt_ids.size(1)

        # Expand each prompt into G generations (group-major order).
        rep_ids = prompt_ids.repeat_interleave(G, dim=0)
        rep_mask = prompt_mask.repeat_interleave(G, dim=0)

        full_ids = self._generate(rep_ids, rep_mask, G)  # (B*G, prompt_len + C)
        completion_ids = full_ids[:, prompt_len:]  # (B*G, C)
        completion_mask = self._completion_mask(completion_ids)  # (B*G, C)
        full_mask = torch.cat([rep_mask, completion_mask], dim=1)

        # Decode for the reward function.
        prompts = self._decode_prompts(prompt_ids, prompt_mask)  # length B
        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)  # B*G
        expanded_columns = {
            key: [vals[i] for i in range(B) for _ in range(G)]
            for key, vals in columns.items()
        }

        rewards = self._score(prompts, completions, expanded_columns, device)  # (B*G,)
        advantages = self._advantages(rewards, B, G)  # (B*G,)

        rollout = {
            "full_ids": full_ids,
            "full_mask": full_mask,
            "prompt_len": prompt_len,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item() if rewards.numel() > 1 else 0.0,
            "completion_length_mean": completion_mask.sum(dim=1).float().mean().item(),
        }

        # Old policy log-probs are only needed when we reuse the rollout for
        # multiple inner PPO iterations; otherwise old == current (ratio == 1).
        rollout["old_logprobs"] = (
            self._policy_logps(self.model, rollout, with_grad=False)
            if config.num_iterations > 1
            else None
        )
        rollout["ref_logprobs"] = self._ref_logps(rollout) if self._needs_ref else None

        self.sample_completions = [
            {"prompt": prompts[i // G], "completion": completions[i], "reward": rewards[i].item()}
            for i in range(min(len(completions), 4))
        ]
        return rollout

    def _generate(self, input_ids, attention_mask, num_generations):
        config = self.config
        model = self.accelerator.unwrap_model(self.model)
        gen_kwargs = dict(
            max_new_tokens=config.max_completion_length,
            do_sample=True,
            temperature=config.temperature,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if config.top_p is not None and config.top_p < 1.0:
            gen_kwargs["top_p"] = config.top_p
        if config.top_k is not None:
            gen_kwargs["top_k"] = config.top_k

        if config.generation_batch_size is not None:
            row_batch = max(1, config.generation_batch_size * num_generations)
        else:
            row_batch = input_ids.size(0)

        was_training = model.training
        chunks = []
        with _disable_grad_checkpointing(model), torch.no_grad():
            model.eval()
            for start in range(0, input_ids.size(0), row_batch):
                stop = start + row_batch
                chunks.append(
                    model.generate(
                        input_ids=input_ids[start:stop],
                        attention_mask=attention_mask[start:stop],
                        **gen_kwargs,
                    )
                )
            if was_training:
                model.train()
        if len(chunks) > 1:
            max_len = max(chunk.size(1) for chunk in chunks)
            chunks = [self._pad_generated(chunk, max_len) for chunk in chunks]
        return torch.cat(chunks, dim=0)

    def _pad_generated(self, generated, length):
        if generated.size(1) >= length:
            return generated
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
        padding = torch.full(
            (generated.size(0), length - generated.size(1)),
            pad_id,
            dtype=generated.dtype,
            device=generated.device,
        )
        return torch.cat([generated, padding], dim=1)

    def _completion_mask(self, completion_ids):
        """1 for real completion tokens up to and including the first EOS, else 0."""
        eos_id = self.tokenizer.eos_token_id
        C = completion_ids.size(1)
        if eos_id is None:
            return torch.ones_like(completion_ids)
        is_eos = completion_ids == eos_id
        has_eos = is_eos.any(dim=1)
        first_eos = torch.argmax(is_eos.int(), dim=1)  # 0 when no EOS
        seq_len = torch.where(has_eos, first_eos + 1, torch.full_like(first_eos, C))
        idx = torch.arange(C, device=completion_ids.device).unsqueeze(0)
        return (idx < seq_len.unsqueeze(1)).long()

    def _policy_logps(self, model, rollout, with_grad):
        """Per-token log-probs of the completion tokens under ``model`` -> (N, C)."""
        full_ids = rollout["full_ids"]
        full_mask = rollout["full_mask"]
        prompt_len = rollout["prompt_len"]

        grad_ctx = nullcontext() if with_grad else torch.no_grad()
        ckpt_ctx = nullcontext() if with_grad else _disable_grad_checkpointing(model)
        with ckpt_ctx, grad_ctx:
            logits = model(input_ids=full_ids, attention_mask=full_mask, use_cache=False).logits

        # Logits at positions [prompt_len-1 : -1] predict the completion tokens.
        comp_logits = logits[:, prompt_len - 1:-1, :]  # (N, C, V)
        comp_labels = full_ids[:, prompt_len:]  # (N, C)
        vocab_size = comp_logits.size(-1)
        safe_labels = comp_labels.clamp(min=0, max=vocab_size - 1)
        return _per_token_logps(comp_logits, safe_labels)  # (N, C)

    def _ref_logps(self, rollout):
        if self.ref_model is not None:
            return self._policy_logps(self.ref_model, rollout, with_grad=False)

        model = self.accelerator.unwrap_model(self.model)
        if hasattr(model, "disable_adapter"):
            with model.disable_adapter():
                return self._policy_logps(model, rollout, with_grad=False)

        raise ValueError(
            "GRPOLoss has beta > 0 but no reference policy is available. "
            "Use a PEFT/LoRA model (reference = disabled adapter), or run without "
            "PEFT so a frozen reference copy can be made, or set beta=0."
        )

    def _decode_prompts(self, prompt_ids, prompt_mask):
        """Decode the real (non-pad) tokens of each prompt -> list[str] of length B."""
        texts = []
        for ids, mask in zip(prompt_ids, prompt_mask):
            real = ids[mask.bool()]
            texts.append(self.tokenizer.decode(real, skip_special_tokens=True))
        return texts

    def _score(self, prompts, completions, columns, device):
        raw = self.reward_fn(prompts, completions, **columns)
        if len(raw) != len(completions):
            raise ValueError(
                "reward_fn must return one reward per completion "
                f"(got {len(raw)} rewards for {len(completions)} completions)"
            )
        rewards = torch.tensor([float(r) for r in raw], dtype=torch.float32, device=device)
        bad = ~torch.isfinite(rewards)
        if bad.any():
            logger.warning(
                "reward_fn returned %d non-finite value(s); treating them as 0.0",
                int(bad.sum().item()),
            )
            rewards = torch.where(bad, torch.zeros_like(rewards), rewards)
        return rewards

    def _advantages(self, rewards, B, G):
        grouped = rewards.view(B, G)
        advantages = grouped - grouped.mean(dim=1, keepdim=True)
        if getattr(self.loss_fn, "scale_rewards", True):
            advantages = advantages / (grouped.std(dim=1, keepdim=True) + 1e-4)
        return advantages.reshape(B * G).to(self.accelerator.device)
