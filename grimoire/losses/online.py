"""Shared machinery for online RL methods (GRPO, RLOO, Online DPO, RAFT).

An online method is a two-phase loss object:

1. ``rollout(model, batch)`` — the ONLINE phase. Turns a prompt-only batch
   into a scored experience batch: generate completions, decode, call the
   reward function, and build the tensors the loss phase consumes. Runs
   entirely under ``torch.no_grad()``. The trainer calls it before the loss
   whenever a method exposes it; offline losses simply don't.
2. ``__call__(model, experience)`` — the loss phase. A pure function of the
   experience batch, structurally identical to the offline losses.

``OnlineMethod`` owns the phase-1 building blocks every method shares:
generation from LEFT-padded prompts, post-EOS masking, reward scoring, and
frozen-reference log-probs. Subclasses implement ``rollout`` (how the scored
generations become an experience batch) and ``__call__`` (the loss on it).

All online methods call ``model.generate()``, which needs full weight access —
the trainer refuses DeepSpeed ZeRO-3 for any loss that defines ``rollout``;
ZeRO-2 or lower, or FSDP, is fine.
"""

import logging

import torch

from ..data.grpo import GRPOCollator
from .utils import (
    DEFAULT_FUSED_CHUNK_SIZE,
    _disable_grad_checkpointing,
    _unwrap_model,
    forward_per_token_logps,
    masked_avg_logps,
)

logger = logging.getLogger(__name__)


class OnlineMethod:
    """Base class for online methods: rollout building blocks, no loss.

    Subclasses define ``rollout(model, batch)`` (usually on top of
    ``_generate_and_score``) and ``__call__(model, batch, training)``.
    """

    def __init__(
        self,
        reward_fn,
        tokenizer,
        num_generations=4,
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
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.label_pad_token_id = label_pad_token_id
        self.ref_model = ref_model
        self.fused = fused
        self.fused_chunk_size = fused_chunk_size
        self._pad_token_id = 0
        self._warned_no_ref = False

    def create_collator(self, pad_token_id):
        self._pad_token_id = pad_token_id
        return GRPOCollator(pad_token_id=pad_token_id)

    def _avg_logps(self, model, input_ids, attention_mask, labels):
        per_token_logps, loss_mask = forward_per_token_logps(
            model, input_ids, attention_mask, labels, self.label_pad_token_id,
            fused=self.fused, fused_chunk_size=self.fused_chunk_size,
        )
        return masked_avg_logps(per_token_logps, loss_mask)

    # --- shared rollout building blocks ------------------------------------ #

    @torch.no_grad()
    def _generate_and_score(self, model, batch):
        """Generate G completions per prompt and score them with ``reward_fn``.

        The common front half of every online rollout. Returns a dict with
        ``input_ids`` / ``attention_mask`` / ``labels`` for the [B*G]
        prompt+completion sequences (prompt region and post-EOS padding masked
        in the labels), float32 ``rewards`` [B*G], and ``completion_length``.
        Rows are grouped by prompt: rows i*G..(i+1)*G-1 belong to prompt i.
        """
        input_ids = batch["input_ids"]  # [B, prompt_len], left-padded
        attention_mask = batch["attention_mask"]  # [B, prompt_len]
        G = self.num_generations
        prompt_len = input_ids.size(1)

        # 1. Generate G completions per prompt. generate() needs the unwrapped
        # model (DDP/FSDP wrappers don't forward .generate); the loss phase gets
        # the wrapped model back from the trainer for correct gradient sync.
        repeated_ids = input_ids.repeat_interleave(G, dim=0)  # [B*G, prompt_len]
        repeated_mask = attention_mask.repeat_interleave(G, dim=0)
        gen_model = _unwrap_model(model)

        with _disable_grad_checkpointing(model):
            was_training = gen_model.training
            gen_model.eval()
            generated = gen_model.generate(
                input_ids=repeated_ids,
                attention_mask=repeated_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self._pad_token_id,
                # The trainer sets model.config.use_cache = False for training;
                # be explicit or generation runs cache-less (quadratic decode).
                use_cache=True,
            )  # [B*G, prompt_len + completion_len]
            if was_training:
                gen_model.train()

        completion_ids = generated[:, prompt_len:]
        completion_length = completion_ids.size(1)

        # Mask completion tokens after the first EOS (generate() pads there, and
        # pad_token_id may collide with real token IDs, e.g. pad == eos).
        completion_mask = self._completion_mask(completion_ids)
        gen_attention_mask = torch.cat([repeated_mask, completion_mask], dim=1)

        # Labels: mask the prompt region and padding after EOS.
        gen_labels = generated.clone()
        gen_labels[:, :prompt_len] = self.label_pad_token_id
        gen_labels[:, prompt_len:] = torch.where(
            completion_mask.bool(), completion_ids, self.label_pad_token_id
        )

        # 2. Score completions with the reward function (the environment).
        prompt_texts = self.tokenizer.batch_decode(repeated_ids, skip_special_tokens=True)
        completion_texts = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        del repeated_ids, completion_ids
        rewards = self.reward_fn(prompt_texts, completion_texts)  # list[float] or tensor
        del prompt_texts, completion_texts
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, dtype=torch.float32, device=input_ids.device)
        elif rewards.device != input_ids.device:
            rewards = rewards.to(input_ids.device)  # [B*G]

        return {
            "input_ids": generated,
            "attention_mask": gen_attention_mask,
            "labels": gen_labels,
            "rewards": rewards,
            "completion_length": completion_length,
        }

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

    def _reference_logps(self, model, input_ids, attention_mask, labels, required=False):
        """Average log-probs under the frozen reference policy.

        The reference is ``ref_model`` if provided, else the base model via
        ``disable_adapter()`` for PEFT models. When neither is available,
        raises if ``required`` (methods whose objective needs a reference,
        e.g. Online DPO) or warns once and returns None otherwise (KL-penalty
        methods, where the term is skippable).
        """
        if self.ref_model is not None:
            return self._avg_logps(self.ref_model, input_ids, attention_mask, labels)
        if hasattr(model, "disable_adapter"):
            with _disable_grad_checkpointing(model), model.disable_adapter():
                return self._avg_logps(model, input_ids, attention_mask, labels)
        if required:
            raise ValueError(
                f"{type(self).__name__} requires a reference policy: pass "
                "ref_model or use a PEFT model with disable_adapter()."
            )
        if not self._warned_no_ref:
            logger.warning(
                "%s: beta > 0 but no reference policy is available "
                "(no ref_model and the model has no disable_adapter()). "
                "Skipping the KL penalty — pass ref_model or set beta=0.",
                type(self).__name__,
            )
            self._warned_no_ref = True
        return None
