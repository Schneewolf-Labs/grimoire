"""Reward function contract for GRPO — the membrane between Grimoire and Merlina.

Grimoire defines ONLY the Protocol. Concrete rewards (regex/format/math-correctness
judges, LLM/VLM-as-judge clients, sandboxed user-supplied code, weighted
compositions of several rewards) live in the caller (Merlina) and are passed
across the boundary as a single synchronous callable. Grimoire never knows what
is behind it — only that it scores completions.
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class RewardFn(Protocol):
    """A synchronous callable that scores completions.

    Called once per rollout group-batch by ``GRPOTrainer``. MUST be synchronous
    and MUST return one finite float per completion (same length/order as
    ``completions``). Any async / network / subprocess / sandboxing is the
    caller's problem and must be hidden behind this sync interface before it
    reaches the trainer.

    Args:
        prompts:     decoded prompt strings, length B (the prompts in this step).
        completions: decoded completion strings, length B * num_generations,
                     ordered group-major as
                     ``[p0g0, p0g1, ..., p0g{G-1}, p1g0, ...]``.
        **columns:   any extra dataset columns for the corresponding prompts,
                     each a list aligned to ``completions`` (broadcast across the
                     G group). e.g. a ``solution``/``answer`` column for
                     verifiable-reward tasks.

    Returns:
        list[float] of length ``len(completions)``. NaN/inf are treated as 0.0
        with a logged warning by the trainer (one bad reward shouldn't kill a run).
    """

    def __call__(
        self,
        prompts: Sequence[str],
        completions: Sequence[str],
        **columns: Sequence,
    ) -> list[float]: ...
