"""End-to-end smoke tests for GRPOTrainer (HuggingFace .generate() backend)."""

import shutil
import tempfile
import types

import pytest
import torch
import torch.nn as nn
from datasets import Dataset

from grimoire import GRPOConfig, GRPOTrainer, TrainerCallback
from grimoire.losses.grpo import GRPOLoss


class TinyGenerativeLM(nn.Module):
    """Minimal causal LM with sampling-based generate() for GRPO testing."""

    def __init__(self, vocab_size=64, hidden_size=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size
        self.config = types.SimpleNamespace(is_encoder_decoder=False, use_cache=False)

    def forward(self, input_ids, attention_mask=None, labels=None, use_cache=False):
        logits = self.head(self.embed(input_ids))
        return types.SimpleNamespace(logits=logits, loss=None)

    def generate(self, input_ids, attention_mask=None, max_new_tokens=8,
                 do_sample=True, temperature=1.0, pad_token_id=0, **kwargs):
        generated = input_ids
        for _ in range(max_new_tokens):
            logits = self.forward(generated).logits[:, -1, :]
            if temperature != 1.0:
                logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
        return generated

    def get_input_embeddings(self):
        return self.embed


class GenTokenizer:
    """Minimal tokenizer stub for GRPO testing."""

    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "</s>"
    eos_token_id = 1
    chat_template = None

    def batch_decode(self, token_ids, skip_special_tokens=True):
        return [" ".join(str(t) for t in ids.tolist()) for ids in token_ids]

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(t) for t in token_ids.tolist())

    def save_pretrained(self, path):
        pass


def _length_reward(prompts, completions, **columns):
    """Score by completion length — produces within-group variance."""
    return [float(len(c)) for c in completions]


def make_prompt_dataset(n=8, prompt_len=4, vocab_size=64, with_column=False):
    data = {
        "input_ids": [torch.randint(2, vocab_size, (prompt_len,)).tolist() for _ in range(n)],
        "attention_mask": [[1] * prompt_len for _ in range(n)],
    }
    if with_column:
        data["solution"] = [f"sol-{i}" for i in range(n)]
    return Dataset.from_dict(data)


def _config(tmpdir, **overrides):
    base = dict(
        output_dir=tmpdir,
        num_epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        mixed_precision="no",
        gradient_checkpointing=False,
        logging_steps=1,
        save_on_epoch_end=False,
        num_generations=2,
        max_completion_length=4,
    )
    base.update(overrides)
    return GRPOConfig(**base)


class TestGRPOTrainerSmoke:
    def test_trains_one_epoch_beta_zero(self):
        torch.manual_seed(0)
        tmpdir = tempfile.mkdtemp()
        try:
            seen = []

            class Capture(TrainerCallback):
                def on_step_end(self, trainer, step, loss, metrics):
                    seen.append(metrics)

            trainer = GRPOTrainer(
                model=TinyGenerativeLM(),
                tokenizer=GenTokenizer(),
                config=_config(tmpdir),
                loss_fn=GRPOLoss(beta=0.0),
                reward_fn=_length_reward,
                train_dataset=make_prompt_dataset(n=8),
                callbacks=[Capture()],
            )
            trainer.train()

            assert len(seen) > 0
            m = seen[0]
            for key in ("reward/mean", "reward/std", "kl", "clip_frac",
                        "ratio_mean", "completion_length/mean"):
                assert key in m
            assert m["kl"] == 0.0  # beta == 0 disables KL
            # ratio ~ 1 on a single inner iteration (old == current policy)
            assert abs(m["ratio_mean"] - 1.0) < 0.1
            assert len(trainer.sample_completions) > 0
            assert {"prompt", "completion", "reward"} <= set(trainer.sample_completions[0])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_trains_with_reference_via_deepcopy(self):
        """beta > 0 without PEFT makes a frozen reference copy of the model."""
        torch.manual_seed(0)
        tmpdir = tempfile.mkdtemp()
        try:
            trainer = GRPOTrainer(
                model=TinyGenerativeLM(),
                tokenizer=GenTokenizer(),
                config=_config(tmpdir),
                loss_fn=GRPOLoss(beta=0.1),
                reward_fn=_length_reward,
                train_dataset=make_prompt_dataset(n=8),
            )
            assert trainer.ref_model is not None
            trainer.train()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_num_iterations_runs(self):
        torch.manual_seed(0)
        tmpdir = tempfile.mkdtemp()
        try:
            trainer = GRPOTrainer(
                model=TinyGenerativeLM(),
                tokenizer=GenTokenizer(),
                config=_config(tmpdir, num_iterations=2),
                loss_fn=GRPOLoss(beta=0.0),
                reward_fn=_length_reward,
                train_dataset=make_prompt_dataset(n=8),
            )
            trainer.train()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_passthrough_columns_reach_reward_fn(self):
        torch.manual_seed(0)
        tmpdir = tempfile.mkdtemp()
        captured = {}

        def reward_with_columns(prompts, completions, **columns):
            captured["columns"] = columns
            captured["n_prompts"] = len(prompts)
            captured["n_completions"] = len(completions)
            return [1.0] * len(completions)

        try:
            trainer = GRPOTrainer(
                model=TinyGenerativeLM(),
                tokenizer=GenTokenizer(),
                config=_config(tmpdir, num_generations=2),
                loss_fn=GRPOLoss(beta=0.0),
                reward_fn=reward_with_columns,
                train_dataset=make_prompt_dataset(n=8, with_column=True),
            )
            trainer.train()

            # B=2 prompts, G=2 generations -> completions/columns length 4, prompts length 2.
            assert captured["n_prompts"] == 2
            assert captured["n_completions"] == 4
            assert "solution" in captured["columns"]
            assert len(captured["columns"]["solution"]) == 4
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestGRPOTrainerUnits:
    def _trainer(self, tmpdir, loss_fn=None, **cfg):
        return GRPOTrainer(
            model=TinyGenerativeLM(),
            tokenizer=GenTokenizer(),
            config=_config(tmpdir, **cfg),
            loss_fn=loss_fn or GRPOLoss(beta=0.0),
            reward_fn=_length_reward,
            train_dataset=make_prompt_dataset(n=8),
        )

    def test_advantages_scaled_vs_centered(self):
        tmpdir = tempfile.mkdtemp()
        try:
            rewards = torch.tensor([0.0, 2.0])  # one group of G=2
            scaled = self._trainer(tmpdir, loss_fn=GRPOLoss(scale_rewards=True))
            centered = self._trainer(tmpdir, loss_fn=GRPOLoss(scale_rewards=False))

            a_scaled = scaled._advantages(rewards, B=1, G=2)
            a_centered = centered._advantages(rewards, B=1, G=2)

            # mean-centered: [-1, 1]
            assert torch.allclose(a_centered, torch.tensor([-1.0, 1.0]))
            # scaled divides by (std + eps), so magnitudes differ from centered
            assert not torch.allclose(a_scaled, a_centered)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_constant_rewards_give_zero_advantages(self):
        tmpdir = tempfile.mkdtemp()
        try:
            t = self._trainer(tmpdir)
            adv = t._advantages(torch.tensor([1.0, 1.0]), B=1, G=2)
            assert torch.allclose(adv, torch.zeros(2), atol=1e-6)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_completion_mask_stops_after_eos(self):
        tmpdir = tempfile.mkdtemp()
        try:
            t = self._trainer(tmpdir)
            # eos_token_id == 1
            ids = torch.tensor([[5, 1, 7, 8], [5, 6, 7, 8]])
            mask = t._completion_mask(ids)
            assert mask[0].tolist() == [1, 1, 0, 0]  # up to and including first EOS
            assert mask[1].tolist() == [1, 1, 1, 1]  # no EOS -> all kept
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_non_finite_rewards_sanitized(self):
        tmpdir = tempfile.mkdtemp()
        try:
            t = GRPOTrainer(
                model=TinyGenerativeLM(),
                tokenizer=GenTokenizer(),
                config=_config(tmpdir),
                loss_fn=GRPOLoss(beta=0.0),
                reward_fn=lambda prompts, completions, **cols: [float("nan"), float("inf"), 1.0],
                train_dataset=make_prompt_dataset(n=8),
            )
            rewards = t._score(
                prompts=["p"],
                completions=["a", "b", "c"],
                columns={},
                device=torch.device("cpu"),
            )
            # NaN/inf are replaced with 0.0; the finite value is preserved.
            assert torch.isfinite(rewards).all()
            assert rewards.tolist() == [0.0, 0.0, 1.0]
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_use_vllm_raises_not_implemented(self):
        tmpdir = tempfile.mkdtemp()
        try:
            with pytest.raises(NotImplementedError):
                GRPOTrainer(
                    model=TinyGenerativeLM(),
                    tokenizer=GenTokenizer(),
                    config=_config(tmpdir, use_vllm=True),
                    loss_fn=GRPOLoss(beta=0.0),
                    reward_fn=_length_reward,
                    train_dataset=make_prompt_dataset(n=8),
                )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
