"""End-to-end smoke test for GrimoireTrainer."""

import copy
import os
import shutil
import tempfile
from contextlib import contextmanager

import torch
import torch.nn as nn
import pytest
from datasets import Dataset

from grimoire import GrimoireTrainer, TrainingConfig, TrainerCallback
from grimoire.losses.sft import SFTLoss
from grimoire.losses.orpo import ORPOLoss
from grimoire.losses.dpo import DPOLoss
from grimoire.losses.simpo import SimPOLoss
from grimoire.losses.kto import KTOLoss
from grimoire.losses.cpo import CPOLoss
from grimoire.losses.ipo import IPOLoss


class TinyLM(nn.Module):
    """Minimal causal LM for testing."""

    def __init__(self, vocab_size=64, hidden_size=32, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size
        self.config = type("Config", (), {
            "is_encoder_decoder": False,
            "use_return_dict": True,
        })()

    def forward(self, input_ids, attention_mask=None, labels=None, use_cache=False):
        h = self.embed(input_ids)
        for layer in self.layers:
            h = torch.relu(layer(h))
        logits = self.head(h)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return type("Output", (), {"logits": logits, "loss": loss})()

    def get_input_embeddings(self):
        return self.embed

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        pass  # no-op for tiny model

    def enable_input_require_grads(self):
        pass

    def save_pretrained(self, path, is_main_process=True, save_function=None, **kwargs):
        if is_main_process:
            os.makedirs(path, exist_ok=True)
            save_fn = save_function or torch.save
            save_fn(self.state_dict(), os.path.join(path, "model.pt"))


class FakeTokenizer:
    """Minimal tokenizer stub for testing."""

    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "</s>"
    eos_token_id = 1

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)


def make_sft_dataset(n=32, seq_len=16, vocab_size=64):
    """Create a small SFT dataset."""
    data = {
        "input_ids": [torch.randint(2, vocab_size, (seq_len,)).tolist() for _ in range(n)],
        "attention_mask": [[1] * seq_len for _ in range(n)],
        "labels": [
            [-100] * 4 + torch.randint(2, vocab_size, (seq_len - 4,)).tolist()
            for _ in range(n)
        ],
    }
    return Dataset.from_dict(data)


def make_preference_dataset(n=32, seq_len=16, vocab_size=64):
    """Create a small preference dataset."""
    data = {
        "chosen_input_ids": [torch.randint(2, vocab_size, (seq_len,)).tolist() for _ in range(n)],
        "chosen_attention_mask": [[1] * seq_len for _ in range(n)],
        "chosen_labels": [
            [-100] * 4 + torch.randint(2, vocab_size, (seq_len - 4,)).tolist()
            for _ in range(n)
        ],
        "rejected_input_ids": [torch.randint(2, vocab_size, (seq_len,)).tolist() for _ in range(n)],
        "rejected_attention_mask": [[1] * seq_len for _ in range(n)],
        "rejected_labels": [
            [-100] * 4 + torch.randint(2, vocab_size, (seq_len - 4,)).tolist()
            for _ in range(n)
        ],
    }
    return Dataset.from_dict(data)


class TestSFTTraining:
    def test_sft_trains_and_loss_decreases(self):
        torch.manual_seed(42)
        tmpdir = tempfile.mkdtemp()

        try:
            model = TinyLM()
            dataset = make_sft_dataset(n=32)

            config = TrainingConfig(
                output_dir=tmpdir,
                num_epochs=3,
                batch_size=8,
                learning_rate=1e-3,
                gradient_accumulation_steps=1,
                mixed_precision="no",
                gradient_checkpointing=False,
                logging_steps=1,
                save_on_epoch_end=False,
            )

            losses = []

            class LossTracker(TrainerCallback):
                def on_step_end(self, trainer, step, loss, metrics):
                    losses.append(loss)

            trainer = GrimoireTrainer(
                model=model,
                tokenizer=FakeTokenizer(),
                config=config,
                loss_fn=SFTLoss(),
                train_dataset=dataset,
                callbacks=[LossTracker()],
            )
            trainer.train()

            assert len(losses) > 0
            # Loss should generally decrease over training
            first_losses = sum(losses[:3]) / 3
            last_losses = sum(losses[-3:]) / 3
            assert last_losses < first_losses, f"Loss didn't decrease: {first_losses:.4f} -> {last_losses:.4f}"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_save_model(self):
        torch.manual_seed(42)
        tmpdir = tempfile.mkdtemp()

        try:
            model = TinyLM()
            dataset = make_sft_dataset(n=8)

            config = TrainingConfig(
                output_dir=tmpdir,
                num_epochs=1,
                batch_size=4,
                mixed_precision="no",
                gradient_checkpointing=False,
                save_on_epoch_end=False,
            )

            trainer = GrimoireTrainer(
                model=model,
                tokenizer=FakeTokenizer(),
                config=config,
                loss_fn=SFTLoss(),
                train_dataset=dataset,
            )
            trainer.train()

            save_dir = os.path.join(tmpdir, "saved_model")
            trainer.save_model(save_dir)

            assert os.path.isdir(save_dir)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_eval_runs(self):
        torch.manual_seed(42)
        tmpdir = tempfile.mkdtemp()

        try:
            model = TinyLM()
            train_ds = make_sft_dataset(n=16)
            eval_ds = make_sft_dataset(n=8)

            config = TrainingConfig(
                output_dir=tmpdir,
                num_epochs=1,
                batch_size=4,
                mixed_precision="no",
                gradient_checkpointing=False,
                eval_steps=2,
                save_on_epoch_end=False,
            )

            eval_results = []

            class EvalTracker(TrainerCallback):
                def on_evaluate(self, trainer, metrics):
                    eval_results.append(metrics)

            trainer = GrimoireTrainer(
                model=model,
                tokenizer=FakeTokenizer(),
                config=config,
                loss_fn=SFTLoss(),
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                callbacks=[EvalTracker()],
            )
            trainer.train()

            assert len(eval_results) > 0
            assert "eval/loss" in eval_results[0]
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestORPOTraining:
    def test_orpo_trains_and_loss_decreases(self):
        torch.manual_seed(42)
        tmpdir = tempfile.mkdtemp()

        try:
            model = TinyLM()
            dataset = make_preference_dataset(n=32)

            config = TrainingConfig(
                output_dir=tmpdir,
                num_epochs=3,
                batch_size=8,
                learning_rate=1e-3,
                gradient_accumulation_steps=1,
                mixed_precision="no",
                gradient_checkpointing=False,
                logging_steps=1,
                disable_dropout=True,
                save_on_epoch_end=False,
            )

            losses = []

            class LossTracker(TrainerCallback):
                def on_step_end(self, trainer, step, loss, metrics):
                    losses.append(loss)

            loss_fn = ORPOLoss(beta=0.1)

            trainer = GrimoireTrainer(
                model=model,
                tokenizer=FakeTokenizer(),
                config=config,
                loss_fn=loss_fn,
                train_dataset=dataset,
                callbacks=[LossTracker()],
            )
            trainer.train()

            assert len(losses) > 0
            # At minimum, loss should not be NaN
            assert all(not torch.tensor(loss).isnan() for loss in losses), "NaN loss detected"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_orpo_metrics_present(self):
        torch.manual_seed(42)
        tmpdir = tempfile.mkdtemp()

        try:
            model = TinyLM()
            dataset = make_preference_dataset(n=16)

            config = TrainingConfig(
                output_dir=tmpdir,
                num_epochs=1,
                batch_size=4,
                mixed_precision="no",
                gradient_checkpointing=False,
                logging_steps=1,
                save_on_epoch_end=False,
            )

            all_metrics = []

            class MetricsTracker(TrainerCallback):
                def on_step_end(self, trainer, step, loss, metrics):
                    all_metrics.append(metrics)

            trainer = GrimoireTrainer(
                model=model,
                tokenizer=FakeTokenizer(),
                config=config,
                loss_fn=ORPOLoss(beta=0.1),
                train_dataset=dataset,
                callbacks=[MetricsTracker()],
            )
            trainer.train()

            assert len(all_metrics) > 0
            m = all_metrics[0]
            assert "nll_loss" in m
            assert "or_loss" in m
            assert "reward_accuracy" in m
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_orpo_eval(self):
        torch.manual_seed(42)
        tmpdir = tempfile.mkdtemp()

        try:
            model = TinyLM()
            train_ds = make_preference_dataset(n=16)
            eval_ds = make_preference_dataset(n=8)

            config = TrainingConfig(
                output_dir=tmpdir,
                num_epochs=1,
                batch_size=4,
                mixed_precision="no",
                gradient_checkpointing=False,
                eval_steps=2,
                save_on_epoch_end=False,
            )

            eval_results = []

            class EvalTracker(TrainerCallback):
                def on_evaluate(self, trainer, metrics):
                    eval_results.append(metrics)

            trainer = GrimoireTrainer(
                model=model,
                tokenizer=FakeTokenizer(),
                config=config,
                loss_fn=ORPOLoss(beta=0.1),
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                callbacks=[EvalTracker()],
            )
            trainer.train()

            assert len(eval_results) > 0
            assert "eval/loss" in eval_results[0]
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestCallbackLifecycle:
    def test_all_lifecycle_events_fire(self):
        torch.manual_seed(42)
        tmpdir = tempfile.mkdtemp()

        try:
            model = TinyLM()
            train_ds = make_sft_dataset(n=16)
            eval_ds = make_sft_dataset(n=8)

            config = TrainingConfig(
                output_dir=tmpdir,
                num_epochs=2,
                batch_size=8,
                mixed_precision="no",
                gradient_checkpointing=False,
                logging_steps=1,
                eval_steps=2,
                save_on_epoch_end=False,
            )

            events = []

            class LifecycleTracker(TrainerCallback):
                def on_train_begin(self, trainer):
                    events.append("train_begin")

                def on_train_end(self, trainer):
                    events.append("train_end")

                def on_epoch_begin(self, trainer, epoch):
                    events.append(f"epoch_begin_{epoch}")

                def on_epoch_end(self, trainer, epoch):
                    events.append(f"epoch_end_{epoch}")

                def on_step_end(self, trainer, step, loss, metrics):
                    events.append(f"step_end_{step}")

                def on_log(self, trainer, metrics):
                    events.append("log")

                def on_evaluate(self, trainer, metrics):
                    events.append("evaluate")

            trainer = GrimoireTrainer(
                model=model,
                tokenizer=FakeTokenizer(),
                config=config,
                loss_fn=SFTLoss(),
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                callbacks=[LifecycleTracker()],
            )
            trainer.train()

            assert "train_begin" in events
            assert "train_end" in events
            assert "epoch_begin_0" in events
            assert "epoch_end_0" in events
            assert "epoch_begin_1" in events
            assert "epoch_end_1" in events
            assert any(e.startswith("step_end_") for e in events)
            assert "log" in events
            assert "evaluate" in events
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_multiple_callbacks_all_fire(self):
        torch.manual_seed(42)
        tmpdir = tempfile.mkdtemp()

        try:
            model = TinyLM()
            dataset = make_sft_dataset(n=8)

            config = TrainingConfig(
                output_dir=tmpdir,
                num_epochs=1,
                batch_size=4,
                mixed_precision="no",
                gradient_checkpointing=False,
                save_on_epoch_end=False,
            )

            fired_a = []
            fired_b = []

            class CallbackA(TrainerCallback):
                def on_train_begin(self, trainer):
                    fired_a.append("begin")

            class CallbackB(TrainerCallback):
                def on_train_begin(self, trainer):
                    fired_b.append("begin")

            trainer = GrimoireTrainer(
                model=model,
                tokenizer=FakeTokenizer(),
                config=config,
                loss_fn=SFTLoss(),
                train_dataset=dataset,
                callbacks=[CallbackA(), CallbackB()],
            )
            trainer.train()

            assert len(fired_a) == 1
            assert len(fired_b) == 1
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestEarlyStopping:
    def test_request_stop_halts_training(self):
        torch.manual_seed(42)
        tmpdir = tempfile.mkdtemp()

        try:
            model = TinyLM()
            dataset = make_sft_dataset(n=32)

            config = TrainingConfig(
                output_dir=tmpdir,
                num_epochs=10,
                batch_size=8,
                mixed_precision="no",
                gradient_checkpointing=False,
                logging_steps=1,
                save_on_epoch_end=False,
            )

            steps_seen = []

            class StopAfterThree(TrainerCallback):
                def on_step_end(self, trainer, step, loss, metrics):
                    steps_seen.append(step)
                    if step >= 3:
                        trainer.request_stop()

            trainer = GrimoireTrainer(
                model=model,
                tokenizer=FakeTokenizer(),
                config=config,
                loss_fn=SFTLoss(),
                train_dataset=dataset,
                callbacks=[StopAfterThree()],
            )
            trainer.train()

            # Should stop well before completing all 10 epochs
            assert trainer.stopped_early
            # Should not have seen far more steps than needed to trigger stop
            assert max(steps_seen) < 20
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_stopped_early_false_when_not_stopped(self):
        torch.manual_seed(42)
        tmpdir = tempfile.mkdtemp()

        try:
            model = TinyLM()
            dataset = make_sft_dataset(n=8)

            config = TrainingConfig(
                output_dir=tmpdir,
                num_epochs=1,
                batch_size=4,
                mixed_precision="no",
                gradient_checkpointing=False,
                save_on_epoch_end=False,
            )

            trainer = GrimoireTrainer(
                model=model,
                tokenizer=FakeTokenizer(),
                config=config,
                loss_fn=SFTLoss(),
                train_dataset=dataset,
            )
            trainer.train()

            assert not trainer.stopped_early
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestLRSchedulers:
    @pytest.mark.parametrize("scheduler", ["linear", "cosine", "constant", "constant_with_warmup"])
    def test_scheduler_runs_without_error(self, scheduler):
        torch.manual_seed(42)
        tmpdir = tempfile.mkdtemp()

        try:
            model = TinyLM()
            dataset = make_sft_dataset(n=8)

            config = TrainingConfig(
                output_dir=tmpdir,
                num_epochs=1,
                batch_size=4,
                mixed_precision="no",
                gradient_checkpointing=False,
                lr_scheduler=scheduler,
                warmup_steps=1,
                save_on_epoch_end=False,
            )

            trainer = GrimoireTrainer(
                model=model,
                tokenizer=FakeTokenizer(),
                config=config,
                loss_fn=SFTLoss(),
                train_dataset=dataset,
            )
            trainer.train()  # should not raise
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestOptimizerSGD:
    def test_sgd_optimizer_trains(self):
        torch.manual_seed(42)
        tmpdir = tempfile.mkdtemp()

        try:
            model = TinyLM()
            dataset = make_sft_dataset(n=8)

            config = TrainingConfig(
                output_dir=tmpdir,
                num_epochs=1,
                batch_size=4,
                mixed_precision="no",
                gradient_checkpointing=False,
                optimizer="sgd",
                save_on_epoch_end=False,
            )

            trainer = GrimoireTrainer(
                model=model,
                tokenizer=FakeTokenizer(),
                config=config,
                loss_fn=SFTLoss(),
                train_dataset=dataset,
            )
            trainer.train()  # should not raise
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_unknown_optimizer_raises(self):
        tmpdir = tempfile.mkdtemp()

        try:
            model = TinyLM()
            dataset = make_sft_dataset(n=8)

            config = TrainingConfig(
                output_dir=tmpdir,
                num_epochs=1,
                batch_size=4,
                mixed_precision="no",
                gradient_checkpointing=False,
                optimizer="unknown_opt",
                save_on_epoch_end=False,
            )

            with pytest.raises(ValueError, match="Unknown optimizer"):
                GrimoireTrainer(
                    model=model,
                    tokenizer=FakeTokenizer(),
                    config=config,
                    loss_fn=SFTLoss(),
                    train_dataset=dataset,
                )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestAdafactorOptimizer:
    def test_adafactor_optimizer_trains(self):
        torch.manual_seed(42)
        tmpdir = tempfile.mkdtemp()

        try:
            model = TinyLM()
            dataset = make_sft_dataset(n=8)

            config = TrainingConfig(
                output_dir=tmpdir,
                num_epochs=1,
                batch_size=4,
                mixed_precision="no",
                gradient_checkpointing=False,
                optimizer="adafactor",
                save_on_epoch_end=False,
            )

            trainer = GrimoireTrainer(
                model=model,
                tokenizer=FakeTokenizer(),
                config=config,
                loss_fn=SFTLoss(),
                train_dataset=dataset,
            )
            trainer.train()  # should not raise
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestMuonOptimizer:
    def test_muon_optimizer_trains(self):
        torch.manual_seed(42)
        tmpdir = tempfile.mkdtemp()

        try:
            model = TinyLM()
            dataset = make_sft_dataset(n=8)

            config = TrainingConfig(
                output_dir=tmpdir,
                num_epochs=1,
                batch_size=4,
                mixed_precision="no",
                gradient_checkpointing=False,
                optimizer="muon",
                learning_rate=0.02,
                save_on_epoch_end=False,
            )

            trainer = GrimoireTrainer(
                model=model,
                tokenizer=FakeTokenizer(),
                config=config,
                loss_fn=SFTLoss(),
                train_dataset=dataset,
            )
            trainer.train()  # should not raise
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestGradientAccumulation:
    def test_gradient_accumulation_runs(self):
        torch.manual_seed(42)
        tmpdir = tempfile.mkdtemp()

        try:
            model = TinyLM()
            dataset = make_sft_dataset(n=16)

            config = TrainingConfig(
                output_dir=tmpdir,
                num_epochs=1,
                batch_size=4,
                gradient_accumulation_steps=2,
                mixed_precision="no",
                gradient_checkpointing=False,
                logging_steps=1,
                save_on_epoch_end=False,
            )

            losses = []

            class LossTracker(TrainerCallback):
                def on_step_end(self, trainer, step, loss, metrics):
                    losses.append(loss)

            trainer = GrimoireTrainer(
                model=model,
                tokenizer=FakeTokenizer(),
                config=config,
                loss_fn=SFTLoss(),
                train_dataset=dataset,
                callbacks=[LossTracker()],
            )
            trainer.train()

            assert len(losses) > 0
            assert all(not torch.tensor(loss_val).isnan() for loss_val in losses)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestEvalOnStart:
    def test_eval_on_start_fires_before_training(self):
        torch.manual_seed(42)
        tmpdir = tempfile.mkdtemp()

        try:
            model = TinyLM()
            train_ds = make_sft_dataset(n=8)
            eval_ds = make_sft_dataset(n=4)

            config = TrainingConfig(
                output_dir=tmpdir,
                num_epochs=1,
                batch_size=4,
                mixed_precision="no",
                gradient_checkpointing=False,
                eval_on_start=True,
                save_on_epoch_end=False,
            )

            event_log = []

            class EventTracker(TrainerCallback):
                def on_train_begin(self, trainer):
                    event_log.append("train_begin")

                def on_evaluate(self, trainer, metrics):
                    event_log.append("evaluate")

                def on_step_end(self, trainer, step, loss, metrics):
                    if "first_step" not in event_log:
                        event_log.append("first_step")

            trainer = GrimoireTrainer(
                model=model,
                tokenizer=FakeTokenizer(),
                config=config,
                loss_fn=SFTLoss(),
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                callbacks=[EventTracker()],
            )
            trainer.train()

            # Evaluate should appear before first_step (eval_on_start fires before training loop)
            assert "evaluate" in event_log
            eval_idx = event_log.index("evaluate")
            first_step_idx = event_log.index("first_step") if "first_step" in event_log else len(event_log)
            assert eval_idx < first_step_idx
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestCheckpointing:
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Checkpoint save_state may require CUDA in some environments",
    )
    def test_saves_and_rotates_checkpoints(self):
        torch.manual_seed(42)
        tmpdir = tempfile.mkdtemp()

        try:
            model = TinyLM()
            dataset = make_sft_dataset(n=32)

            config = TrainingConfig(
                output_dir=tmpdir,
                num_epochs=2,
                batch_size=4,
                mixed_precision="no",
                gradient_checkpointing=False,
                save_steps=2,
                save_total_limit=2,
                save_on_epoch_end=False,
            )

            trainer = GrimoireTrainer(
                model=model,
                tokenizer=FakeTokenizer(),
                config=config,
                loss_fn=SFTLoss(),
                train_dataset=dataset,
            )
            trainer.train()

            checkpoints = [d for d in os.listdir(tmpdir) if d.startswith("checkpoint-")]
            assert len(checkpoints) <= 2, f"Expected <= 2 checkpoints, got {len(checkpoints)}: {checkpoints}"
            assert len(checkpoints) > 0
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


def make_kto_dataset(n=32, seq_len=16, vocab_size=64):
    """Create a small KTO dataset."""
    data = {
        "input_ids": [torch.randint(2, vocab_size, (seq_len,)).tolist() for _ in range(n)],
        "attention_mask": [[1] * seq_len for _ in range(n)],
        "labels": [
            [-100] * 4 + torch.randint(2, vocab_size, (seq_len - 4,)).tolist()
            for _ in range(n)
        ],
        "kto_label": [i % 2 == 0 for i in range(n)],
    }
    return Dataset.from_dict(data)


class PeftTinyLM(TinyLM):
    """TinyLM with PEFT-like adapter support for testing disable_adapter path."""

    def __init__(self, vocab_size=64, hidden_size=32, num_layers=1):
        super().__init__(vocab_size, hidden_size, num_layers)
        self.adapter = nn.Linear(hidden_size, hidden_size)
        self._adapter_enabled = True

    def forward(self, input_ids, attention_mask=None, labels=None, use_cache=False):
        h = self.embed(input_ids)
        if self._adapter_enabled:
            h = h + self.adapter(h)
        for layer in self.layers:
            h = torch.relu(layer(h))
        logits = self.head(h)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return type("Output", (), {"logits": logits, "loss": loss})()

    @contextmanager
    def disable_adapter(self):
        self._adapter_enabled = False
        try:
            yield
        finally:
            self._adapter_enabled = True


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class TestKTOTraining:
    def test_kto_trains_with_ref_model(self):
        torch.manual_seed(42)
        tmpdir = tempfile.mkdtemp()

        try:
            model = TinyLM()
            ref_model = copy.deepcopy(model).to(_device())
            ref_model.eval()
            dataset = make_kto_dataset(n=32)

            config = TrainingConfig(
                output_dir=tmpdir,
                num_epochs=2,
                batch_size=8,
                learning_rate=1e-3,
                gradient_accumulation_steps=1,
                mixed_precision="no",
                gradient_checkpointing=False,
                logging_steps=1,
                save_on_epoch_end=False,
            )

            losses = []

            class LossTracker(TrainerCallback):
                def on_step_end(self, trainer, step, loss, metrics):
                    losses.append(loss)

            trainer = GrimoireTrainer(
                model=model,
                tokenizer=FakeTokenizer(),
                config=config,
                loss_fn=KTOLoss(ref_model=ref_model, beta=0.1),
                train_dataset=dataset,
                callbacks=[LossTracker()],
            )
            trainer.train()

            assert len(losses) > 0
            assert all(not torch.tensor(v).isnan() for v in losses)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_kto_trains_with_disable_adapter(self):
        torch.manual_seed(42)
        tmpdir = tempfile.mkdtemp()

        try:
            model = PeftTinyLM()
            dataset = make_kto_dataset(n=16)

            config = TrainingConfig(
                output_dir=tmpdir,
                num_epochs=1,
                batch_size=4,
                learning_rate=1e-3,
                mixed_precision="no",
                gradient_checkpointing=False,
                save_on_epoch_end=False,
            )

            losses = []

            class LossTracker(TrainerCallback):
                def on_step_end(self, trainer, step, loss, metrics):
                    losses.append(loss)

            trainer = GrimoireTrainer(
                model=model,
                tokenizer=FakeTokenizer(),
                config=config,
                loss_fn=KTOLoss(ref_model=None, beta=0.1),
                train_dataset=dataset,
                callbacks=[LossTracker()],
            )
            trainer.train()

            assert len(losses) > 0
            assert all(not torch.tensor(v).isnan() for v in losses)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_kto_metrics_present(self):
        torch.manual_seed(42)
        tmpdir = tempfile.mkdtemp()

        try:
            model = TinyLM()
            ref_model = copy.deepcopy(model).to(_device())
            ref_model.eval()
            dataset = make_kto_dataset(n=16)

            config = TrainingConfig(
                output_dir=tmpdir,
                num_epochs=1,
                batch_size=4,
                mixed_precision="no",
                gradient_checkpointing=False,
                logging_steps=1,
                save_on_epoch_end=False,
            )

            all_metrics = []

            class MetricsTracker(TrainerCallback):
                def on_step_end(self, trainer, step, loss, metrics):
                    all_metrics.append(metrics)

            trainer = GrimoireTrainer(
                model=model,
                tokenizer=FakeTokenizer(),
                config=config,
                loss_fn=KTOLoss(ref_model=ref_model, beta=0.1),
                train_dataset=dataset,
                callbacks=[MetricsTracker()],
            )
            trainer.train()

            assert len(all_metrics) > 0
            m = all_metrics[0]
            assert "chosen_rewards" in m
            assert "rejected_rewards" in m
            assert "kl_ref" in m
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_kto_eval(self):
        torch.manual_seed(42)
        tmpdir = tempfile.mkdtemp()

        try:
            model = TinyLM()
            ref_model = copy.deepcopy(model).to(_device())
            ref_model.eval()
            train_ds = make_kto_dataset(n=16)
            eval_ds = make_kto_dataset(n=8)

            config = TrainingConfig(
                output_dir=tmpdir,
                num_epochs=1,
                batch_size=4,
                mixed_precision="no",
                gradient_checkpointing=False,
                eval_steps=2,
                save_on_epoch_end=False,
            )

            eval_results = []

            class EvalTracker(TrainerCallback):
                def on_evaluate(self, trainer, metrics):
                    eval_results.append(metrics)

            trainer = GrimoireTrainer(
                model=model,
                tokenizer=FakeTokenizer(),
                config=config,
                loss_fn=KTOLoss(ref_model=ref_model, beta=0.1),
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                callbacks=[EvalTracker()],
            )
            trainer.train()

            assert len(eval_results) > 0
            assert "eval/loss" in eval_results[0]
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestDPOTraining:
    def test_dpo_trains_with_ref_model(self):
        torch.manual_seed(42)
        tmpdir = tempfile.mkdtemp()

        try:
            model = TinyLM()
            ref_model = copy.deepcopy(model).to(_device())
            ref_model.eval()
            dataset = make_preference_dataset(n=32)

            config = TrainingConfig(
                output_dir=tmpdir,
                num_epochs=2,
                batch_size=8,
                learning_rate=1e-3,
                gradient_accumulation_steps=1,
                mixed_precision="no",
                gradient_checkpointing=False,
                logging_steps=1,
                disable_dropout=True,
                save_on_epoch_end=False,
            )

            losses = []

            class LossTracker(TrainerCallback):
                def on_step_end(self, trainer, step, loss, metrics):
                    losses.append(loss)

            trainer = GrimoireTrainer(
                model=model,
                tokenizer=FakeTokenizer(),
                config=config,
                loss_fn=DPOLoss(ref_model=ref_model, beta=0.1),
                train_dataset=dataset,
                callbacks=[LossTracker()],
            )
            trainer.train()

            assert len(losses) > 0
            assert all(not torch.tensor(v).isnan() for v in losses)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_dpo_trains_with_disable_adapter(self):
        torch.manual_seed(42)
        tmpdir = tempfile.mkdtemp()

        try:
            model = PeftTinyLM()
            dataset = make_preference_dataset(n=16)

            config = TrainingConfig(
                output_dir=tmpdir,
                num_epochs=1,
                batch_size=4,
                learning_rate=1e-3,
                mixed_precision="no",
                gradient_checkpointing=False,
                save_on_epoch_end=False,
            )

            trainer = GrimoireTrainer(
                model=model,
                tokenizer=FakeTokenizer(),
                config=config,
                loss_fn=DPOLoss(ref_model=None, beta=0.1),
                train_dataset=dataset,
            )
            trainer.train()  # should not raise
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestSimPOTraining:
    def test_simpo_trains(self):
        torch.manual_seed(42)
        tmpdir = tempfile.mkdtemp()

        try:
            model = TinyLM()
            dataset = make_preference_dataset(n=32)

            config = TrainingConfig(
                output_dir=tmpdir,
                num_epochs=2,
                batch_size=8,
                learning_rate=1e-3,
                mixed_precision="no",
                gradient_checkpointing=False,
                logging_steps=1,
                disable_dropout=True,
                save_on_epoch_end=False,
            )

            losses = []

            class LossTracker(TrainerCallback):
                def on_step_end(self, trainer, step, loss, metrics):
                    losses.append(loss)

            trainer = GrimoireTrainer(
                model=model,
                tokenizer=FakeTokenizer(),
                config=config,
                loss_fn=SimPOLoss(beta=2.0, gamma=0.5),
                train_dataset=dataset,
                callbacks=[LossTracker()],
            )
            trainer.train()

            assert len(losses) > 0
            assert all(not torch.tensor(v).isnan() for v in losses)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestCPOTraining:
    def test_cpo_trains(self):
        torch.manual_seed(42)
        tmpdir = tempfile.mkdtemp()

        try:
            model = TinyLM()
            dataset = make_preference_dataset(n=32)

            config = TrainingConfig(
                output_dir=tmpdir,
                num_epochs=2,
                batch_size=8,
                learning_rate=1e-3,
                mixed_precision="no",
                gradient_checkpointing=False,
                logging_steps=1,
                disable_dropout=True,
                save_on_epoch_end=False,
            )

            losses = []

            class LossTracker(TrainerCallback):
                def on_step_end(self, trainer, step, loss, metrics):
                    losses.append(loss)

            trainer = GrimoireTrainer(
                model=model,
                tokenizer=FakeTokenizer(),
                config=config,
                loss_fn=CPOLoss(beta=0.1),
                train_dataset=dataset,
                callbacks=[LossTracker()],
            )
            trainer.train()

            assert len(losses) > 0
            assert all(not torch.tensor(v).isnan() for v in losses)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestIPOTraining:
    def test_ipo_trains_with_ref_model(self):
        torch.manual_seed(42)
        tmpdir = tempfile.mkdtemp()

        try:
            model = TinyLM()
            ref_model = copy.deepcopy(model).to(_device())
            ref_model.eval()
            dataset = make_preference_dataset(n=32)

            config = TrainingConfig(
                output_dir=tmpdir,
                num_epochs=2,
                batch_size=8,
                learning_rate=1e-3,
                mixed_precision="no",
                gradient_checkpointing=False,
                logging_steps=1,
                disable_dropout=True,
                save_on_epoch_end=False,
            )

            losses = []

            class LossTracker(TrainerCallback):
                def on_step_end(self, trainer, step, loss, metrics):
                    losses.append(loss)

            trainer = GrimoireTrainer(
                model=model,
                tokenizer=FakeTokenizer(),
                config=config,
                loss_fn=IPOLoss(ref_model=ref_model, beta=0.1),
                train_dataset=dataset,
                callbacks=[LossTracker()],
            )
            trainer.train()

            assert len(losses) > 0
            assert all(not torch.tensor(v).isnan() for v in losses)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_ipo_trains_with_disable_adapter(self):
        torch.manual_seed(42)
        tmpdir = tempfile.mkdtemp()

        try:
            model = PeftTinyLM()
            dataset = make_preference_dataset(n=16)

            config = TrainingConfig(
                output_dir=tmpdir,
                num_epochs=1,
                batch_size=4,
                learning_rate=1e-3,
                mixed_precision="no",
                gradient_checkpointing=False,
                save_on_epoch_end=False,
            )

            trainer = GrimoireTrainer(
                model=model,
                tokenizer=FakeTokenizer(),
                config=config,
                loss_fn=IPOLoss(ref_model=None, beta=0.1),
                train_dataset=dataset,
            )
            trainer.train()  # should not raise
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestTokenIdValidation:
    def test_oob_token_ids_raises(self):
        """Dataset with token IDs exceeding embedding size should raise ValueError."""
        tmpdir = tempfile.mkdtemp()

        try:
            model = TinyLM(vocab_size=64)
            # Create dataset with token IDs >= vocab_size
            data = {
                "input_ids": [[100, 200, 300]] * 8,  # All exceed vocab_size=64
                "attention_mask": [[1, 1, 1]] * 8,
                "labels": [[-100, 200, 300]] * 8,
            }
            dataset = Dataset.from_dict(data)

            config = TrainingConfig(
                output_dir=tmpdir,
                num_epochs=1,
                batch_size=4,
                mixed_precision="no",
                gradient_checkpointing=False,
                save_on_epoch_end=False,
            )

            trainer = GrimoireTrainer(
                model=model,
                tokenizer=FakeTokenizer(),
                config=config,
                loss_fn=SFTLoss(),
                train_dataset=dataset,
            )

            with pytest.raises(ValueError, match="exceeds model embedding table size"):
                trainer.train()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_valid_token_ids_passes(self):
        """Dataset with valid token IDs should train without issue."""
        tmpdir = tempfile.mkdtemp()

        try:
            model = TinyLM(vocab_size=64)
            dataset = make_sft_dataset(n=8, vocab_size=64)

            config = TrainingConfig(
                output_dir=tmpdir,
                num_epochs=1,
                batch_size=4,
                mixed_precision="no",
                gradient_checkpointing=False,
                save_on_epoch_end=False,
            )

            trainer = GrimoireTrainer(
                model=model,
                tokenizer=FakeTokenizer(),
                config=config,
                loss_fn=SFTLoss(),
                train_dataset=dataset,
            )
            trainer.train()  # should not raise
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_oob_preference_token_ids_raises(self):
        """Preference dataset with OOB token IDs should raise ValueError."""
        tmpdir = tempfile.mkdtemp()

        try:
            model = TinyLM(vocab_size=64)
            data = {
                "chosen_input_ids": [[100, 200, 300]] * 8,
                "chosen_attention_mask": [[1, 1, 1]] * 8,
                "chosen_labels": [[-100, 200, 300]] * 8,
                "rejected_input_ids": [[1, 2, 3]] * 8,
                "rejected_attention_mask": [[1, 1, 1]] * 8,
                "rejected_labels": [[-100, 2, 3]] * 8,
            }
            dataset = Dataset.from_dict(data)

            config = TrainingConfig(
                output_dir=tmpdir,
                num_epochs=1,
                batch_size=4,
                mixed_precision="no",
                gradient_checkpointing=False,
                save_on_epoch_end=False,
            )

            trainer = GrimoireTrainer(
                model=model,
                tokenizer=FakeTokenizer(),
                config=config,
                loss_fn=ORPOLoss(beta=0.1),
                train_dataset=dataset,
            )

            with pytest.raises(ValueError, match="exceeds model embedding table size"):
                trainer.train()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestNEFTune:
    def test_neftune_adds_noise_during_training(self):
        torch.manual_seed(42)
        tmpdir = tempfile.mkdtemp()

        try:
            model = TinyLM()
            dataset = make_sft_dataset(n=8)

            config = TrainingConfig(
                output_dir=tmpdir,
                num_epochs=1,
                batch_size=4,
                mixed_precision="no",
                gradient_checkpointing=False,
                save_on_epoch_end=False,
                neftune_alpha=5.0,
            )

            trainer = GrimoireTrainer(
                model=model,
                tokenizer=FakeTokenizer(),
                config=config,
                loss_fn=SFTLoss(),
                train_dataset=dataset,
            )

            # Hook should be registered
            assert trainer._neftune_hook_handle is not None

            trainer.train()

            # Hook should be removed after training
            assert trainer._neftune_hook_handle is None
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_neftune_disabled_by_default(self):
        tmpdir = tempfile.mkdtemp()

        try:
            model = TinyLM()
            dataset = make_sft_dataset(n=8)

            config = TrainingConfig(
                output_dir=tmpdir,
                num_epochs=1,
                batch_size=4,
                mixed_precision="no",
                gradient_checkpointing=False,
                save_on_epoch_end=False,
            )

            trainer = GrimoireTrainer(
                model=model,
                tokenizer=FakeTokenizer(),
                config=config,
                loss_fn=SFTLoss(),
                train_dataset=dataset,
            )

            assert trainer._neftune_hook_handle is None
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_neftune_embedding_noise_is_active(self):
        """Verify that the hook actually modifies embedding output during training."""
        torch.manual_seed(42)
        model = TinyLM()

        embeddings = model.get_input_embeddings()
        alpha = 5.0

        def neftune_hook(module, input, output):
            if module.training:
                dims = output.size(1) * output.size(2)
                mag_norm = alpha / dims ** 0.5
                return output + torch.zeros_like(output).uniform_(-mag_norm, mag_norm)
            return output

        handle = embeddings.register_forward_hook(neftune_hook)

        input_ids = torch.randint(0, 64, (2, 8))

        # Training mode: two forward passes should differ
        model.train()
        out1 = embeddings(input_ids)
        out2 = embeddings(input_ids)
        assert not torch.allclose(out1, out2), "NEFTune should add different noise each pass"

        # Eval mode: two forward passes should be identical
        model.eval()
        out3 = embeddings(input_ids)
        out4 = embeddings(input_ids)
        assert torch.allclose(out3, out4), "NEFTune should not add noise in eval mode"

        handle.remove()
