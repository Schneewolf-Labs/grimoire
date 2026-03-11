"""Tests for TrainingConfig dataclass."""

from grimoire.config import TrainingConfig


class TestTrainingConfigDefaults:
    def test_default_output_dir(self):
        config = TrainingConfig()
        assert config.output_dir == "./output"

    def test_default_training_hyperparams(self):
        config = TrainingConfig()
        assert config.num_epochs == 3
        assert config.batch_size == 4
        assert config.gradient_accumulation_steps == 1
        assert config.learning_rate == 2e-5
        assert config.weight_decay == 0.01
        assert config.warmup_ratio == 0.1
        assert config.warmup_steps == 0
        assert config.max_grad_norm == 1.0
        assert config.max_length == 2048

    def test_default_precision_and_optimization(self):
        config = TrainingConfig()
        assert config.mixed_precision == "bf16"
        assert config.gradient_checkpointing is True
        assert config.optimizer == "adamw"
        assert config.lr_scheduler == "cosine"
        assert config.disable_dropout is False

    def test_default_logging(self):
        config = TrainingConfig()
        assert config.logging_steps == 10
        assert config.log_with is None
        assert config.project_name is None
        assert config.run_name is None

    def test_default_evaluation(self):
        config = TrainingConfig()
        assert config.eval_steps is None
        assert config.eval_on_start is False

    def test_default_checkpointing(self):
        config = TrainingConfig()
        assert config.save_steps is None
        assert config.save_total_limit == 2
        assert config.save_on_epoch_end is True
        assert config.resume_from_checkpoint is None

    def test_default_seed(self):
        config = TrainingConfig()
        assert config.seed == 42


class TestTrainingConfigCustomValues:
    def test_custom_output_dir(self):
        config = TrainingConfig(output_dir="/tmp/my_run")
        assert config.output_dir == "/tmp/my_run"

    def test_custom_training_hyperparams(self):
        config = TrainingConfig(
            num_epochs=10,
            batch_size=16,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            weight_decay=0.0,
            max_grad_norm=0.5,
            max_length=512,
        )
        assert config.num_epochs == 10
        assert config.batch_size == 16
        assert config.gradient_accumulation_steps == 4
        assert config.learning_rate == 1e-4
        assert config.weight_decay == 0.0
        assert config.max_grad_norm == 0.5
        assert config.max_length == 512

    def test_warmup_steps_field(self):
        # warmup_steps > 0 means it overrides warmup_ratio (used in trainer)
        config = TrainingConfig(warmup_steps=100, warmup_ratio=0.05)
        assert config.warmup_steps == 100
        assert config.warmup_ratio == 0.05  # both stored, trainer uses warmup_steps

    def test_custom_optimizer_and_scheduler(self):
        config = TrainingConfig(optimizer="sgd", lr_scheduler="linear")
        assert config.optimizer == "sgd"
        assert config.lr_scheduler == "linear"

    def test_custom_logging(self):
        config = TrainingConfig(
            logging_steps=5,
            log_with="wandb",
            project_name="my_project",
            run_name="run_001",
        )
        assert config.logging_steps == 5
        assert config.log_with == "wandb"
        assert config.project_name == "my_project"
        assert config.run_name == "run_001"

    def test_custom_evaluation(self):
        config = TrainingConfig(eval_steps=50, eval_on_start=True)
        assert config.eval_steps == 50
        assert config.eval_on_start is True

    def test_custom_checkpointing(self):
        config = TrainingConfig(
            save_steps=100,
            save_total_limit=5,
            save_on_epoch_end=False,
            resume_from_checkpoint="/tmp/checkpoint-500",
        )
        assert config.save_steps == 100
        assert config.save_total_limit == 5
        assert config.save_on_epoch_end is False
        assert config.resume_from_checkpoint == "/tmp/checkpoint-500"

    def test_disable_dropout(self):
        config = TrainingConfig(disable_dropout=True)
        assert config.disable_dropout is True

    def test_custom_seed(self):
        config = TrainingConfig(seed=0)
        assert config.seed == 0

    def test_mixed_precision_options(self):
        for mp in ("no", "fp16", "bf16"):
            config = TrainingConfig(mixed_precision=mp)
            assert config.mixed_precision == mp
