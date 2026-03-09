# Callbacks

Callbacks let you hook into the training loop at key points — logging, early stopping, custom checkpointing, etc.

## Writing a Callback

Subclass `TrainerCallback` and override the hooks you need:

```python
from grimoire import TrainerCallback

class MyCallback(TrainerCallback):
    def on_train_begin(self, trainer):
        print(f"Training for {trainer.max_steps} steps")

    def on_step_end(self, trainer, step, loss, metrics):
        print(f"Step {step}: loss={loss:.4f}")

    def on_train_end(self, trainer):
        print("Done!")

trainer = GrimoireTrainer(..., callbacks=[MyCallback()])
```

Multiple callbacks are supported — pass a list and they'll all fire in order.

## Available Hooks

| Hook | Arguments | When it fires |
|------|-----------|---------------|
| `on_train_begin` | `trainer` | Once, before the first epoch |
| `on_train_end` | `trainer` | Once, after all epochs complete (or early stop) |
| `on_epoch_begin` | `trainer`, `epoch` | Start of each epoch (0-indexed) |
| `on_epoch_end` | `trainer`, `epoch` | End of each epoch |
| `on_step_end` | `trainer`, `step`, `loss`, `metrics` | After each optimization step (not each micro-batch) |
| `on_log` | `trainer`, `metrics` | Every `logging_steps` steps |
| `on_evaluate` | `trainer`, `metrics` | After each evaluation run |
| `on_save` | `trainer`, `path` | After each checkpoint save |

## Trainer Properties Available in Callbacks

Inside any callback, the `trainer` object gives you access to:

| Property | Type | Description |
|----------|------|-------------|
| `trainer.global_step` | `int` | Current optimization step |
| `trainer.current_epoch` | `int` | Current epoch (0-indexed) |
| `trainer.max_steps` | `int` | Total number of optimization steps |
| `trainer.config` | `TrainingConfig` | The training configuration |
| `trainer.model` | `nn.Module` | The model (wrapped by accelerate) |
| `trainer.optimizer` | `Optimizer` | The optimizer |
| `trainer.lr_scheduler` | `LRScheduler` | The learning rate scheduler |
| `trainer.accelerator` | `Accelerator` | The accelerate instance |
| `trainer.stopped_early` | `bool` | Whether training was stopped via `request_stop()` |

## Early Stopping

Call `trainer.request_stop()` to gracefully stop training after the current step:

```python
class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=3):
        self.patience = patience
        self.best_loss = float("inf")
        self.wait = 0

    def on_evaluate(self, trainer, metrics):
        loss = metrics["eval/loss"]
        if loss < self.best_loss:
            self.best_loss = loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"No improvement for {self.patience} evals, stopping")
                trainer.request_stop()
```

After training, check `trainer.stopped_early` to know whether it finished naturally or was stopped.

## Metrics Logging

The `on_log` hook receives a dict with all current metrics:

```python
class LoggingCallback(TrainerCallback):
    def on_log(self, trainer, metrics):
        # metrics includes:
        #   train/loss, train/learning_rate, train/epoch,
        #   train/global_step, train/progress
        #   + any loss-specific metrics (e.g., train/reward_margin)
        print(f"[{metrics['train/global_step']}] loss={metrics['train/loss']:.4f}")
```

The `on_evaluate` hook receives eval metrics:

```python
class EvalCallback(TrainerCallback):
    def on_evaluate(self, trainer, metrics):
        # metrics includes:
        #   eval/loss + any loss-specific metrics
        print(f"Eval loss: {metrics['eval/loss']:.4f}")
```

## Custom Checkpointing

The `on_save` hook fires after each checkpoint with the checkpoint path:

```python
class CheckpointCallback(TrainerCallback):
    def on_save(self, trainer, path):
        print(f"Checkpoint saved to {path}")
        # e.g., upload to cloud storage, notify a webhook, etc.
```
