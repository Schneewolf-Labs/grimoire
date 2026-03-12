import os
import math
import glob
import shutil
import logging

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import get_scheduler

from .config import TrainingConfig

logger = logging.getLogger(__name__)


class GrimoireTrainer:
    """Multi-GPU training loop powered by accelerate.

    Works with any loss function that implements:
        loss, metrics = loss_fn(model, batch, training=True)
        collator = loss_fn.create_collator(pad_token_id)
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: TrainingConfig,
        loss_fn,
        train_dataset,
        eval_dataset=None,
        data_collator=None,
        peft_config=None,
        callbacks=None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.loss_fn = loss_fn
        self.callbacks = callbacks or []
        self.global_step = 0
        self.current_epoch = 0
        self._stop_requested = False

        # Ensure pad token exists
        if tokenizer.pad_token is None:
            logger.warning("Tokenizer has no pad_token, using eos_token as pad_token")
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Apply PEFT / LoRA
        if peft_config is not None:
            from peft import get_peft_model, prepare_model_for_kbit_training

            if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
                model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, peft_config)
            if hasattr(model, "print_trainable_parameters"):
                model.print_trainable_parameters()

        # Disable dropout for training stability (important for preference learning)
        if config.disable_dropout:
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0

        # Gradient checkpointing (use_reentrant=False for DDP/FSDP compatibility)
        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

        # Initialize accelerator
        tracker_kwargs = {}
        if config.log_with == "wandb":
            wandb_kwargs = {}
            if config.run_name:
                wandb_kwargs["name"] = config.run_name
            if config.wandb_tags:
                wandb_kwargs["tags"] = config.wandb_tags
            if config.wandb_notes:
                wandb_kwargs["notes"] = config.wandb_notes
            tracker_kwargs["wandb"] = wandb_kwargs

        self.accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with=config.log_with,
            project_dir=config.output_dir,
        )

        set_seed(config.seed)

        # Data collator — use loss function's default if not provided
        if data_collator is None:
            data_collator = loss_fn.create_collator(tokenizer.pad_token_id)

        # Dataloaders
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=data_collator,
            num_workers=config.dataloader_num_workers,
            pin_memory=config.dataloader_pin_memory,
            drop_last=True,
        )

        self.eval_dataloader = None
        if eval_dataset is not None:
            self.eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                collate_fn=data_collator,
                num_workers=config.dataloader_num_workers,
                pin_memory=config.dataloader_pin_memory,
            )

        # Optimizer
        optimizer = self._create_optimizer(model)

        # LR scheduler
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / config.gradient_accumulation_steps
        )
        self.max_steps = num_update_steps_per_epoch * config.num_epochs

        warmup_steps = config.warmup_steps if config.warmup_steps > 0 else int(self.max_steps * config.warmup_ratio)
        lr_scheduler = get_scheduler(
            config.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.max_steps,
        )

        # Prepare with accelerator (handles DDP, DeepSpeed, FSDP wrapping)
        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = (
            self.accelerator.prepare(model, optimizer, self.train_dataloader, lr_scheduler)
        )

        if self.eval_dataloader is not None:
            self.eval_dataloader = self.accelerator.prepare(self.eval_dataloader)

        # Init experiment tracking
        if config.log_with:
            self.accelerator.init_trackers(
                config.project_name or "grimoire",
                config=_config_to_dict(config),
                init_kwargs=tracker_kwargs,
            )

        # Resume from checkpoint
        if config.resume_from_checkpoint:
            self.accelerator.load_state(config.resume_from_checkpoint)
            # Extract step number from checkpoint directory name
            try:
                self.global_step = int(config.resume_from_checkpoint.rstrip("/").split("-")[-1])
            except ValueError:
                logger.warning("Could not parse step from checkpoint path, starting from step 0")

    def train(self):
        config = self.config

        self._log_info("***** Starting training *****")
        self._log_info(f"  Num examples = {len(self.train_dataloader.dataset)}")
        self._log_info(f"  Num epochs = {config.num_epochs}")
        self._log_info(f"  Batch size per device = {config.batch_size}")
        self._log_info(
            f"  Total batch size = "
            f"{config.batch_size * self.accelerator.num_processes * config.gradient_accumulation_steps}"
        )
        self._log_info(f"  Gradient accumulation steps = {config.gradient_accumulation_steps}")
        self._log_info(f"  Total optimization steps = {self.max_steps}")
        self._log_info(f"  Number of processes = {self.accelerator.num_processes}")

        self._fire("on_train_begin")

        if config.eval_on_start and self.eval_dataloader:
            self.evaluate()

        # Handle resuming mid-epoch
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / config.gradient_accumulation_steps
        )
        starting_epoch = self.global_step // num_update_steps_per_epoch if self.global_step > 0 else 0
        resume_step_in_epoch = self.global_step - (starting_epoch * num_update_steps_per_epoch)

        # Progress bar (main process only)
        progress_bar = tqdm(
            total=self.max_steps,
            initial=self.global_step,
            desc="Training",
            disable=not self.accelerator.is_main_process,
            dynamic_ncols=True,
        )

        for epoch in range(starting_epoch, config.num_epochs):
            self.current_epoch = epoch
            self._fire("on_epoch_begin", epoch=epoch)
            self.model.train()

            # Skip already-completed batches when resuming
            active_dataloader = self.train_dataloader
            if epoch == starting_epoch and resume_step_in_epoch > 0:
                active_dataloader = self.accelerator.skip_first_batches(
                    self.train_dataloader,
                    resume_step_in_epoch * config.gradient_accumulation_steps,
                )

            running_loss = 0.0
            steps_in_epoch = 0

            for step, batch in enumerate(active_dataloader):
                with self.accelerator.accumulate(self.model):
                    loss, metrics = self.loss_fn(self.model, batch, training=True)
                    self.accelerator.backward(loss)

                    if config.max_grad_norm and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), config.max_grad_norm)

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                # Only count actual optimization steps (after gradient accumulation)
                if self.accelerator.sync_gradients:
                    self.global_step += 1
                    steps_in_epoch += 1
                    running_loss += loss.detach().item()

                    # Update progress bar
                    avg_loss = running_loss / steps_in_epoch
                    lr = self.lr_scheduler.get_last_lr()[0]
                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")

                    self._fire("on_step_end", step=self.global_step, loss=loss.item(), metrics=metrics)

                    # Logging
                    if self.global_step % config.logging_steps == 0:
                        progress = self.global_step / self.max_steps
                        log_metrics = {
                            "train/loss": avg_loss,
                            "train/learning_rate": lr,
                            "train/epoch": epoch + (step + 1) / len(self.train_dataloader),
                            "train/global_step": self.global_step,
                            "train/progress": progress,
                            **{f"train/{k}": v for k, v in metrics.items()},
                        }
                        self._log_metrics(log_metrics)
                        self._fire("on_log", metrics=log_metrics)

                    # Evaluation
                    if config.eval_steps and self.eval_dataloader and self.global_step % config.eval_steps == 0:
                        self.evaluate()
                        self.model.train()

                    # Checkpointing
                    if config.save_steps and self.global_step % config.save_steps == 0:
                        self._save_checkpoint()

                    # Graceful stop
                    if self._stop_requested:
                        self._log_info(f"Stopping early at step {self.global_step}")
                        break

            self._fire("on_epoch_end", epoch=epoch)

            if self._stop_requested:
                break

            if config.save_on_epoch_end:
                self._save_checkpoint()

            if self.eval_dataloader:
                self.evaluate()

        progress_bar.close()
        self._fire("on_train_end")

        if config.log_with:
            self.accelerator.end_training()

        self._log_info("***** Training complete *****")

    @torch.no_grad()
    def evaluate(self):
        """Run evaluation loop and return metrics."""
        self.model.eval()
        total_loss = 0.0
        total_metrics = {}
        num_batches = 0

        for batch in self.eval_dataloader:
            loss, metrics = self.loss_fn(self.model, batch, training=False)

            # Reduce loss across all processes (average, not gather+mean)
            loss = self.accelerator.reduce(loss, reduction="mean")
            total_loss += loss.item()

            # Reduce metrics across all processes (batched into single tensor)
            if metrics:
                keys = sorted(metrics.keys())
                vals = torch.tensor([metrics[k] for k in keys], device=self.accelerator.device)
                vals = self.accelerator.reduce(vals, reduction="mean")
                for k, v in zip(keys, vals):
                    total_metrics[k] = total_metrics.get(k, 0.0) + v.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_metrics = {k: v / max(num_batches, 1) for k, v in total_metrics.items()}

        eval_results = {"eval/loss": avg_loss, **{f"eval/{k}": v for k, v in avg_metrics.items()}}

        self._log_info(f"  Eval — loss: {avg_loss:.4f}" + "".join(f" | {k}: {v:.4f}" for k, v in avg_metrics.items()))
        self._log_metrics(eval_results)
        self._fire("on_evaluate", metrics=eval_results)

        self.model.train()
        return eval_results

    def request_stop(self):
        """Request graceful stop at the end of the current step."""
        self._stop_requested = True
        self._log_info("Stop requested — will stop after current step")

    @property
    def stopped_early(self):
        """Whether training was stopped early via request_stop()."""
        return self._stop_requested

    def save_model(self, output_dir=None):
        """Save the final model and tokenizer."""
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.accelerator.wait_for_everyone()
        unwrapped = self.accelerator.unwrap_model(self.model)
        unwrapped.save_pretrained(
            output_dir,
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save,
        )
        if self.accelerator.is_main_process:
            self.tokenizer.save_pretrained(output_dir)
            self._log_info(f"Model saved to {output_dir}")

    # ---- Internal helpers ----

    def _create_optimizer(self, model):
        no_decay = ["bias", "layer_norm.weight", "LayerNorm.weight"]
        param_groups = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        lr = self.config.learning_rate

        opt = self.config.optimizer

        if opt in ("adamw", "adamw_torch", "adamw_hf"):
            kwargs = {}
            if torch.cuda.is_available():
                kwargs["fused"] = True
            return torch.optim.AdamW(param_groups, lr=lr, **kwargs)
        elif opt in ("adamw_8bit", "adamw_bnb_8bit"):
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(param_groups, lr=lr)
        elif opt == "paged_adamw_8bit":
            import bitsandbytes as bnb
            return bnb.optim.PagedAdamW8bit(param_groups, lr=lr)
        elif opt in ("paged_adamw_32bit", "paged_adamw"):
            import bitsandbytes as bnb
            return bnb.optim.PagedAdamW32bit(param_groups, lr=lr)
        elif opt == "adafactor":
            from transformers.optimization import Adafactor
            return Adafactor(param_groups, lr=lr, relative_step=False, scale_parameter=False)
        elif opt == "sgd":
            return torch.optim.SGD(param_groups, lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {opt}")

    def _save_checkpoint(self):
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}")
        self.accelerator.save_state(checkpoint_dir)
        self._fire("on_save", path=checkpoint_dir)
        self._log_info(f"Checkpoint saved to {checkpoint_dir}")

        if self.accelerator.is_main_process and self.config.save_total_limit:
            self._rotate_checkpoints()

    def _rotate_checkpoints(self):
        checkpoints = sorted(
            glob.glob(os.path.join(self.config.output_dir, "checkpoint-*")),
            key=lambda x: int(x.rsplit("-", 1)[-1]),
        )
        if len(checkpoints) > self.config.save_total_limit:
            for old in checkpoints[: len(checkpoints) - self.config.save_total_limit]:
                shutil.rmtree(old)
                logger.debug(f"Deleted old checkpoint: {old}")

    def _log_metrics(self, metrics):
        if self.config.log_with:
            self.accelerator.log(metrics, step=self.global_step)
        if self.accelerator.is_main_process:
            parts = [f"{k}: {_fmt(v)}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()]
            logger.info(f"[step {self.global_step}] {' | '.join(parts)}")

    def _log_info(self, msg):
        if self.accelerator.is_main_process:
            logger.info(msg)

    def _fire(self, event, **kwargs):
        for cb in self.callbacks:
            fn = getattr(cb, event, None)
            if fn:
                fn(self, **kwargs)


def _fmt(v):
    """Format a float for logging — use scientific notation for very small values."""
    if abs(v) < 1e-3 and v != 0.0:
        return f"{v:.2e}"
    return f"{v:.4f}"


def _config_to_dict(config):
    """Convert a dataclass config to a dict for experiment tracking."""
    return {k: v for k, v in config.__dict__.items() if not k.startswith("_")}
