import gc
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


def _newton_schulz_5(G, steps=5):
    """Compute the matrix sign function via 5 Newton-Schulz iterations.

    Approximates G @ (G^T G)^{-1/2} which orthogonalizes the matrix.
    Uses quintic iteration coefficients from Bernstein & Bonev (2024).
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G / (G.norm() + 1e-7)
    if G.size(-2) > G.size(-1):
        X = X.transpose(-1, -2)
    for _ in range(steps):
        A = X @ X.transpose(-1, -2)
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.transpose(-1, -2)
    return X


class Muon(torch.optim.Optimizer):
    """Muon optimizer — Momentum + Orthogonalization.

    Applies Newton-Schulz orthogonalization to the momentum buffer for 2D+
    parameters (linear layers). Non-matrix parameters (embeddings, biases,
    norms) should be handled by a separate AdamW optimizer passed via
    ``adam_optimizer``.

    Based on: https://github.com/KellerJordan/Muon

    Args:
        params: Parameters for Muon (should be 2D+ tensors only).
        lr: Learning rate for Muon params (default: 0.02).
        momentum: Momentum coefficient (default: 0.95).
        nesterov: Use Nesterov momentum (default: True).
        ns_steps: Newton-Schulz iteration count (default: 5).
        adam_optimizer: AdamW optimizer for non-matrix params (optional).
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, adam_optimizer=None):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps)
        super().__init__(params, defaults)
        self.adam_optimizer = adam_optimizer

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                # Orthogonalize via Newton-Schulz for 2D+ params
                orig_shape = g.shape
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                g = _newton_schulz_5(g, steps=ns_steps)
                g = g.view(orig_shape)

                # Scale update by max(1, sqrt(fan_in / fan_out))
                # This normalizes the update magnitude across layers
                if g.ndim >= 2:
                    fan_in = g.shape[1] * (g[0].numel() // g.shape[1] if g.ndim > 2 else 1)
                    fan_out = g.shape[0]
                    scale = max(1, (fan_in / fan_out) ** 0.5)
                    p.add_(g, alpha=-lr * scale)
                else:
                    p.add_(g, alpha=-lr)

                state["step"] += 1

        # Step the inner AdamW for non-matrix params
        if self.adam_optimizer is not None:
            self.adam_optimizer.step()

        return loss

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        if self.adam_optimizer is not None:
            self.adam_optimizer.zero_grad(set_to_none=set_to_none)

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

        # Resize embeddings if tokenizer has more tokens than the model
        # (common with abliterated/extended models).  Without this, any
        # token ID >= vocab_size causes an out-of-bounds nn.Embedding
        # lookup on GPU — an async CUDA error that surfaces later at the
        # next synchronize() call (typically eval), making it look like
        # eval is broken.  TRL does this automatically; we must too.
        if hasattr(model, "get_input_embeddings") and hasattr(model, "resize_token_embeddings"):
            embedding_size = model.get_input_embeddings().weight.shape[0]
            if len(tokenizer) > embedding_size:
                logger.warning(
                    f"Tokenizer vocab ({len(tokenizer)}) > model embeddings ({embedding_size}). "
                    f"Resizing embeddings to match tokenizer."
                )
                model.resize_token_embeddings(len(tokenizer))

        # Apply PEFT / LoRA
        if peft_config is not None:
            from peft import get_peft_model, prepare_model_for_kbit_training

            if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
                # use_reentrant=False is required for flash attention backward.
                # The interaction between gradient checkpointing +
                # torch.no_grad() + quantized weights is handled by
                # _disable_grad_checkpointing() in the loss functions.
                model = prepare_model_for_kbit_training(
                    model,
                    use_gradient_checkpointing=config.gradient_checkpointing,
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                )
            model = get_peft_model(model, peft_config)
            if hasattr(model, "print_trainable_parameters"):
                model.print_trainable_parameters()

        # Disable dropout for training stability (important for preference learning)
        if config.disable_dropout:
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0

        # Disable KV cache at the config level — transformers.Trainer always
        # does this (TrainingArguments.use_cache defaults to False).  Even
        # though we pass use_cache=False as a forward kwarg, some model
        # architectures (e.g. Qwen) have internal layers that check
        # model.config.use_cache directly.  Without this, the model may
        # allocate or manage KV cache tensors during training, wasting
        # memory and potentially causing CUDA errors.
        if hasattr(model, "config"):
            model.config.use_cache = False

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
                drop_last=True,
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

        # NEFTune: add uniform noise to embeddings during training
        self._neftune_hook_handle = None
        if config.neftune_alpha is not None and config.neftune_alpha > 0:
            self._neftune_hook_handle = self._register_neftune_hook(config.neftune_alpha)

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

        # Pre-flight: validate that token IDs are within embedding bounds.
        # Out-of-bounds IDs cause async CUDA errors that surface later at
        # unrelated operations (flash-attn, cuBLAS, etc.), making them
        # extremely hard to diagnose.
        self._validate_token_ids()

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
                    del batch  # Free input tensors before backward
                    self.accelerator.backward(loss)

                    if config.max_grad_norm and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), config.max_grad_norm)

                    self.optimizer.step()
                    # Only step LR when the optimizer actually updated —
                    # fp16 GradScaler may skip steps on inf/nan gradients
                    if not self.accelerator.optimizer_step_was_skipped:
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
                        try:
                            self.evaluate()
                        except RuntimeError as e:
                            self._log_info(f"Eval failed at step {self.global_step}: {e}")
                            if _is_cuda_error(e):
                                self._log_info("CUDA context corrupted — stopping training")
                                self._stop_requested = True
                        self.model.train()

                    # Checkpointing
                    if config.save_steps and self.global_step % config.save_steps == 0:
                        try:
                            self._save_checkpoint()
                        except RuntimeError as e:
                            self._log_info(f"Checkpoint failed at step {self.global_step}: {e}")

                    # Graceful stop
                    if self._stop_requested:
                        self._log_info(f"Stopping early at step {self.global_step}")
                        break

            self._fire("on_epoch_end", epoch=epoch)

            if self._stop_requested:
                break

            if config.save_on_epoch_end:
                try:
                    self._save_checkpoint()
                except RuntimeError as e:
                    self._log_info(f"End-of-epoch checkpoint failed: {e}")

            if self.eval_dataloader:
                try:
                    self.evaluate()
                except RuntimeError as e:
                    self._log_info(f"End-of-epoch eval failed: {e}")
                    if _is_cuda_error(e):
                        self._log_info("CUDA context corrupted — stopping training")
                        self._stop_requested = True

        progress_bar.close()
        self._fire("on_train_end")

        # Remove NEFTune hook so the model produces clean outputs for inference
        if self._neftune_hook_handle is not None:
            self._neftune_hook_handle.remove()
            self._neftune_hook_handle = None

        if config.log_with:
            self.accelerator.end_training()

        self._log_info("***** Training complete *****")

    @torch.no_grad()
    def evaluate(self):
        """Run evaluation loop and return metrics."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Disable gradient checkpointing during eval — it interacts badly
        # with torch.no_grad() on quantized models (bitsandbytes 4-bit),
        # causing CUDA illegal memory access.  TRL does the same thing via
        # its disable_gradient_checkpointing context manager.
        grad_ckpt_was_enabled = getattr(self.model, "is_gradient_checkpointing", False)
        if grad_ckpt_was_enabled:
            self.model.gradient_checkpointing_disable()

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

            # Free batch tensors and defragment CUDA memory between eval steps
            # (mirrors transformers Trainer.evaluation_loop behavior)
            del batch, loss, metrics
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_loss = total_loss / max(num_batches, 1)
        avg_metrics = {k: v / max(num_batches, 1) for k, v in total_metrics.items()}

        eval_results = {"eval/loss": avg_loss, **{f"eval/{k}": v for k, v in avg_metrics.items()}}

        self._log_info(f"  Eval — loss: {avg_loss:.4f}" + "".join(f" | {k}: {v:.4f}" for k, v in avg_metrics.items()))
        self._log_metrics(eval_results)
        self._fire("on_evaluate", metrics=eval_results)

        # Restore gradient checkpointing and training mode
        if grad_ckpt_was_enabled:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.model.train()

        # Defragment CUDA memory before returning to training.
        # Eval forward passes create allocations of varying sizes that
        # fragment the memory pool.  Without this, the next training
        # backward pass may fail to find a contiguous block (OOM despite
        # having enough total free memory).
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

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

    def _register_neftune_hook(self, alpha):
        """Register a forward hook that adds uniform noise to embeddings during training.

        NEFTune (Jain et al., 2023) adds noise scaled by alpha / sqrt(seq_len * hidden_dim)
        to the embedding output. The hook is only active when the model is in training mode.
        """
        unwrapped = self.accelerator.unwrap_model(self.model)
        embeddings = unwrapped.get_input_embeddings()

        def neftune_forward_hook(module, input, output):
            if module.training:
                dims = output.size(1) * output.size(2)
                mag_norm = alpha / dims ** 0.5
                return output + torch.zeros_like(output).uniform_(-mag_norm, mag_norm)
            return output

        return embeddings.register_forward_hook(neftune_forward_hook)

    def _validate_token_ids(self):
        """Check that dataset token IDs fit within the model's embedding table.

        An out-of-bounds embedding lookup silently corrupts CUDA memory and
        surfaces later as an unrelated illegal-memory-access error in flash
        attention, cuBLAS, or other kernels — extremely hard to diagnose.
        """
        unwrapped = self.accelerator.unwrap_model(self.model)
        if not hasattr(unwrapped, "get_input_embeddings"):
            return
        emb = unwrapped.get_input_embeddings()
        if emb is None:
            return
        vocab_size = emb.weight.shape[0]

        # Scan a few raw examples from the dataset (CPU-only, no GPU cost,
        # doesn't consume dataloader batches).
        dataset = self.train_dataloader.dataset
        n_to_check = min(len(dataset), 50)
        for i in range(n_to_check):
            example = dataset[i]
            for key in ("input_ids", "chosen_input_ids", "rejected_input_ids"):
                ids = example.get(key)
                if ids is None:
                    continue
                max_id = max(ids) if isinstance(ids, list) else ids.max().item()
                if max_id >= vocab_size:
                    raise ValueError(
                        f"Token ID {max_id} in '{key}' (example {i}) exceeds "
                        f"model embedding table size ({vocab_size}). The dataset "
                        f"was likely tokenized with a different tokenizer or "
                        f"the model's vocabulary is smaller than expected."
                    )

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
        elif opt == "muon":
            return self._create_muon_optimizer(model, lr)
        elif opt == "sgd":
            return torch.optim.SGD(param_groups, lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {opt}")

    def _create_muon_optimizer(self, model, lr):
        """Create a Muon optimizer with AdamW for non-matrix params.

        Muon applies Newton-Schulz orthogonalization to momentum updates for
        2D+ parameters (linear layers), while using AdamW for 1D parameters
        (embeddings, biases, layer norms). The embedding/head params use a
        lower learning rate (0.1x) since they benefit from more conservative
        updates.
        """
        muon_params = []
        adam_decay_params = []
        adam_no_decay_params = []

        no_decay = ["bias", "layer_norm.weight", "LayerNorm.weight"]

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Muon works on 2D+ params (linear layers) that aren't embeddings
            if param.ndim >= 2 and "embed" not in name and "lm_head" not in name:
                muon_params.append(param)
            elif any(nd in name for nd in no_decay):
                adam_no_decay_params.append(param)
            else:
                adam_decay_params.append(param)

        adam_lr = lr * 0.1
        kwargs = {}
        if torch.cuda.is_available():
            kwargs["fused"] = True
        adam_optimizer = torch.optim.AdamW(
            [
                {"params": adam_decay_params, "weight_decay": self.config.weight_decay},
                {"params": adam_no_decay_params, "weight_decay": 0.0},
            ],
            lr=adam_lr,
            **kwargs,
        )

        return Muon(muon_params, lr=lr, momentum=0.95, adam_optimizer=adam_optimizer)

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


def _is_cuda_error(exc):
    """Check if a RuntimeError is a fatal CUDA error (corrupted context, unrecoverable)."""
    msg = str(exc).lower()
    return "cuda error" in msg or "cuda" in msg and "illegal" in msg


def _config_to_dict(config):
    """Convert a dataclass config to a dict for experiment tracking."""
    return {k: v for k, v in config.__dict__.items() if not k.startswith("_")}
