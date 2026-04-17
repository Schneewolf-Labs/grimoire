"""SFT benchmark — Grimoire.

Run:
    python benchmarks/sft/grimoire_sft.py --framework grimoire
    accelerate launch benchmarks/sft/grimoire_sft.py --framework grimoire  # multi-GPU
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (
    BenchmarkResult,
    Timer,
    common_args,
    detect_num_gpus,
    framework_version,
    gpu_name,
    load_sft_samples,
    peak_vram_gb,
    precision_str,
    reset_peak_memory,
    save_result,
)


class _LossTracker:
    """Callback that records the most recent training loss."""

    last_loss = float("nan")
    num_steps = 0

    def on_train_begin(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass

    def on_epoch_begin(self, trainer, epoch):
        pass

    def on_epoch_end(self, trainer, epoch):
        pass

    def on_step_end(self, trainer, step, loss, metrics):
        self.last_loss = float(loss)
        self.num_steps = int(step)

    def on_log(self, trainer, metrics):
        pass

    def on_evaluate(self, trainer, metrics):
        pass

    def on_save(self, trainer, path):
        pass


def main():
    args = common_args("sft").parse_args()
    assert args.framework == "grimoire"

    from grimoire import GrimoireTrainer, SFTLoss, TrainingConfig, tokenize_sft

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if args.bf16 and not args.fp16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)

    if args.lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank * 2, task_type="CAUSAL_LM"))

    raw = load_sft_samples(args.dataset, args.split, args.num_samples, args.seed)
    dataset = raw.map(
        lambda x: tokenize_sft(
            x, tokenizer, max_length=args.max_length,
            prompt_field="prompt", response_field="response",
        ),
        remove_columns=raw.column_names,
    )

    total_tokens = sum(sum(1 for t in ex["labels"] if t != -100) for ex in dataset)

    config = TrainingConfig(
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        mixed_precision=precision_str(args),
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=10,
        save_on_epoch_end=False,
        seed=args.seed,
    )

    tracker = _LossTracker()
    trainer = GrimoireTrainer(
        model=model, tokenizer=tokenizer, config=config,
        loss_fn=SFTLoss(), train_dataset=dataset, callbacks=[tracker],
    )

    reset_peak_memory()
    with Timer() as t:
        trainer.train()

    final_loss = tracker.last_loss
    num_steps = tracker.num_steps or getattr(trainer, "global_step", 0) or 0

    result = BenchmarkResult(
        framework="grimoire",
        task="sft",
        model=args.model,
        dataset=args.dataset,
        num_samples=len(dataset),
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_length=args.max_length,
        lora=args.lora,
        precision=precision_str(args),
        num_gpus=detect_num_gpus(),
        wall_clock_seconds=t.elapsed,
        train_samples_per_second=len(dataset) * args.num_epochs / max(t.elapsed, 1e-9),
        train_tokens_per_second=total_tokens * args.num_epochs / max(t.elapsed, 1e-9),
        peak_vram_gb=peak_vram_gb(),
        final_train_loss=final_loss,
        num_train_steps=num_steps,
        total_train_tokens=total_tokens * args.num_epochs,
        framework_version=framework_version("grimoire"),
        torch_version=torch.__version__,
        gpu_name=gpu_name(),
        tag=args.tag,
    )
    path = save_result(result, args.results_file)
    print(f"[grimoire sft] wrote {path}")


if __name__ == "__main__":
    main()
