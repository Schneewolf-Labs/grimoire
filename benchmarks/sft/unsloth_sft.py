"""SFT benchmark — Unsloth.

Unsloth is single-GPU only and patches the model loader. We keep the
hyperparameters aligned with the other scripts so results are still
directly comparable on a single GPU.

Run:
    python benchmarks/sft/unsloth_sft.py --framework unsloth
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))

import torch

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


def main():
    args = common_args("sft").parse_args()
    assert args.framework == "unsloth"

    # Unsloth must be imported before transformers/trl to install its patches.
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_length,
        dtype=torch.bfloat16 if args.bf16 and not args.fp16 else torch.float16,
        load_in_4bit=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.lora:
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing="unsloth" if args.gradient_checkpointing else False,
            random_state=args.seed,
        )

    raw = load_sft_samples(args.dataset, args.split, args.num_samples, args.seed)
    dataset = raw.map(lambda x: {"text": x["prompt"] + x["response"]}, remove_columns=raw.column_names)

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16 and not args.fp16,
        fp16=args.fp16,
        # Unsloth manages gradient checkpointing via get_peft_model above when LoRA.
        gradient_checkpointing=args.gradient_checkpointing and not args.lora,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        max_length=args.max_length,
        dataset_text_field="text",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, args=sft_config, train_dataset=dataset,
    )

    reset_peak_memory()
    with Timer() as t:
        train_output = trainer.train()

    tokenized = dataset.map(
        lambda x: {"n": len(tokenizer(x["text"], truncation=True, max_length=args.max_length)["input_ids"])},
    )
    total_tokens = sum(tokenized["n"])
    metrics = train_output.metrics if hasattr(train_output, "metrics") else {}
    final_loss = float(metrics.get("train_loss", float("nan")))

    result = BenchmarkResult(
        framework="unsloth",
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
        train_samples_per_second=float(metrics.get("train_samples_per_second", len(dataset) * args.num_epochs / max(t.elapsed, 1e-9))),
        train_tokens_per_second=total_tokens * args.num_epochs / max(t.elapsed, 1e-9),
        peak_vram_gb=peak_vram_gb(),
        final_train_loss=final_loss,
        num_train_steps=int(metrics.get("train_steps", getattr(trainer.state, "global_step", 0))),
        total_train_tokens=total_tokens * args.num_epochs,
        framework_version=framework_version("unsloth"),
        torch_version=torch.__version__,
        gpu_name=gpu_name(),
        tag=args.tag,
        extra={"trainer_metrics": {k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()}},
    )
    path = save_result(result, args.results_file)
    print(f"[unsloth sft] wrote {path}")


if __name__ == "__main__":
    main()
