"""DPO benchmark — Unsloth.

Single-GPU only. LoRA is strongly recommended here — Unsloth's DPO path
is optimized for adapter training.

Run:
    python benchmarks/dpo/unsloth_dpo.py --framework unsloth --lora
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
    load_preference_samples,
    peak_vram_gb,
    precision_str,
    reset_peak_memory,
    save_result,
)


def main():
    args = common_args("dpo").parse_args()
    assert args.framework == "unsloth"

    from unsloth import FastLanguageModel, PatchDPOTrainer
    PatchDPOTrainer()
    from trl import DPOConfig, DPOTrainer

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

    dataset = load_preference_samples(args.dataset, args.split, args.num_samples, args.seed)

    total_tokens = 0
    for ex in dataset:
        total_tokens += len(tokenizer(ex["prompt"] + ex["chosen"], truncation=True, max_length=args.max_length)["input_ids"])
        total_tokens += len(tokenizer(ex["prompt"] + ex["rejected"], truncation=True, max_length=args.max_length)["input_ids"])

    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16 and not args.fp16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing and not args.lora,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        max_length=args.max_length,
        max_prompt_length=args.max_length // 2,
        beta=0.1,
        seed=args.seed,
    )

    trainer = DPOTrainer(
        model=model, ref_model=None, tokenizer=tokenizer, args=dpo_config, train_dataset=dataset,
    )

    reset_peak_memory()
    with Timer() as t:
        train_output = trainer.train()

    metrics = train_output.metrics if hasattr(train_output, "metrics") else {}
    final_loss = float(metrics.get("train_loss", float("nan")))

    result = BenchmarkResult(
        framework="unsloth",
        task="dpo",
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
    print(f"[unsloth dpo] wrote {path}")


if __name__ == "__main__":
    main()
