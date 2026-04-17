"""SFT benchmark — TRL (HuggingFace Transformers Reinforcement Learning).

Run:
    python benchmarks/sft/trl_sft.py --framework trl
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


def main():
    args = common_args("sft").parse_args()
    assert args.framework == "trl"

    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if args.bf16 and not args.fp16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)

    peft_config = None
    if args.lora:
        from peft import LoraConfig
        peft_config = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank * 2, task_type="CAUSAL_LM")

    raw = load_sft_samples(args.dataset, args.split, args.num_samples, args.seed)
    # TRL's SFTTrainer accepts a single "text" field by default.
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
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        max_length=args.max_length,
        dataset_text_field="text",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, args=sft_config,
        train_dataset=dataset, peft_config=peft_config,
    )

    reset_peak_memory()
    with Timer() as t:
        train_output = trainer.train()

    # Approximate token count (all text tokens count in TRL SFT)
    tokenized = dataset.map(
        lambda x: {"n": len(tokenizer(x["text"], truncation=True, max_length=args.max_length)["input_ids"])},
    )
    total_tokens = sum(tokenized["n"])

    metrics = train_output.metrics if hasattr(train_output, "metrics") else {}
    final_loss = float(metrics.get("train_loss", float("nan")))

    result = BenchmarkResult(
        framework="trl",
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
        framework_version=framework_version("trl"),
        torch_version=torch.__version__,
        gpu_name=gpu_name(),
        tag=args.tag,
        extra={"trainer_metrics": {k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()}},
    )
    path = save_result(result, args.results_file)
    print(f"[trl sft] wrote {path}")


if __name__ == "__main__":
    main()
