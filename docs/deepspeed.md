# Multi-GPU, DeepSpeed, and FSDP

Grimoire uses `accelerate` for distributed training. No code changes needed — the same script works on 1 GPU or many.

## Quick Start

```bash
# Interactive setup (recommended first time)
accelerate config

# Or launch directly
accelerate launch --multi_gpu --num_processes 4 train.py
```

## Multi-GPU (DDP)

The simplest distributed setup. Each GPU gets a full copy of the model.

```bash
accelerate launch --multi_gpu --num_processes 4 train.py
```

**When to use:** Your model fits in a single GPU's memory. This is the fastest option since there's no sharding overhead.

## DeepSpeed

Use DeepSpeed when the model is too large to fit on a single GPU, or when you want memory-efficient training with ZeRO.

```bash
accelerate launch --use_deepspeed --deepspeed_config ds_config.json train.py
```

### ZeRO Stage 2

Shards optimizer states and gradients across GPUs. The model itself is replicated.

```json
{
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "none"
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto"
}
```

**When to use:** Model fits on one GPU, but optimizer states don't (common with Adam which keeps 2x model size in states). Good default for multi-GPU training.

### ZeRO Stage 3

Shards everything — model parameters, optimizer states, and gradients — across GPUs.

```json
{
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "none"
        },
        "offload_param": {
            "device": "none"
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto"
}
```

**When to use:** Model doesn't fit on a single GPU. Slower than Stage 2 due to parameter gathering, but lets you train much larger models.

### CPU Offloading

For extreme memory pressure, offload optimizer states or parameters to CPU:

```json
"offload_optimizer": {
    "device": "cpu",
    "pin_memory": true
},
"offload_param": {
    "device": "cpu",
    "pin_memory": true
}
```

This trades speed for memory — only use it when you've exhausted GPU memory.

## FSDP

PyTorch's Fully Sharded Data Parallel. Alternative to DeepSpeed ZeRO Stage 3.

```bash
accelerate launch --use_fsdp \
    --fsdp_sharding_strategy FULL_SHARD \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer \
    train.py
```

**When to use:** Same use case as ZeRO Stage 3 (model too large for one GPU). FSDP is native PyTorch, so it may integrate better with other PyTorch tools. DeepSpeed generally has better documentation and more tuning knobs.

## Grimoire-Specific Notes

### Gradient Checkpointing

Enabled by default (`gradient_checkpointing=True` in `TrainingConfig`). Grimoire uses `use_reentrant=False` for compatibility with DDP and FSDP — this is handled automatically.

### Mixed Precision

Set via `TrainingConfig.mixed_precision`:

```python
config = TrainingConfig(
    mixed_precision="bf16",   # recommended for modern GPUs (A100, H100)
    # mixed_precision="fp16", # older GPUs (V100)
    # mixed_precision="no",   # full precision
)
```

This is passed directly to `accelerate.Accelerator`. If you're also specifying precision in your DeepSpeed config, make sure they match.

### Reference Models (DPO/IPO/KTO)

Reference models are not wrapped by `accelerate` — they stay on their original device. For multi-GPU setups, each process loads its own copy of the reference model. This means DPO/IPO/KTO memory usage is roughly `2x model size per GPU`, regardless of sharding strategy.

If memory is tight, consider reference-free methods (ORPO, SimPO, CPO) instead.

### Gradient Accumulation

Set `gradient_accumulation_steps` in `TrainingConfig`, not in the DeepSpeed config. Use `"auto"` in the DeepSpeed config so accelerate passes the value through:

```python
config = TrainingConfig(
    batch_size=2,
    gradient_accumulation_steps=4,  # effective batch size = 2 * 4 * num_gpus
)
```

## Decision Guide

| Situation | Recommendation |
|-----------|----------------|
| Model fits on 1 GPU | Multi-GPU DDP (fastest) |
| Optimizer states don't fit | DeepSpeed ZeRO Stage 2 |
| Model doesn't fit on 1 GPU | DeepSpeed ZeRO Stage 3 or FSDP |
| Extreme memory pressure | ZeRO Stage 3 + CPU offloading |
| Need reference model (DPO/IPO/KTO) | Consider reference-free methods first; otherwise budget 2x memory |
