# Grimoire

A simple, multi-GPU LLM fine-tuning library. One training loop, pluggable loss functions.

Built as the training engine behind [Merlina](https://github.com/schneewolflabs/Merlina), replacing TRL's ORPO trainer after it was marked experimental.

## Why

TRL's ORPO implementation is unstable — it lives in `trl.experimental` and can break between releases. But ORPO's math is simple: it's just SFT loss plus an odds ratio term. The training loop infrastructure (multi-GPU, checkpointing, gradient accumulation) is the hard part, and `accelerate` already handles it well.

Grimoire is the result: ~400 lines of code that give you SFT and ORPO training with native multi-GPU support.

## Install

```bash
pip install -e .

# With optional dependencies
pip install -e ".[quantization]"   # bitsandbytes for 4-bit/8-bit
pip install -e ".[logging]"        # wandb
pip install -e ".[all]"            # everything
```

## Quick start

### SFT (Supervised Fine-Tuning)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from grimoire import GrimoireTrainer, TrainingConfig
from grimoire.losses import SFTLoss
from grimoire.data import tokenize_sft

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B-Instruct")

# Tokenize — mask prompt tokens so the model only learns to generate responses
dataset = dataset.map(
    lambda x: tokenize_sft(x, tokenizer, max_length=2048,
                           prompt_field="prompt", response_field="response"),
    remove_columns=dataset.column_names,
)

trainer = GrimoireTrainer(
    model=model,
    tokenizer=tokenizer,
    config=TrainingConfig(
        output_dir="./output",
        num_epochs=3,
        batch_size=4,
        learning_rate=2e-5,
    ),
    loss_fn=SFTLoss(),
    train_dataset=dataset,
)

trainer.train()
trainer.save_model("./my-model")
```

### ORPO (Odds Ratio Preference Optimization)

Same trainer, different loss function. No reference model needed.

```python
from grimoire.losses import ORPOLoss
from grimoire.data import tokenize_preference

# Dataset has prompt, chosen, rejected columns
dataset = dataset.map(
    lambda x: tokenize_preference(x, tokenizer, max_length=2048),
    remove_columns=dataset.column_names,
)

trainer = GrimoireTrainer(
    model=model,
    tokenizer=tokenizer,
    config=TrainingConfig(
        output_dir="./output",
        num_epochs=2,
        batch_size=2,
        learning_rate=5e-6,
        disable_dropout=True,     # recommended for preference learning
    ),
    loss_fn=ORPOLoss(beta=0.1),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

### With LoRA

Pass a `peft_config` and Grimoire handles the rest — including `prepare_model_for_kbit_training` for quantized models.

```python
from peft import LoraConfig

trainer = GrimoireTrainer(
    model=model,
    tokenizer=tokenizer,
    config=TrainingConfig(...),
    loss_fn=SFTLoss(),
    train_dataset=dataset,
    peft_config=LoraConfig(
        r=64,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                         "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
    ),
)
```

## Multi-GPU

No code changes. Configure with `accelerate` and launch:

```bash
# Interactive setup
accelerate config

# Or launch directly
accelerate launch --multi_gpu --num_processes 4 train.py

# With DeepSpeed
accelerate launch --use_deepspeed --deepspeed_config ds_config.json train.py
```

The same script works on 1 GPU or 8. `accelerate` handles DDP, DeepSpeed ZeRO, and FSDP.

## Callbacks

Subclass `TrainerCallback` and override the hooks you need:

```python
from grimoire import TrainerCallback

class MyCallback(TrainerCallback):
    def on_step_end(self, trainer, step, loss, metrics):
        if should_stop():
            trainer.request_stop()   # graceful early stopping

    def on_log(self, trainer, metrics):
        print(f"Step {trainer.global_step}/{trainer.max_steps}: {metrics}")

    def on_evaluate(self, trainer, metrics):
        print(f"Eval loss: {metrics['eval/loss']:.4f}")

trainer = GrimoireTrainer(..., callbacks=[MyCallback()])
```

Available hooks: `on_train_begin`, `on_train_end`, `on_epoch_begin`, `on_epoch_end`, `on_step_end`, `on_log`, `on_evaluate`, `on_save`.

## Configuration

`TrainingConfig` fields with their defaults:

| Field | Default | Description |
|---|---|---|
| `output_dir` | `"./output"` | Checkpoints and saved models |
| `num_epochs` | `3` | Number of training epochs |
| `batch_size` | `4` | Per-device batch size |
| `gradient_accumulation_steps` | `1` | Steps before optimizer update |
| `learning_rate` | `2e-5` | Peak learning rate |
| `weight_decay` | `0.01` | L2 regularization |
| `warmup_ratio` | `0.1` | Fraction of steps for LR warmup |
| `warmup_steps` | `0` | Overrides `warmup_ratio` if > 0 |
| `max_grad_norm` | `1.0` | Gradient clipping |
| `max_length` | `2048` | Maximum sequence length |
| `mixed_precision` | `"bf16"` | `"no"`, `"fp16"`, or `"bf16"` |
| `gradient_checkpointing` | `True` | Trade compute for memory |
| `optimizer` | `"adamw"` | See supported optimizers below |
| `lr_scheduler` | `"cosine"` | `"linear"`, `"cosine"`, `"constant"`, `"constant_with_warmup"` |
| `disable_dropout` | `False` | Set `True` for ORPO/DPO |
| `logging_steps` | `10` | Log metrics every N steps |
| `eval_steps` | `None` | Evaluate every N steps |
| `save_steps` | `None` | Checkpoint every N steps |
| `save_total_limit` | `2` | Max checkpoints to keep |
| `save_on_epoch_end` | `True` | Checkpoint after each epoch |
| `resume_from_checkpoint` | `None` | Path to resume from |
| `seed` | `42` | Random seed |
| `log_with` | `None` | `"wandb"` for W&B tracking |

**Supported optimizers:** `adamw`, `adamw_torch`, `adamw_hf`, `adamw_8bit`, `paged_adamw_8bit`, `paged_adamw_32bit`, `adafactor`, `sgd`

## How it works

### Architecture

```
grimoire/
├── trainer.py         # GrimoireTrainer — the training loop
├── config.py          # TrainingConfig dataclass
├── callbacks.py       # TrainerCallback base class
├── losses/
│   ├── sft.py         # SFT loss — model(input_ids, labels=labels).loss
│   └── orpo.py        # ORPO loss — NLL + odds ratio
└── data/
    ├── sft.py         # SFTCollator + tokenize_sft()
    └── preference.py  # PreferenceCollator + tokenize_preference()
```

### Loss function interface

Every loss function is a callable with a `create_collator` method:

```python
class MyLoss:
    def __call__(self, model, batch, training=True):
        # Your forward pass and loss computation
        return loss, metrics_dict

    def create_collator(self, pad_token_id):
        # Return a collator that knows your batch format
        return MyCollator(pad_token_id)
```

The trainer calls `loss_fn(model, batch, training=True)` in the training loop and `loss_fn(model, batch, training=False)` during evaluation. This is the only interface — adding a new training method (DPO, CPO, etc.) means writing a new loss class. The training loop doesn't change.

### ORPO loss

From the [paper](https://arxiv.org/abs/2403.07691):

```
L = NLL(chosen) + beta * -mean(log(sigmoid(log_odds_ratio)))
```

Where the odds ratio compares how likely the model thinks the chosen response is vs the rejected one. No reference model needed — the signal comes directly from the contrast between chosen and rejected.

Implementation details:
- **Single forward pass** — chosen and rejected are concatenated into one batch, run through the model once, then split. This is faster and required for FSDP.
- **Average log probabilities** — normalized by response length so short and long responses are comparable.
- **`log1p` for stability** — `log(1-P)` is computed as `log1p(-exp(log_P))` to avoid numerical issues near P=1.

### SFT loss

Delegates entirely to the model's built-in cross-entropy:

```python
outputs = model(input_ids=..., attention_mask=..., labels=...)
return outputs.loss, {}
```

Prompt tokens are masked with `-100` in labels during tokenization, so they're automatically excluded from the loss.

## Adding a new training method

Write a loss function. That's it. Here's a sketch for DPO:

```python
class DPOLoss:
    def __init__(self, beta=0.1, ref_model=None):
        self.beta = beta
        self.ref_model = ref_model

    def __call__(self, model, batch, training=True):
        # 1. Get policy log probs (same as ORPO)
        # 2. Get reference log probs from ref_model
        # 3. DPO loss = -log(sigmoid(beta * (policy_diff - ref_diff)))
        return loss, metrics

    def create_collator(self, pad_token_id):
        return PreferenceCollator(pad_token_id=pad_token_id)
```

The trainer, collators, tokenization — all reused. Zero changes to existing code.

## License

MIT
