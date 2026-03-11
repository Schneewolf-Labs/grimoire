# Writing a Custom Loss Function

Adding a new training method to Grimoire means writing a loss class. The trainer, data loading, checkpointing, and multi-GPU support all stay the same.

## The Interface

Every loss function must implement two things:

```python
class MyLoss:
    def __call__(self, model, batch, training=True):
        """Compute the loss and return metrics.

        Args:
            model: The model (already wrapped by accelerate).
            batch: Dict of tensors from the data collator.
            training: True during training, False during evaluation.

        Returns:
            loss: A scalar tensor.
            metrics: A dict of metric name -> float value.
        """
        loss = ...
        metrics = {"my_metric": some_value}
        return loss, metrics

    def create_collator(self, pad_token_id):
        """Return a data collator that produces batches for this loss.

        Args:
            pad_token_id: The tokenizer's pad token ID.

        Returns:
            A callable that takes a list of examples and returns a batched dict.
        """
        return MyCollator(pad_token_id=pad_token_id)
```

That's it. The trainer calls `loss_fn(model, batch, training=True)` in the training loop and `loss_fn(model, batch, training=False)` during evaluation.

## The `training` Flag

The `training` parameter lets you use different logic for training vs evaluation:

- **Training:** Compute the full loss (e.g., preference terms, reference model comparison)
- **Evaluation:** Typically just NLL on the chosen/desirable sequences

All built-in preference losses use NLL on chosen sequences for eval, since preference metrics don't generalize well to held-out data:

```python
def __call__(self, model, batch, training=True):
    if not training:
        # Simple NLL eval
        outputs = model(
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
            labels=batch["chosen_labels"],
        )
        return outputs.loss, {}
    # Full training loss below...
```

## Metrics

The metrics dict you return gets:
1. Prefixed with `train/` or `eval/` by the trainer
2. Logged every `logging_steps` steps (to console and optionally wandb)
3. Passed to callbacks via `on_log` and `on_evaluate`

Return floats, not tensors. Detach before calling `.item()`:

```python
metrics = {
    "nll_loss": nll_loss.detach().item(),
    "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
    "reward_accuracy": (chosen_rewards > rejected_rewards).float().mean().item(),
}
```

Common metrics across built-in losses:
- `chosen_rewards` / `rejected_rewards` — implicit reward values
- `reward_margin` — gap between chosen and rejected rewards
- `reward_accuracy` — fraction of examples where chosen reward > rejected

## Choosing a Collator

Grimoire has three built-in collators. Pick the one that matches your data format:

### SFTCollator

For single-sequence data (prompt + response concatenated). Produces `input_ids`, `attention_mask`, `labels`.

```python
from grimoire.data.sft import SFTCollator

def create_collator(self, pad_token_id):
    return SFTCollator(pad_token_id=pad_token_id)
```

### PreferenceCollator

For paired preference data (chosen + rejected). Produces `chosen_input_ids`, `chosen_attention_mask`, `chosen_labels`, `rejected_input_ids`, `rejected_attention_mask`, `rejected_labels`.

```python
from grimoire.data.preference import PreferenceCollator

def create_collator(self, pad_token_id):
    return PreferenceCollator(pad_token_id=pad_token_id)
```

Used by: ORPO, DPO, SimPO, CPO, IPO.

### KTOCollator

For unpaired binary feedback. Like SFTCollator but adds a `kto_label` boolean tensor.

```python
from grimoire.data.kto import KTOCollator

def create_collator(self, pad_token_id):
    return KTOCollator(pad_token_id=pad_token_id)
```

### GRPOCollator

For prompt-only data. Produces `input_ids` and `attention_mask` with no labels — completions are generated during training.

```python
from grimoire.data.grpo import GRPOCollator

def create_collator(self, pad_token_id):
    return GRPOCollator(pad_token_id=pad_token_id)
```

### Writing a Custom Collator

If your data format doesn't fit any built-in collator, write your own. A collator is just a callable that takes a list of dicts and returns a batched dict of tensors:

```python
class MyCollator:
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        # features is a list of dicts from dataset.__getitem__
        # Return a dict of batched tensors
        max_len = max(len(f["input_ids"]) for f in features)
        # ... pad and stack into tensors ...
        return {"input_ids": input_ids, "attention_mask": attention_mask, ...}
```

## Walkthrough: A Minimal Preference Loss

Here's a complete, minimal preference loss that just maximizes the gap between chosen and rejected log probabilities:

```python
import torch
import torch.nn.functional as F
from grimoire.data.preference import PreferenceCollator
from grimoire.losses.orpo import _pad_dim1


class SimplePreferenceLoss:
    """Minimal preference loss: -mean(log(sigmoid(chosen_logp - rejected_logp)))"""

    def __init__(self, beta=1.0, label_pad_token_id=-100):
        self.beta = beta
        self.label_pad_token_id = label_pad_token_id
        self._pad_token_id = 0

    def __call__(self, model, batch, training=True):
        if not training:
            outputs = model(
                input_ids=batch["chosen_input_ids"],
                attention_mask=batch["chosen_attention_mask"],
                labels=batch["chosen_labels"],
            )
            return outputs.loss, {}

        len_chosen = batch["chosen_input_ids"].size(0)

        # Concatenate chosen + rejected for a single forward pass
        max_len = max(batch["chosen_input_ids"].size(1),
                      batch["rejected_input_ids"].size(1))

        input_ids = torch.cat([
            _pad_dim1(batch["chosen_input_ids"], max_len, self._pad_token_id),
            _pad_dim1(batch["rejected_input_ids"], max_len, self._pad_token_id),
        ], dim=0)

        attention_mask = torch.cat([
            _pad_dim1(batch["chosen_attention_mask"], max_len, 0),
            _pad_dim1(batch["rejected_attention_mask"], max_len, 0),
        ], dim=0)

        labels = torch.cat([
            _pad_dim1(batch["chosen_labels"], max_len, self.label_pad_token_id),
            _pad_dim1(batch["rejected_labels"], max_len, self.label_pad_token_id),
        ], dim=0)

        # Forward pass
        logits = model(input_ids=input_ids, attention_mask=attention_mask,
                       use_cache=False).logits

        # Average log probabilities per sequence
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].clone()
        loss_mask = shift_labels != self.label_pad_token_id
        shift_labels[shift_labels == self.label_pad_token_id] = 0

        per_token_logps = torch.gather(
            F.log_softmax(shift_logits, dim=-1),
            dim=2, index=shift_labels.unsqueeze(2),
        ).squeeze(2)

        avg_logps = (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        chosen_logps = avg_logps[:len_chosen]
        rejected_logps = avg_logps[len_chosen:]

        # Loss
        loss = -F.logsigmoid(self.beta * (chosen_logps - rejected_logps)).mean()

        metrics = {
            "reward_margin": (chosen_logps - rejected_logps).detach().mean().item(),
            "reward_accuracy": (chosen_logps > rejected_logps).float().mean().item(),
        }
        return loss, metrics

    def create_collator(self, pad_token_id):
        self._pad_token_id = pad_token_id
        return PreferenceCollator(pad_token_id=pad_token_id,
                                  label_pad_token_id=self.label_pad_token_id)
```

Use it like any built-in loss:

```python
trainer = GrimoireTrainer(
    model=model, tokenizer=tokenizer, config=config,
    loss_fn=SimplePreferenceLoss(beta=1.0), train_dataset=pref_dataset,
)
trainer.train()
```

## Key Patterns from Built-In Losses

1. **Single forward pass for preference methods.** Concatenate chosen + rejected, run once, split. Required for FSDP and faster than two passes.

2. **Average log probabilities.** Normalize by response length so short and long responses are comparable: `(per_token_logps * mask).sum(-1) / mask.sum(-1)`.

3. **Reference models.** Store as `self.ref_model`, call with `torch.no_grad()`. The caller (not the trainer) manages the reference model lifecycle.

4. **`_pad_dim1` helper.** Reuse from `grimoire.losses.orpo` to pad sequences to equal length before concatenation.

5. **`use_cache=False`.** Always pass this in the forward call — caching is incompatible with gradient checkpointing.
