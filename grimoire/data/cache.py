"""Cache reference model log probabilities in the dataset.

Since the reference model is frozen, its log probabilities never change
during training. Computing them once and storing them in the dataset
eliminates the need to keep the reference model in memory and halves
the forward passes per step for DPO, KTO, and IPO.
"""

import torch
from torch.utils.data import DataLoader

from ..losses.utils import get_batch_logps as _get_batch_logps, _disable_grad_checkpointing


def cache_reference_log_probs(
    ref_model,
    dataset,
    collator,
    batch_size=8,
    label_pad_token_id=-100,
    num_workers=0,
):
    """Precompute reference model log probabilities and add them to the dataset.

    After caching, the reference model can be deleted to free memory.
    The loss functions (DPO, KTO, IPO) will automatically use the cached
    values instead of running a forward pass through the reference model.

    Args:
        ref_model: Frozen reference model (must be in eval mode).
        dataset: Training dataset (list of dicts or HuggingFace Dataset).
        collator: Data collator (from loss_fn.create_collator).
        batch_size: Batch size for ref model inference.
        label_pad_token_id: Token ID used for masked labels (default -100).
        num_workers: DataLoader workers (default 0).

    Returns:
        The dataset with cached reference log probabilities added:
        - Preference data (DPO/IPO): adds ``ref_chosen_logps`` and
          ``ref_rejected_logps`` per example.
        - KTO data: adds ``ref_logps`` per example.

    Example::

        from grimoire.losses import DPOLoss
        from grimoire.data import cache_reference_log_probs

        loss_fn = DPOLoss(ref_model=ref_model, beta=0.1)
        collator = loss_fn.create_collator(tokenizer.pad_token_id)
        dataset = cache_reference_log_probs(ref_model, dataset, collator)

        # Free the ref model
        del ref_model
        torch.cuda.empty_cache()

        # Train without ref_model
        trainer = GrimoireTrainer(
            model=model, tokenizer=tokenizer, config=config,
            loss_fn=DPOLoss(beta=0.1), train_dataset=dataset,
        )
    """
    if ref_model.training:
        raise ValueError("ref_model must be in eval mode (call ref_model.eval() first)")

    device = next(ref_model.parameters()).device

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
    )

    # Detect data format from first example
    first = dataset[0]
    is_preference = "chosen_input_ids" in first

    if is_preference:
        all_chosen_logps = []
        all_rejected_logps = []
    else:
        all_logps = []

    with _disable_grad_checkpointing(ref_model), torch.no_grad():
        for batch in loader:
            if is_preference:
                chosen_logps = _forward_logps(
                    ref_model,
                    batch["chosen_input_ids"].to(device),
                    batch["chosen_attention_mask"].to(device),
                    batch["chosen_labels"].to(device),
                    label_pad_token_id,
                )
                all_chosen_logps.append(chosen_logps.cpu())

                rejected_logps = _forward_logps(
                    ref_model,
                    batch["rejected_input_ids"].to(device),
                    batch["rejected_attention_mask"].to(device),
                    batch["rejected_labels"].to(device),
                    label_pad_token_id,
                )
                all_rejected_logps.append(rejected_logps.cpu())
            else:
                logps = _forward_logps(
                    ref_model,
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["labels"].to(device),
                    label_pad_token_id,
                )
                all_logps.append(logps.cpu())

    if is_preference:
        chosen_logps = torch.cat(all_chosen_logps)
        rejected_logps = torch.cat(all_rejected_logps)
        return _add_columns(dataset, {
            "ref_chosen_logps": chosen_logps.tolist(),
            "ref_rejected_logps": rejected_logps.tolist(),
        })
    else:
        logps = torch.cat(all_logps)
        return _add_columns(dataset, {
            "ref_logps": logps.tolist(),
        })


def _forward_logps(model, input_ids, attention_mask, labels, label_pad_token_id):
    """Run a forward pass and return average log probs per sequence."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    return _get_batch_logps(logits, labels, label_pad_token_id)



def _add_columns(dataset, columns):
    """Add columns to a dataset, handling both list-of-dicts and HF Datasets."""
    # HuggingFace Dataset
    if hasattr(dataset, "add_column"):
        for name, values in columns.items():
            dataset = dataset.add_column(name, values)
        return dataset

    # List of dicts
    for name, values in columns.items():
        for i, v in enumerate(values):
            dataset[i][name] = v
    return dataset
