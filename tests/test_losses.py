"""Tests for SFT and ORPO loss functions."""

import torch
import torch.nn as nn
import pytest
from grimoire.losses.sft import SFTLoss
from grimoire.losses.orpo import ORPOLoss, _pad_dim1


class SimpleModel(nn.Module):
    """Tiny model for testing loss computation."""

    def __init__(self, vocab_size=32, hidden_size=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size
        self.config = type("Config", (), {"is_encoder_decoder": False})()

    def forward(self, input_ids, attention_mask=None, labels=None, use_cache=False):
        h = self.embed(input_ids)
        logits = self.head(h)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return type("Output", (), {"logits": logits, "loss": loss})()


class TestSFTLoss:
    def test_returns_scalar_loss(self):
        model = SimpleModel()
        loss_fn = SFTLoss()

        batch = {
            "input_ids": torch.randint(0, 32, (2, 10)),
            "attention_mask": torch.ones(2, 10, dtype=torch.long),
            "labels": torch.randint(0, 32, (2, 10)),
        }
        loss, metrics = loss_fn(model, batch)

        assert loss.dim() == 0  # scalar
        assert loss.item() > 0
        assert isinstance(metrics, dict)

    def test_masked_labels_reduce_loss(self):
        model = SimpleModel()
        loss_fn = SFTLoss()

        input_ids = torch.randint(0, 32, (2, 10))

        # All tokens in loss
        batch_full = {
            "input_ids": input_ids.clone(),
            "attention_mask": torch.ones(2, 10, dtype=torch.long),
            "labels": input_ids.clone(),
        }

        # Half tokens masked
        labels_masked = input_ids.clone()
        labels_masked[:, :5] = -100
        batch_masked = {
            "input_ids": input_ids.clone(),
            "attention_mask": torch.ones(2, 10, dtype=torch.long),
            "labels": labels_masked,
        }

        loss_full, _ = loss_fn(model, batch_full)
        loss_masked, _ = loss_fn(model, batch_masked)

        # Different number of tokens in loss should give different loss values
        assert loss_full.item() != loss_masked.item()

    def test_creates_correct_collator(self):
        loss_fn = SFTLoss()
        from grimoire.data.sft import SFTCollator
        collator = loss_fn.create_collator(pad_token_id=0)
        assert isinstance(collator, SFTCollator)


class TestORPOLoss:
    def test_returns_scalar_loss_and_metrics(self):
        model = SimpleModel()
        loss_fn = ORPOLoss(beta=0.1)
        loss_fn._pad_token_id = 0

        batch = {
            "chosen_input_ids": torch.randint(0, 32, (2, 8)),
            "chosen_attention_mask": torch.ones(2, 8, dtype=torch.long),
            "chosen_labels": torch.randint(0, 32, (2, 8)),
            "rejected_input_ids": torch.randint(0, 32, (2, 8)),
            "rejected_attention_mask": torch.ones(2, 8, dtype=torch.long),
            "rejected_labels": torch.randint(0, 32, (2, 8)),
        }
        # Ensure labels have some -100 to simulate prompt masking
        batch["chosen_labels"][:, :2] = -100
        batch["rejected_labels"][:, :2] = -100

        loss, metrics = loss_fn(model, batch, training=True)

        assert loss.dim() == 0
        assert loss.item() > 0
        assert "nll_loss" in metrics
        assert "or_loss" in metrics
        assert "chosen_rewards" in metrics
        assert "rejected_rewards" in metrics
        assert "log_odds_ratio" in metrics
        assert "reward_margin" in metrics
        assert "reward_accuracy" in metrics

    def test_eval_mode_uses_chosen_only(self):
        model = SimpleModel()
        loss_fn = ORPOLoss(beta=0.1)

        batch = {
            "chosen_input_ids": torch.randint(0, 32, (2, 8)),
            "chosen_attention_mask": torch.ones(2, 8, dtype=torch.long),
            "chosen_labels": torch.randint(0, 32, (2, 8)),
            "rejected_input_ids": torch.randint(0, 32, (2, 8)),
            "rejected_attention_mask": torch.ones(2, 8, dtype=torch.long),
            "rejected_labels": torch.randint(0, 32, (2, 8)),
        }

        loss, metrics = loss_fn(model, batch, training=False)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_beta_scales_or_loss(self):
        model = SimpleModel()
        torch.manual_seed(42)

        batch = {
            "chosen_input_ids": torch.randint(0, 32, (2, 8)),
            "chosen_attention_mask": torch.ones(2, 8, dtype=torch.long),
            "chosen_labels": torch.randint(0, 32, (2, 8)),
            "rejected_input_ids": torch.randint(0, 32, (2, 8)),
            "rejected_attention_mask": torch.ones(2, 8, dtype=torch.long),
            "rejected_labels": torch.randint(0, 32, (2, 8)),
        }
        batch["chosen_labels"][:, :2] = -100
        batch["rejected_labels"][:, :2] = -100

        loss_fn_low = ORPOLoss(beta=0.01)
        loss_fn_low._pad_token_id = 0
        loss_fn_high = ORPOLoss(beta=1.0)
        loss_fn_high._pad_token_id = 0

        _, metrics_low = loss_fn_low(model, batch, training=True)
        _, metrics_high = loss_fn_high(model, batch, training=True)

        # Higher beta should scale the OR loss component
        assert abs(metrics_high["or_loss"]) > abs(metrics_low["or_loss"])

    def test_creates_correct_collator(self):
        loss_fn = ORPOLoss()
        from grimoire.data.preference import PreferenceCollator
        collator = loss_fn.create_collator(pad_token_id=0)
        assert isinstance(collator, PreferenceCollator)

    def test_handles_different_chosen_rejected_lengths(self):
        model = SimpleModel()
        loss_fn = ORPOLoss(beta=0.1)
        loss_fn._pad_token_id = 0

        batch = {
            "chosen_input_ids": torch.randint(0, 32, (2, 6)),
            "chosen_attention_mask": torch.ones(2, 6, dtype=torch.long),
            "chosen_labels": torch.randint(0, 32, (2, 6)),
            "rejected_input_ids": torch.randint(0, 32, (2, 10)),
            "rejected_attention_mask": torch.ones(2, 10, dtype=torch.long),
            "rejected_labels": torch.randint(0, 32, (2, 10)),
        }
        batch["chosen_labels"][:, :2] = -100
        batch["rejected_labels"][:, :2] = -100

        loss, metrics = loss_fn(model, batch, training=True)
        assert loss.dim() == 0
        assert not torch.isnan(loss)


class TestPadDim1:
    def test_pads_correctly(self):
        t = torch.tensor([[1, 2], [3, 4]])
        padded = _pad_dim1(t, 4, 0)
        assert padded.shape == (2, 4)
        assert padded[0].tolist() == [1, 2, 0, 0]

    def test_no_op_when_already_long_enough(self):
        t = torch.tensor([[1, 2, 3]])
        padded = _pad_dim1(t, 2, 0)
        assert padded.shape == (1, 3)

    def test_preserves_device_and_dtype(self):
        t = torch.tensor([[1, 2]], dtype=torch.long)
        padded = _pad_dim1(t, 4, -100)
        assert padded.dtype == torch.long
        assert padded[0].tolist() == [1, 2, -100, -100]
