"""Tests for SFT, ORPO, DPO, and SimPO loss functions."""

import copy

import torch
import torch.nn as nn
import pytest
from grimoire.losses.sft import SFTLoss
from grimoire.losses.orpo import ORPOLoss, _pad_dim1
from grimoire.losses.dpo import DPOLoss
from grimoire.losses.simpo import SimPOLoss


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


def _make_preference_batch(vocab_size=32, chosen_len=8, rejected_len=8, batch_size=2, prompt_len=2):
    """Helper to create a preference batch with prompt masking."""
    batch = {
        "chosen_input_ids": torch.randint(0, vocab_size, (batch_size, chosen_len)),
        "chosen_attention_mask": torch.ones(batch_size, chosen_len, dtype=torch.long),
        "chosen_labels": torch.randint(0, vocab_size, (batch_size, chosen_len)),
        "rejected_input_ids": torch.randint(0, vocab_size, (batch_size, rejected_len)),
        "rejected_attention_mask": torch.ones(batch_size, rejected_len, dtype=torch.long),
        "rejected_labels": torch.randint(0, vocab_size, (batch_size, rejected_len)),
    }
    batch["chosen_labels"][:, :prompt_len] = -100
    batch["rejected_labels"][:, :prompt_len] = -100
    return batch


class TestDPOLoss:
    def _make_loss(self, ref_model=None, beta=0.1):
        model = SimpleModel()
        if ref_model is None:
            ref_model = copy.deepcopy(model)
            ref_model.eval()
        loss_fn = DPOLoss(ref_model=ref_model, beta=beta)
        loss_fn._pad_token_id = 0
        return model, loss_fn

    def test_returns_scalar_loss_and_metrics(self):
        model, loss_fn = self._make_loss()
        batch = _make_preference_batch()

        loss, metrics = loss_fn(model, batch, training=True)

        assert loss.dim() == 0
        assert loss.item() > 0
        assert "chosen_rewards" in metrics
        assert "rejected_rewards" in metrics
        assert "reward_margin" in metrics
        assert "reward_accuracy" in metrics
        assert "log_odds_ratio" in metrics

    def test_eval_mode_uses_chosen_only(self):
        model, loss_fn = self._make_loss()
        batch = _make_preference_batch()

        loss, metrics = loss_fn(model, batch, training=False)
        assert loss.dim() == 0
        assert loss.item() > 0
        assert metrics == {}

    def test_beta_scales_loss(self):
        torch.manual_seed(42)
        ref_model = SimpleModel()
        ref_model.eval()

        # Use a different policy model so pi != pi_ref (otherwise loss is always log(2))
        policy = SimpleModel()
        batch = _make_preference_batch()

        loss_fn_low = DPOLoss(ref_model=ref_model, beta=0.01)
        loss_fn_low._pad_token_id = 0
        loss_fn_high = DPOLoss(ref_model=ref_model, beta=1.0)
        loss_fn_high._pad_token_id = 0

        loss_low, _ = loss_fn_low(policy, batch, training=True)
        loss_high, _ = loss_fn_high(policy, batch, training=True)

        # With identical policy and ref, loss = log(2) regardless of beta.
        # With different models, higher beta amplifies the difference.
        assert loss_low.item() != loss_high.item()

    def test_creates_correct_collator(self):
        ref_model = SimpleModel()
        loss_fn = DPOLoss(ref_model=ref_model)
        from grimoire.data.preference import PreferenceCollator
        collator = loss_fn.create_collator(pad_token_id=0)
        assert isinstance(collator, PreferenceCollator)

    def test_handles_different_chosen_rejected_lengths(self):
        model, loss_fn = self._make_loss()
        batch = _make_preference_batch(chosen_len=6, rejected_len=10)

        loss, metrics = loss_fn(model, batch, training=True)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_ref_model_affects_loss(self):
        """Verify that changing the reference model changes the loss."""
        torch.manual_seed(42)
        batch = _make_preference_batch()

        policy = SimpleModel()

        # ref_model identical to policy
        ref_same = copy.deepcopy(policy)
        ref_same.eval()
        loss_fn_same = DPOLoss(ref_model=ref_same, beta=0.1)
        loss_fn_same._pad_token_id = 0
        loss_same, _ = loss_fn_same(policy, batch, training=True)

        # ref_model with different weights
        torch.manual_seed(999)
        ref_diff = SimpleModel()
        ref_diff.eval()
        loss_fn_diff = DPOLoss(ref_model=ref_diff, beta=0.1)
        loss_fn_diff._pad_token_id = 0
        loss_diff, _ = loss_fn_diff(policy, batch, training=True)

        assert loss_same.item() != loss_diff.item()

    def test_identical_policy_and_ref_gives_log2(self):
        """When policy == ref, logratios cancel and loss = log(2)."""
        torch.manual_seed(42)
        model = SimpleModel()
        ref_model = copy.deepcopy(model)
        ref_model.eval()

        loss_fn = DPOLoss(ref_model=ref_model, beta=0.1)
        loss_fn._pad_token_id = 0
        batch = _make_preference_batch()

        loss, _ = loss_fn(model, batch, training=True)
        # -log(sigmoid(0)) = log(2) ≈ 0.6931
        assert abs(loss.item() - 0.6931) < 0.01


class TestSimPOLoss:
    def test_returns_scalar_loss_and_metrics(self):
        model = SimpleModel()
        loss_fn = SimPOLoss(beta=2.0, gamma=0.5)
        loss_fn._pad_token_id = 0

        batch = _make_preference_batch()

        loss, metrics = loss_fn(model, batch, training=True)

        assert loss.dim() == 0
        assert loss.item() > 0
        assert "chosen_rewards" in metrics
        assert "rejected_rewards" in metrics
        assert "reward_margin" in metrics
        assert "reward_accuracy" in metrics
        assert "logps_diff" in metrics

    def test_eval_mode_uses_chosen_only(self):
        model = SimpleModel()
        loss_fn = SimPOLoss()

        batch = _make_preference_batch()

        loss, metrics = loss_fn(model, batch, training=False)
        assert loss.dim() == 0
        assert loss.item() > 0
        assert metrics == {}

    def test_beta_scales_loss(self):
        torch.manual_seed(42)
        model = SimpleModel()
        batch = _make_preference_batch()

        loss_fn_low = SimPOLoss(beta=0.5, gamma=0.0)
        loss_fn_low._pad_token_id = 0
        loss_fn_high = SimPOLoss(beta=5.0, gamma=0.0)
        loss_fn_high._pad_token_id = 0

        loss_low, _ = loss_fn_low(model, batch, training=True)
        loss_high, _ = loss_fn_high(model, batch, training=True)

        # Higher beta amplifies the logp difference, changing the loss
        assert loss_low.item() != loss_high.item()

    def test_gamma_increases_loss(self):
        """Higher gamma margin should increase loss when chosen-rejected gap is small."""
        torch.manual_seed(42)
        model = SimpleModel()
        batch = _make_preference_batch()

        loss_fn_no_margin = SimPOLoss(beta=2.0, gamma=0.0)
        loss_fn_no_margin._pad_token_id = 0
        loss_fn_high_margin = SimPOLoss(beta=2.0, gamma=5.0)
        loss_fn_high_margin._pad_token_id = 0

        loss_no, _ = loss_fn_no_margin(model, batch, training=True)
        loss_high, _ = loss_fn_high_margin(model, batch, training=True)

        # Higher gamma subtracts more from the logp diff, making sigmoid input
        # more negative, so loss increases
        assert loss_high.item() > loss_no.item()

    def test_handles_different_chosen_rejected_lengths(self):
        model = SimpleModel()
        loss_fn = SimPOLoss(beta=2.0, gamma=0.5)
        loss_fn._pad_token_id = 0

        batch = _make_preference_batch(chosen_len=6, rejected_len=10)

        loss, metrics = loss_fn(model, batch, training=True)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_creates_correct_collator(self):
        loss_fn = SimPOLoss()
        from grimoire.data.preference import PreferenceCollator
        collator = loss_fn.create_collator(pad_token_id=0)
        assert isinstance(collator, PreferenceCollator)

    def test_no_ref_model_needed(self):
        """SimPO should work without any reference model."""
        loss_fn = SimPOLoss()
        assert not hasattr(loss_fn, "ref_model")
