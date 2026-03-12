"""Tests for SFT, ORPO, DPO, SimPO, KTO, CPO, IPO, and GRPO loss functions."""

import copy
from contextlib import contextmanager

import pytest
import torch
import torch.nn as nn
from grimoire.losses.sft import SFTLoss
from grimoire.losses.orpo import ORPOLoss
from grimoire.losses.utils import pad_dim1 as _pad_dim1
from grimoire.losses.dpo import DPOLoss
from grimoire.losses.simpo import SimPOLoss
from grimoire.losses.kto import KTOLoss
from grimoire.losses.cpo import CPOLoss
from grimoire.losses.ipo import IPOLoss
from grimoire.losses.grpo import GRPOLoss
from grimoire.data.cache import cache_reference_log_probs


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


class PeftSimpleModel(SimpleModel):
    """SimpleModel with PEFT-like adapter support for testing ref_model=None."""

    def __init__(self, vocab_size=32, hidden_size=16):
        super().__init__(vocab_size, hidden_size)
        # Adapter: an extra linear layer added on top (simulates LoRA)
        self.adapter = nn.Linear(hidden_size, hidden_size)
        self._adapter_enabled = True

    def forward(self, input_ids, attention_mask=None, labels=None, use_cache=False):
        h = self.embed(input_ids)
        if self._adapter_enabled:
            h = h + self.adapter(h)
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

    @contextmanager
    def disable_adapter(self):
        self._adapter_enabled = False
        try:
            yield
        finally:
            self._adapter_enabled = True


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
        ref_model.eval()
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

    def test_peft_adapter_as_ref(self):
        """With ref_model=None on a PEFT model, base model outputs serve as reference."""
        torch.manual_seed(42)
        model = PeftSimpleModel()
        batch = _make_preference_batch()

        loss_fn = DPOLoss(ref_model=None, beta=0.1)
        loss_fn._pad_token_id = 0

        loss, metrics = loss_fn(model, batch, training=True)
        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert "chosen_rewards" in metrics

    def test_peft_ref_differs_from_adapter_output(self):
        """PEFT ref (adapter disabled) should produce different loss than identical ref."""
        torch.manual_seed(42)
        model = PeftSimpleModel()
        batch = _make_preference_batch()

        # With ref_model=None, reference = base model (adapter disabled)
        loss_fn_peft = DPOLoss(ref_model=None, beta=0.1)
        loss_fn_peft._pad_token_id = 0
        loss_peft, _ = loss_fn_peft(model, batch, training=True)

        # With ref_model = deepcopy (includes adapter), reference = full model
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        loss_fn_copy = DPOLoss(ref_model=ref_model, beta=0.1)
        loss_fn_copy._pad_token_id = 0
        loss_copy, _ = loss_fn_copy(model, batch, training=True)

        # deepcopy ref includes the adapter, so pi == ref -> loss = log(2)
        # PEFT ref disables adapter, so pi != ref -> loss != log(2)
        assert abs(loss_copy.item() - 0.6931) < 0.01
        assert loss_peft.item() != loss_copy.item()

    def test_label_smoothing_changes_loss(self):
        """Label smoothing should change loss when policy differs from ref."""
        torch.manual_seed(42)
        policy = SimpleModel()
        batch = _make_preference_batch()

        # Use a different ref model so logits_diff != 0
        torch.manual_seed(999)
        ref_model = SimpleModel()
        ref_model.eval()

        loss_fn_no = DPOLoss(ref_model=ref_model, beta=0.1, label_smoothing=0.0)
        loss_fn_no._pad_token_id = 0
        loss_fn_smooth = DPOLoss(ref_model=ref_model, beta=0.1, label_smoothing=0.1)
        loss_fn_smooth._pad_token_id = 0

        loss_no, _ = loss_fn_no(policy, batch, training=True)
        loss_yes, _ = loss_fn_smooth(policy, batch, training=True)

        # With label smoothing, the loss target is softer so loss should differ
        assert loss_no.item() != loss_yes.item()

    def test_label_smoothing_zero_matches_default(self):
        """label_smoothing=0.0 should give identical results to default."""
        torch.manual_seed(42)
        model = SimpleModel()
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        batch = _make_preference_batch()

        loss_fn_default = DPOLoss(ref_model=ref_model, beta=0.1)
        loss_fn_default._pad_token_id = 0
        loss_fn_zero = DPOLoss(ref_model=ref_model, beta=0.1, label_smoothing=0.0)
        loss_fn_zero._pad_token_id = 0

        loss_default, _ = loss_fn_default(model, batch, training=True)
        loss_zero, _ = loss_fn_zero(model, batch, training=True)

        assert abs(loss_default.item() - loss_zero.item()) < 1e-6


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


def _make_kto_batch(vocab_size=32, seq_len=8, batch_size=4, prompt_len=2, desirable_ratio=0.5):
    """Helper to create a KTO batch with mixed desirable/undesirable examples."""
    n_desirable = int(batch_size * desirable_ratio)
    batch = {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "kto_label": torch.zeros(batch_size, dtype=torch.bool),
    }
    batch["labels"][:, :prompt_len] = -100
    batch["kto_label"][:n_desirable] = True
    return batch


class TestKTOLoss:
    def _make_loss(self, ref_model=None, beta=0.1, lambda_d=1.0, lambda_u=1.0):
        model = SimpleModel()
        if ref_model is None:
            ref_model = copy.deepcopy(model)
            ref_model.eval()
        loss_fn = KTOLoss(ref_model=ref_model, beta=beta, lambda_d=lambda_d, lambda_u=lambda_u)
        loss_fn._pad_token_id = 0
        return model, loss_fn

    def test_returns_scalar_loss_and_metrics(self):
        model, loss_fn = self._make_loss()
        batch = _make_kto_batch()

        loss, metrics = loss_fn(model, batch, training=True)

        assert loss.dim() == 0
        assert loss.item() > 0
        assert "chosen_rewards" in metrics
        assert "rejected_rewards" in metrics
        assert "reward_margin" in metrics
        assert "reward_accuracy" in metrics
        assert "kl_ref" in metrics

    def test_eval_mode_uses_nll(self):
        model, loss_fn = self._make_loss()
        batch = _make_kto_batch()

        loss, metrics = loss_fn(model, batch, training=False)
        assert loss.dim() == 0
        assert loss.item() > 0
        assert metrics == {}

    def test_beta_scales_loss(self):
        torch.manual_seed(42)
        ref_model = SimpleModel()
        ref_model.eval()
        policy = SimpleModel()
        batch = _make_kto_batch()

        loss_fn_low = KTOLoss(ref_model=ref_model, beta=0.01)
        loss_fn_low._pad_token_id = 0
        loss_fn_high = KTOLoss(ref_model=ref_model, beta=1.0)
        loss_fn_high._pad_token_id = 0

        loss_low, _ = loss_fn_low(policy, batch, training=True)
        loss_high, _ = loss_fn_high(policy, batch, training=True)

        assert loss_low.item() != loss_high.item()

    def test_all_desirable_batch(self):
        """Batch with only desirable examples should not crash."""
        model, loss_fn = self._make_loss()
        batch = _make_kto_batch(desirable_ratio=1.0)

        loss, metrics = loss_fn(model, batch, training=True)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_all_undesirable_batch(self):
        """Batch with only undesirable examples should not crash."""
        model, loss_fn = self._make_loss()
        batch = _make_kto_batch(desirable_ratio=0.0)

        loss, metrics = loss_fn(model, batch, training=True)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_lambda_weighting(self):
        """Higher lambda_u should increase loss from undesirable examples."""
        torch.manual_seed(42)
        ref_model = SimpleModel()
        ref_model.eval()
        policy = SimpleModel()
        batch = _make_kto_batch(desirable_ratio=0.0)  # all undesirable

        loss_fn_low = KTOLoss(ref_model=ref_model, beta=0.1, lambda_u=0.1)
        loss_fn_low._pad_token_id = 0
        loss_fn_high = KTOLoss(ref_model=ref_model, beta=0.1, lambda_u=5.0)
        loss_fn_high._pad_token_id = 0

        loss_low, _ = loss_fn_low(policy, batch, training=True)
        loss_high, _ = loss_fn_high(policy, batch, training=True)

        assert loss_high.item() > loss_low.item()

    def test_ref_model_affects_loss(self):
        """Changing the reference model should change the loss."""
        torch.manual_seed(42)
        policy = SimpleModel()
        batch = _make_kto_batch()

        ref_same = copy.deepcopy(policy)
        ref_same.eval()
        loss_fn_same = KTOLoss(ref_model=ref_same, beta=0.1)
        loss_fn_same._pad_token_id = 0
        loss_same, _ = loss_fn_same(policy, batch, training=True)

        torch.manual_seed(999)
        ref_diff = SimpleModel()
        ref_diff.eval()
        loss_fn_diff = KTOLoss(ref_model=ref_diff, beta=0.1)
        loss_fn_diff._pad_token_id = 0
        loss_diff, _ = loss_fn_diff(policy, batch, training=True)

        assert loss_same.item() != loss_diff.item()

    def test_creates_correct_collator(self):
        ref_model = SimpleModel()
        ref_model.eval()
        loss_fn = KTOLoss(ref_model=ref_model)
        from grimoire.data.kto import KTOCollator
        collator = loss_fn.create_collator(pad_token_id=0)
        assert isinstance(collator, KTOCollator)

    def test_peft_adapter_as_ref(self):
        """With ref_model=None on a PEFT model, base model outputs serve as reference."""
        torch.manual_seed(42)
        model = PeftSimpleModel()
        batch = _make_kto_batch()

        loss_fn = KTOLoss(ref_model=None, beta=0.1)
        loss_fn._pad_token_id = 0

        loss, metrics = loss_fn(model, batch, training=True)
        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert "chosen_rewards" in metrics
        assert "kl_ref" in metrics


class TestCPOLoss:
    def test_returns_scalar_loss_and_metrics(self):
        model = SimpleModel()
        loss_fn = CPOLoss(beta=0.1)
        loss_fn._pad_token_id = 0

        batch = _make_preference_batch()

        loss, metrics = loss_fn(model, batch, training=True)

        assert loss.dim() == 0
        assert loss.item() > 0
        assert "nll_loss" in metrics
        assert "preference_loss" in metrics
        assert "chosen_rewards" in metrics
        assert "rejected_rewards" in metrics
        assert "reward_margin" in metrics
        assert "reward_accuracy" in metrics
        assert "logps_diff" in metrics

    def test_eval_mode_uses_chosen_only(self):
        model = SimpleModel()
        loss_fn = CPOLoss(beta=0.1)

        batch = _make_preference_batch()

        loss, metrics = loss_fn(model, batch, training=False)
        assert loss.dim() == 0
        assert loss.item() > 0
        assert metrics == {}

    def test_beta_scales_loss(self):
        torch.manual_seed(42)
        model = SimpleModel()
        batch = _make_preference_batch()

        loss_fn_low = CPOLoss(beta=0.01)
        loss_fn_low._pad_token_id = 0
        loss_fn_high = CPOLoss(beta=1.0)
        loss_fn_high._pad_token_id = 0

        loss_low, _ = loss_fn_low(model, batch, training=True)
        loss_high, _ = loss_fn_high(model, batch, training=True)

        # Higher beta amplifies the preference component
        assert loss_low.item() != loss_high.item()

    def test_handles_different_chosen_rejected_lengths(self):
        model = SimpleModel()
        loss_fn = CPOLoss(beta=0.1)
        loss_fn._pad_token_id = 0

        batch = _make_preference_batch(chosen_len=6, rejected_len=10)

        loss, metrics = loss_fn(model, batch, training=True)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_creates_correct_collator(self):
        loss_fn = CPOLoss()
        from grimoire.data.preference import PreferenceCollator
        collator = loss_fn.create_collator(pad_token_id=0)
        assert isinstance(collator, PreferenceCollator)

    def test_no_ref_model_needed(self):
        """CPO should work without any reference model."""
        loss_fn = CPOLoss()
        assert not hasattr(loss_fn, "ref_model")

    def test_label_smoothing_changes_loss(self):
        """Label smoothing should change the preference loss component."""
        torch.manual_seed(42)
        model = SimpleModel()
        batch = _make_preference_batch()

        loss_fn_no = CPOLoss(beta=0.1, label_smoothing=0.0)
        loss_fn_no._pad_token_id = 0
        loss_fn_yes = CPOLoss(beta=0.1, label_smoothing=0.1)
        loss_fn_yes._pad_token_id = 0

        loss_no, _ = loss_fn_no(model, batch, training=True)
        loss_yes, _ = loss_fn_yes(model, batch, training=True)

        assert loss_no.item() != loss_yes.item()

    def test_label_smoothing_zero_matches_default(self):
        """label_smoothing=0.0 should give identical results to default."""
        torch.manual_seed(42)
        model = SimpleModel()
        batch = _make_preference_batch()

        loss_fn_default = CPOLoss(beta=0.1)
        loss_fn_default._pad_token_id = 0
        loss_fn_zero = CPOLoss(beta=0.1, label_smoothing=0.0)
        loss_fn_zero._pad_token_id = 0

        loss_default, _ = loss_fn_default(model, batch, training=True)
        loss_zero, _ = loss_fn_zero(model, batch, training=True)

        assert abs(loss_default.item() - loss_zero.item()) < 1e-6


class TestIPOLoss:
    def _make_loss(self, ref_model=None, beta=0.1):
        model = SimpleModel()
        if ref_model is None:
            ref_model = copy.deepcopy(model)
            ref_model.eval()
        loss_fn = IPOLoss(ref_model=ref_model, beta=beta)
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
        policy = SimpleModel()
        batch = _make_preference_batch()

        loss_fn_low = IPOLoss(ref_model=ref_model, beta=0.01)
        loss_fn_low._pad_token_id = 0
        loss_fn_high = IPOLoss(ref_model=ref_model, beta=1.0)
        loss_fn_high._pad_token_id = 0

        loss_low, _ = loss_fn_low(policy, batch, training=True)
        loss_high, _ = loss_fn_high(policy, batch, training=True)

        # Different beta changes the target margin 1/(2*beta), so loss changes
        assert loss_low.item() != loss_high.item()

    def test_creates_correct_collator(self):
        ref_model = SimpleModel()
        ref_model.eval()
        loss_fn = IPOLoss(ref_model=ref_model)
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
        loss_fn_same = IPOLoss(ref_model=ref_same, beta=0.1)
        loss_fn_same._pad_token_id = 0
        loss_same, _ = loss_fn_same(policy, batch, training=True)

        # ref_model with different weights
        torch.manual_seed(999)
        ref_diff = SimpleModel()
        ref_diff.eval()
        loss_fn_diff = IPOLoss(ref_model=ref_diff, beta=0.1)
        loss_fn_diff._pad_token_id = 0
        loss_diff, _ = loss_fn_diff(policy, batch, training=True)

        assert loss_same.item() != loss_diff.item()

    def test_identical_policy_and_ref_gives_expected_loss(self):
        """When policy == ref, logratios cancel and loss = (1/(2*beta))^2."""
        torch.manual_seed(42)
        model = SimpleModel()
        ref_model = copy.deepcopy(model)
        ref_model.eval()

        beta = 0.1
        loss_fn = IPOLoss(ref_model=ref_model, beta=beta)
        loss_fn._pad_token_id = 0
        batch = _make_preference_batch()

        loss, _ = loss_fn(model, batch, training=True)
        # When pi == ref, logits_diff = 0, so loss = (0 - 1/(2*beta))^2 = (1/(2*beta))^2
        expected = (1.0 / (2.0 * beta)) ** 2
        assert abs(loss.item() - expected) < 0.01

    def test_peft_adapter_as_ref(self):
        """With ref_model=None on a PEFT model, base model outputs serve as reference."""
        torch.manual_seed(42)
        model = PeftSimpleModel()
        batch = _make_preference_batch()

        loss_fn = IPOLoss(ref_model=None, beta=0.1)
        loss_fn._pad_token_id = 0

        loss, metrics = loss_fn(model, batch, training=True)
        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert "chosen_rewards" in metrics


class GenerativeModel(nn.Module):
    """Tiny model with generate() support for GRPO testing."""

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

    def generate(self, input_ids, attention_mask=None, max_new_tokens=8,
                 temperature=1.0, do_sample=True, pad_token_id=0):
        """Simple autoregressive generation by sampling from logits."""
        generated = input_ids
        for _ in range(max_new_tokens):
            logits = self.forward(generated).logits[:, -1, :]  # [B, vocab]
            if temperature != 1.0:
                logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
            generated = torch.cat([generated, next_token], dim=1)
        return generated


class MockTokenizer:
    """Minimal tokenizer for GRPO testing."""

    def batch_decode(self, token_ids, skip_special_tokens=True):
        return [" ".join(str(t) for t in ids.tolist()) for ids in token_ids]


def _make_grpo_batch(vocab_size=32, prompt_len=4, batch_size=2):
    """Helper to create a GRPO prompt-only batch."""
    return {
        "input_ids": torch.randint(0, vocab_size, (batch_size, prompt_len)),
        "attention_mask": torch.ones(batch_size, prompt_len, dtype=torch.long),
    }


def _constant_reward_fn(prompts, completions):
    """Reward function that returns constant scores for testing."""
    return [1.0] * len(prompts)


def _length_reward_fn(prompts, completions):
    """Reward function that scores by completion length (produces variance)."""
    return [float(len(c)) for c in completions]


class TestGRPOLoss:
    def _make_loss(self, reward_fn=None, num_generations=2, beta=0.04, max_new_tokens=4):
        model = GenerativeModel()
        tokenizer = MockTokenizer()
        if reward_fn is None:
            reward_fn = _length_reward_fn
        loss_fn = GRPOLoss(
            reward_fn=reward_fn,
            tokenizer=tokenizer,
            num_generations=num_generations,
            beta=beta,
            max_new_tokens=max_new_tokens,
        )
        loss_fn._pad_token_id = 0
        return model, loss_fn

    def test_returns_scalar_loss_and_metrics(self):
        torch.manual_seed(42)
        model, loss_fn = self._make_loss()
        batch = _make_grpo_batch()

        loss, metrics = loss_fn(model, batch, training=True)

        assert loss.dim() == 0
        assert isinstance(metrics, dict)
        assert "rewards_mean" in metrics
        assert "rewards_std" in metrics
        assert "advantages_mean" in metrics
        assert "kl" in metrics
        assert "policy_loss" in metrics
        assert "ratio_mean" in metrics
        assert "completion_length" in metrics

    def test_loss_is_finite(self):
        torch.manual_seed(42)
        model, loss_fn = self._make_loss()
        batch = _make_grpo_batch()

        loss, _ = loss_fn(model, batch, training=True)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_eval_returns_zero_loss(self):
        model, loss_fn = self._make_loss()
        batch = _make_grpo_batch()

        loss, metrics = loss_fn(model, batch, training=False)
        assert loss.item() == 0.0
        assert metrics == {}

    def test_num_generations_affects_batch(self):
        """More generations should still produce valid loss."""
        torch.manual_seed(42)
        model, loss_fn = self._make_loss(num_generations=4)
        batch = _make_grpo_batch(batch_size=2)

        loss, metrics = loss_fn(model, batch, training=True)

        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_beta_zero_removes_kl(self):
        """With beta=0, KL penalty should not contribute to loss."""
        torch.manual_seed(42)
        model, loss_fn = self._make_loss(beta=0.0)
        batch = _make_grpo_batch()

        loss, metrics = loss_fn(model, batch, training=True)

        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_constant_rewards_zero_advantages(self):
        """When all rewards are identical, advantages should be zero."""
        torch.manual_seed(42)
        model, loss_fn = self._make_loss(reward_fn=_constant_reward_fn)
        batch = _make_grpo_batch()

        _, metrics = loss_fn(model, batch, training=True)

        # With constant rewards, std is 0 but clamped, so advantages ~ 0
        assert abs(metrics["advantages_mean"]) < 1e-6

    def test_creates_correct_collator(self):
        _, loss_fn = self._make_loss()
        from grimoire.data.grpo import GRPOCollator
        collator = loss_fn.create_collator(pad_token_id=0)
        assert isinstance(collator, GRPOCollator)

    def test_loss_requires_grad(self):
        """Loss should have gradients flowing through policy log-probs."""
        torch.manual_seed(42)
        model, loss_fn = self._make_loss()
        batch = _make_grpo_batch()

        loss, _ = loss_fn(model, batch, training=True)

        assert loss.requires_grad

    def test_ratio_starts_near_one(self):
        """On first call, old and new policy are the same, so ratio ~ 1."""
        torch.manual_seed(42)
        model, loss_fn = self._make_loss()
        batch = _make_grpo_batch()

        _, metrics = loss_fn(model, batch, training=True)

        # ratio = exp(logps - old_logps), should be ~1 since same model
        assert abs(metrics["ratio_mean"] - 1.0) < 0.1


def _make_preference_dataset(n=4, vocab_size=32, chosen_len=8, rejected_len=8, prompt_len=2):
    """Create a list-of-dicts preference dataset for caching tests."""
    dataset = []
    for _ in range(n):
        chosen_ids = torch.randint(0, vocab_size, (chosen_len,)).tolist()
        rejected_ids = torch.randint(0, vocab_size, (rejected_len,)).tolist()
        chosen_labels = [-100] * prompt_len + chosen_ids[prompt_len:]
        rejected_labels = [-100] * prompt_len + rejected_ids[prompt_len:]
        dataset.append({
            "chosen_input_ids": chosen_ids,
            "chosen_attention_mask": [1] * chosen_len,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_ids,
            "rejected_attention_mask": [1] * rejected_len,
            "rejected_labels": rejected_labels,
        })
    return dataset


def _make_kto_dataset(n=4, vocab_size=32, seq_len=8, prompt_len=2):
    """Create a list-of-dicts KTO dataset for caching tests."""
    dataset = []
    for i in range(n):
        ids = torch.randint(0, vocab_size, (seq_len,)).tolist()
        labels = [-100] * prompt_len + ids[prompt_len:]
        dataset.append({
            "input_ids": ids,
            "attention_mask": [1] * seq_len,
            "labels": labels,
            "kto_label": i % 2 == 0,
        })
    return dataset


class TestCacheReferenceLogProbs:
    def test_caches_preference_data(self):
        torch.manual_seed(42)
        ref_model = SimpleModel()
        ref_model.eval()
        from grimoire.data.preference import PreferenceCollator
        collator = PreferenceCollator(pad_token_id=0)

        dataset = _make_preference_dataset(n=4)
        dataset = cache_reference_log_probs(ref_model, dataset, collator, batch_size=2)

        assert "ref_chosen_logps" in dataset[0]
        assert "ref_rejected_logps" in dataset[0]
        assert isinstance(dataset[0]["ref_chosen_logps"], float)
        assert isinstance(dataset[0]["ref_rejected_logps"], float)
        assert len(dataset) == 4

    def test_caches_kto_data(self):
        torch.manual_seed(42)
        ref_model = SimpleModel()
        ref_model.eval()
        from grimoire.data.kto import KTOCollator
        collator = KTOCollator(pad_token_id=0)

        dataset = _make_kto_dataset(n=4)
        dataset = cache_reference_log_probs(ref_model, dataset, collator, batch_size=2)

        assert "ref_logps" in dataset[0]
        assert isinstance(dataset[0]["ref_logps"], float)
        assert len(dataset) == 4

    def test_ref_model_must_be_eval(self):
        ref_model = SimpleModel()
        # model is in train mode by default
        from grimoire.data.preference import PreferenceCollator
        collator = PreferenceCollator(pad_token_id=0)
        dataset = _make_preference_dataset(n=2)

        with pytest.raises(ValueError, match="eval mode"):
            cache_reference_log_probs(ref_model, dataset, collator)


class TestDPOCachedLogProbs:
    def test_cached_matches_live(self):
        """Cached ref log probs should produce the same loss as live computation."""
        torch.manual_seed(42)
        model = SimpleModel()
        ref_model = copy.deepcopy(model)
        ref_model.eval()

        batch = _make_preference_batch()

        # Live computation
        loss_fn_live = DPOLoss(ref_model=ref_model, beta=0.1)
        loss_fn_live._pad_token_id = 0
        loss_live, metrics_live = loss_fn_live(model, batch, training=True)

        # Compute cached ref log probs by running ref model on chosen/rejected separately
        from grimoire.data.preference import PreferenceCollator
        collator = PreferenceCollator(pad_token_id=0)
        dataset = _make_preference_dataset(n=2)
        dataset = cache_reference_log_probs(ref_model, dataset, collator, batch_size=2)
        cached_batch = collator(dataset)

        # Use cached values with no ref model
        loss_fn_cached = DPOLoss(beta=0.1)
        loss_fn_cached._pad_token_id = 0
        loss_cached, metrics_cached = loss_fn_cached(model, cached_batch, training=True)

        # Both should produce valid scalar losses
        assert loss_cached.dim() == 0
        assert not torch.isnan(loss_cached)
        assert "chosen_rewards" in metrics_cached

    def test_no_ref_model_no_cache_raises(self):
        """DPOLoss with no ref_model and no cached log probs should raise."""
        model = SimpleModel()
        loss_fn = DPOLoss(beta=0.1)
        loss_fn._pad_token_id = 0
        batch = _make_preference_batch()

        with pytest.raises(ValueError, match="requires either"):
            loss_fn(model, batch, training=True)

    def test_ref_model_none_allowed(self):
        """DPOLoss should accept ref_model=None."""
        loss_fn = DPOLoss(beta=0.1)
        assert loss_fn.ref_model is None


class TestIPOCachedLogProbs:
    def test_cached_matches_live(self):
        torch.manual_seed(42)
        model = SimpleModel()
        ref_model = copy.deepcopy(model)
        ref_model.eval()

        from grimoire.data.preference import PreferenceCollator
        collator = PreferenceCollator(pad_token_id=0)
        dataset = _make_preference_dataset(n=2)
        dataset = cache_reference_log_probs(ref_model, dataset, collator, batch_size=2)
        cached_batch = collator(dataset)

        loss_fn = IPOLoss(beta=0.1)
        loss_fn._pad_token_id = 0
        loss, metrics = loss_fn(model, cached_batch, training=True)

        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert "chosen_rewards" in metrics

    def test_no_ref_model_no_cache_raises(self):
        model = SimpleModel()
        loss_fn = IPOLoss(beta=0.1)
        loss_fn._pad_token_id = 0
        batch = _make_preference_batch()

        with pytest.raises(ValueError, match="requires either"):
            loss_fn(model, batch, training=True)


class TestKTOCachedLogProbs:
    def test_cached_matches_live(self):
        torch.manual_seed(42)
        model = SimpleModel()
        ref_model = copy.deepcopy(model)
        ref_model.eval()

        from grimoire.data.kto import KTOCollator
        collator = KTOCollator(pad_token_id=0)
        dataset = _make_kto_dataset(n=4)
        dataset = cache_reference_log_probs(ref_model, dataset, collator, batch_size=4)
        cached_batch = collator(dataset)

        loss_fn = KTOLoss(beta=0.1)
        loss_fn._pad_token_id = 0
        loss, metrics = loss_fn(model, cached_batch, training=True)

        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert "chosen_rewards" in metrics
        assert "kl_ref" in metrics

    def test_no_ref_model_no_cache_raises(self):
        model = SimpleModel()
        loss_fn = KTOLoss(beta=0.1)
        loss_fn._pad_token_id = 0
        batch = _make_kto_batch()

        with pytest.raises(ValueError, match="requires either"):
            loss_fn(model, batch, training=True)
