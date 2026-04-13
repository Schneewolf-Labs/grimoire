"""Tests for data collators and tokenization utilities."""

import pytest
from grimoire.data.sft import SFTCollator, PackedSFTCollator, tokenize_sft
from grimoire.data.preference import PreferenceCollator, tokenize_preference
from grimoire.data.kto import KTOCollator, tokenize_kto
from grimoire.data.grpo import GRPOCollator, tokenize_grpo


class TestSFTCollator:
    def test_pads_to_max_length(self):
        collator = SFTCollator(pad_token_id=0)
        features = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]},
            {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [-100, 5]},
        ]
        batch = collator(features)

        assert batch["input_ids"].shape == (2, 3)
        assert batch["attention_mask"].shape == (2, 3)
        assert batch["labels"].shape == (2, 3)

        # Second sequence should be padded
        assert batch["input_ids"][1].tolist() == [4, 5, 0]
        assert batch["attention_mask"][1].tolist() == [1, 1, 0]
        assert batch["labels"][1].tolist() == [-100, 5, -100]

    def test_single_element_batch(self):
        collator = SFTCollator(pad_token_id=0)
        features = [{"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]}]
        batch = collator(features)

        assert batch["input_ids"].shape == (1, 2)
        assert batch["input_ids"][0].tolist() == [1, 2]

    def test_equal_lengths_no_padding(self):
        collator = SFTCollator(pad_token_id=0)
        features = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]},
            {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1], "labels": [4, 5, 6]},
        ]
        batch = collator(features)

        assert batch["input_ids"].shape == (2, 3)
        # No padding needed
        assert (batch["attention_mask"] == 1).all()


class TestPackedSFTCollator:
    def test_packs_multiple_sequences(self):
        collator = PackedSFTCollator(pad_token_id=0, max_length=10)
        features = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]},
            {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [-100, 5]},
            {"input_ids": [6, 7, 8], "attention_mask": [1, 1, 1], "labels": [-100, 7, 8]},
        ]
        batch = collator(features)

        # All 3 sequences (total 8 tokens) fit in one row of max_length=10
        assert batch["input_ids"].shape[0] <= 2  # packed into fewer rows
        assert "position_ids" in batch
        assert batch["input_ids"].shape == batch["position_ids"].shape

    def test_position_ids_reset_at_boundaries(self):
        collator = PackedSFTCollator(pad_token_id=0, max_length=10)
        features = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]},
            {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [4, 5]},
        ]
        batch = collator(features)

        # Both sequences packed into one row
        assert batch["input_ids"].shape[0] == 1
        # Position IDs: [0,1,2] for first seq, [0,1] for second
        pos = batch["position_ids"][0].tolist()
        assert pos[:3] == [0, 1, 2]
        assert pos[3:5] == [0, 1]

    def test_overflow_creates_new_bin(self):
        collator = PackedSFTCollator(pad_token_id=0, max_length=4)
        features = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]},
            {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1], "labels": [4, 5, 6]},
        ]
        batch = collator(features)

        # 3+3=6 > max_length=4, so two rows
        assert batch["input_ids"].shape[0] == 2

    def test_truncates_long_sequences(self):
        collator = PackedSFTCollator(pad_token_id=0, max_length=3)
        features = [
            {"input_ids": [1, 2, 3, 4, 5], "attention_mask": [1, 1, 1, 1, 1], "labels": [1, 2, 3, 4, 5]},
        ]
        batch = collator(features)

        assert batch["input_ids"].shape == (1, 3)

    def test_preserves_labels(self):
        collator = PackedSFTCollator(pad_token_id=0, max_length=10)
        features = [
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
            {"input_ids": [3, 4], "attention_mask": [1, 1], "labels": [-100, 4]},
        ]
        batch = collator(features)

        labels = batch["labels"][0].tolist()
        # First seq labels: [-100, 2], second: [-100, 4]
        assert labels[0] == -100
        assert labels[1] == 2
        assert labels[2] == -100
        assert labels[3] == 4

    def test_padding_uses_correct_values(self):
        collator = PackedSFTCollator(pad_token_id=99, label_pad_token_id=-100, max_length=10)
        features = [
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [1, 2]},
            {"input_ids": [3, 4, 5], "attention_mask": [1, 1, 1], "labels": [3, 4, 5]},
        ]
        batch = collator(features)

        row = batch["input_ids"][0]
        # Padding region should use pad_token_id=99
        packed_len = 5  # 2 + 3
        if row.shape[0] > packed_len:
            assert (row[packed_len:] == 99).all()
            assert (batch["labels"][0, packed_len:] == -100).all()
            assert (batch["attention_mask"][0, packed_len:] == 0).all()

    def test_single_sequence(self):
        collator = PackedSFTCollator(pad_token_id=0, max_length=10)
        features = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]},
        ]
        batch = collator(features)

        assert batch["input_ids"].shape == (1, 3)
        assert batch["position_ids"][0].tolist() == [0, 1, 2]


class TestPreferenceCollator:
    def test_pads_chosen_and_rejected_independently(self):
        collator = PreferenceCollator(pad_token_id=0)
        features = [
            {
                "chosen_input_ids": [1, 2, 3], "chosen_attention_mask": [1, 1, 1], "chosen_labels": [-100, 2, 3],
                "rejected_input_ids": [1, 4], "rejected_attention_mask": [1, 1], "rejected_labels": [-100, 4],
            },
            {
                "chosen_input_ids": [5, 6], "chosen_attention_mask": [1, 1], "chosen_labels": [-100, 6],
                "rejected_input_ids": [5, 7, 8, 9], "rejected_attention_mask": [1, 1, 1, 1], "rejected_labels": [-100, 7, 8, 9],
            },
        ]
        batch = collator(features)

        # Chosen padded to max_chosen=3, rejected padded to max_rejected=4
        assert batch["chosen_input_ids"].shape == (2, 3)
        assert batch["rejected_input_ids"].shape == (2, 4)

        # Check padding values
        assert batch["chosen_input_ids"][1].tolist() == [5, 6, 0]
        assert batch["rejected_input_ids"][0].tolist() == [1, 4, 0, 0]
        assert batch["rejected_labels"][0].tolist() == [-100, 4, -100, -100]


class TestTokenizeSFT:
    @pytest.fixture
    def mock_tokenizer(self):
        class MockTokenizer:
            def __call__(self, text, max_length=None, truncation=False, add_special_tokens=True):
                # Simple char-level tokenization for testing
                ids = [ord(c) for c in text[:max_length]] if max_length else [ord(c) for c in text]
                return {"input_ids": ids, "attention_mask": [1] * len(ids)}
        return MockTokenizer()

    def test_text_field_mode(self, mock_tokenizer):
        example = {"text": "hello"}
        result = tokenize_sft(example, mock_tokenizer, text_field="text")

        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" in result
        # In text mode, all tokens are in the loss
        assert result["labels"] == result["input_ids"]

    def test_prompt_response_mode(self, mock_tokenizer):
        example = {"prompt": "hi", "response": "there"}
        result = tokenize_sft(example, mock_tokenizer, prompt_field="prompt", response_field="response")

        assert result["labels"][:2] == [-100, -100]  # prompt masked
        assert result["labels"][2:] == [ord(c) for c in "there"]  # response kept

    def test_raises_without_fields(self, mock_tokenizer):
        with pytest.raises(ValueError):
            tokenize_sft({"text": "hello"}, mock_tokenizer)


class TestTokenizePreference:
    @pytest.fixture
    def mock_tokenizer(self):
        class MockTokenizer:
            def __call__(self, text, max_length=None, truncation=False, add_special_tokens=True):
                ids = [ord(c) for c in text[:max_length]] if max_length else [ord(c) for c in text]
                return {"input_ids": ids, "attention_mask": [1] * len(ids)}
        return MockTokenizer()

    def test_produces_chosen_and_rejected(self, mock_tokenizer):
        example = {"prompt": "q:", "chosen": "good", "rejected": "bad"}
        result = tokenize_preference(example, mock_tokenizer)

        assert "chosen_input_ids" in result
        assert "chosen_labels" in result
        assert "rejected_input_ids" in result
        assert "rejected_labels" in result

    def test_prompt_masked_in_labels(self, mock_tokenizer):
        example = {"prompt": "AB", "chosen": "CD", "rejected": "EF"}
        result = tokenize_preference(example, mock_tokenizer)

        # First 2 tokens (prompt) should be masked
        assert result["chosen_labels"][:2] == [-100, -100]
        assert result["rejected_labels"][:2] == [-100, -100]
        # Response tokens should be kept
        assert result["chosen_labels"][2:] == [ord("C"), ord("D")]
        assert result["rejected_labels"][2:] == [ord("E"), ord("F")]


class TestKTOCollator:
    def test_pads_to_max_length_and_preserves_label(self):
        collator = KTOCollator(pad_token_id=0)
        features = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3], "kto_label": True},
            {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [-100, 5], "kto_label": False},
        ]
        batch = collator(features)

        assert batch["input_ids"].shape == (2, 3)
        assert batch["attention_mask"].shape == (2, 3)
        assert batch["labels"].shape == (2, 3)
        assert batch["kto_label"].tolist() == [True, False]

        # Second sequence should be padded
        assert batch["input_ids"][1].tolist() == [4, 5, 0]
        assert batch["attention_mask"][1].tolist() == [1, 1, 0]
        assert batch["labels"][1].tolist() == [-100, 5, -100]

    def test_single_element_batch(self):
        collator = KTOCollator(pad_token_id=0)
        features = [{"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2], "kto_label": True}]
        batch = collator(features)

        assert batch["input_ids"].shape == (1, 2)
        assert batch["kto_label"].tolist() == [True]

    def test_equal_lengths_no_padding(self):
        collator = KTOCollator(pad_token_id=0)
        features = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3], "kto_label": True},
            {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1], "labels": [4, 5, 6], "kto_label": False},
        ]
        batch = collator(features)

        assert batch["input_ids"].shape == (2, 3)
        assert (batch["attention_mask"] == 1).all()


class TestPreferenceCollatorCachedRefLogps:
    def test_passes_through_cached_ref_logps(self):
        collator = PreferenceCollator(pad_token_id=0)
        features = [
            {
                "chosen_input_ids": [1, 2, 3], "chosen_attention_mask": [1, 1, 1], "chosen_labels": [-100, 2, 3],
                "rejected_input_ids": [1, 4], "rejected_attention_mask": [1, 1], "rejected_labels": [-100, 4],
                "ref_chosen_logps": -1.5, "ref_rejected_logps": -2.0,
            },
            {
                "chosen_input_ids": [5, 6], "chosen_attention_mask": [1, 1], "chosen_labels": [-100, 6],
                "rejected_input_ids": [5, 7, 8], "rejected_attention_mask": [1, 1, 1], "rejected_labels": [-100, 7, 8],
                "ref_chosen_logps": -1.2, "ref_rejected_logps": -2.5,
            },
        ]
        batch = collator(features)

        assert "ref_chosen_logps" in batch
        assert "ref_rejected_logps" in batch
        assert batch["ref_chosen_logps"].tolist() == pytest.approx([-1.5, -1.2])
        assert batch["ref_rejected_logps"].tolist() == pytest.approx([-2.0, -2.5])

    def test_no_ref_logps_when_not_present(self):
        collator = PreferenceCollator(pad_token_id=0)
        features = [
            {
                "chosen_input_ids": [1, 2], "chosen_attention_mask": [1, 1], "chosen_labels": [-100, 2],
                "rejected_input_ids": [1, 3], "rejected_attention_mask": [1, 1], "rejected_labels": [-100, 3],
            },
        ]
        batch = collator(features)
        assert "ref_chosen_logps" not in batch
        assert "ref_rejected_logps" not in batch


class TestKTOCollatorCachedRefLogps:
    def test_passes_through_cached_ref_logps(self):
        collator = KTOCollator(pad_token_id=0)
        features = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3], "kto_label": True, "ref_logps": -1.5},
            {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [-100, 5], "kto_label": False, "ref_logps": -2.0},
        ]
        batch = collator(features)

        assert "ref_logps" in batch
        assert batch["ref_logps"].tolist() == pytest.approx([-1.5, -2.0])

    def test_no_ref_logps_when_not_present(self):
        collator = KTOCollator(pad_token_id=0)
        features = [
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2], "kto_label": True},
        ]
        batch = collator(features)
        assert "ref_logps" not in batch


class TestTokenizeKTO:
    @pytest.fixture
    def mock_tokenizer(self):
        class MockTokenizer:
            def __call__(self, text, max_length=None, truncation=False, add_special_tokens=True):
                ids = [ord(c) for c in text[:max_length]] if max_length else [ord(c) for c in text]
                return {"input_ids": ids, "attention_mask": [1] * len(ids)}
        return MockTokenizer()

    def test_produces_correct_keys(self, mock_tokenizer):
        example = {"prompt": "AB", "response": "CD", "label": True}
        result = tokenize_kto(example, mock_tokenizer)

        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" in result
        assert "kto_label" in result

    def test_prompt_masked_in_labels(self, mock_tokenizer):
        example = {"prompt": "AB", "response": "CD", "label": True}
        result = tokenize_kto(example, mock_tokenizer)

        assert result["labels"][:2] == [-100, -100]
        assert result["labels"][2:] == [ord("C"), ord("D")]

    def test_preserves_label(self, mock_tokenizer):
        desirable = {"prompt": "Q", "response": "A", "label": True}
        undesirable = {"prompt": "Q", "response": "A", "label": False}

        assert tokenize_kto(desirable, mock_tokenizer)["kto_label"] is True
        assert tokenize_kto(undesirable, mock_tokenizer)["kto_label"] is False


class TestGRPOCollator:
    def test_pads_to_max_length(self):
        collator = GRPOCollator(pad_token_id=0)
        features = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
            {"input_ids": [4, 5], "attention_mask": [1, 1]},
        ]
        batch = collator(features)

        assert batch["input_ids"].shape == (2, 3)
        assert batch["attention_mask"].shape == (2, 3)

        # Second sequence should be padded
        assert batch["input_ids"][1].tolist() == [4, 5, 0]
        assert batch["attention_mask"][1].tolist() == [1, 1, 0]

    def test_no_labels_in_output(self):
        """GRPO collator should not produce labels (prompt-only)."""
        collator = GRPOCollator(pad_token_id=0)
        features = [{"input_ids": [1, 2], "attention_mask": [1, 1]}]
        batch = collator(features)

        assert "labels" not in batch

    def test_single_element_batch(self):
        collator = GRPOCollator(pad_token_id=0)
        features = [{"input_ids": [1, 2], "attention_mask": [1, 1]}]
        batch = collator(features)

        assert batch["input_ids"].shape == (1, 2)
        assert batch["input_ids"][0].tolist() == [1, 2]

    def test_equal_lengths_no_padding(self):
        collator = GRPOCollator(pad_token_id=0)
        features = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
            {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1]},
        ]
        batch = collator(features)

        assert batch["input_ids"].shape == (2, 3)
        assert (batch["attention_mask"] == 1).all()


class TestTokenizeSFTTruncation:
    @pytest.fixture
    def mock_tokenizer(self):
        class MockTokenizer:
            def __call__(self, text, max_length=None, truncation=False, add_special_tokens=True):
                ids = [ord(c) for c in text[:max_length]] if max_length else [ord(c) for c in text]
                return {"input_ids": ids, "attention_mask": [1] * len(ids)}
        return MockTokenizer()

    def test_text_field_truncates_to_max_length(self, mock_tokenizer):
        example = {"text": "ABCDEFGHIJ"}  # 10 chars
        result = tokenize_sft(example, mock_tokenizer, max_length=5, text_field="text")

        assert len(result["input_ids"]) == 5
        assert len(result["labels"]) == 5

    def test_prompt_response_truncates_combined(self, mock_tokenizer):
        # prompt=5 chars, response=5 chars, max_length=7 → truncated to 7 total
        example = {"prompt": "ABCDE", "response": "FGHIJ"}
        result = tokenize_sft(
            example, mock_tokenizer, max_length=7, prompt_field="prompt", response_field="response"
        )

        assert len(result["input_ids"]) == 7
        assert len(result["labels"]) == 7

    def test_only_prompt_field_raises_without_response(self, mock_tokenizer):
        with pytest.raises(ValueError):
            tokenize_sft({"prompt": "hello"}, mock_tokenizer, prompt_field="prompt")


class TestTokenizePreferenceCustomFields:
    @pytest.fixture
    def mock_tokenizer(self):
        class MockTokenizer:
            def __call__(self, text, max_length=None, truncation=False, add_special_tokens=True):
                ids = [ord(c) for c in text[:max_length]] if max_length else [ord(c) for c in text]
                return {"input_ids": ids, "attention_mask": [1] * len(ids)}
        return MockTokenizer()

    def test_custom_field_names(self, mock_tokenizer):
        example = {"q": "AB", "win": "CD", "lose": "EF"}
        result = tokenize_preference(
            example, mock_tokenizer,
            prompt_field="q", chosen_field="win", rejected_field="lose"
        )

        assert "chosen_input_ids" in result
        assert "rejected_input_ids" in result
        # prompt "AB" = 2 tokens → first 2 labels masked
        assert result["chosen_labels"][:2] == [-100, -100]
        assert result["rejected_labels"][:2] == [-100, -100]

    def test_max_prompt_length_limits_masking(self, mock_tokenizer):
        # prompt = "ABCD" (4 chars), max_prompt_length=2 → only 2 tokens masked
        example = {"prompt": "ABCD", "chosen": "EF", "rejected": "GH"}
        result = tokenize_preference(
            example, mock_tokenizer, max_prompt_length=2
        )

        # Only first 2 tokens should be masked (not all 4 prompt tokens)
        assert result["chosen_labels"][:2] == [-100, -100]
        assert result["chosen_labels"][2] != -100

    def test_truncates_to_max_length(self, mock_tokenizer):
        example = {"prompt": "AB", "chosen": "CDEFGHIJ", "rejected": "KLMNOPQR"}
        result = tokenize_preference(example, mock_tokenizer, max_length=5)

        assert len(result["chosen_input_ids"]) == 5
        assert len(result["rejected_input_ids"]) == 5


class TestTokenizeKTOCustomFields:
    @pytest.fixture
    def mock_tokenizer(self):
        class MockTokenizer:
            def __call__(self, text, max_length=None, truncation=False, add_special_tokens=True):
                ids = [ord(c) for c in text[:max_length]] if max_length else [ord(c) for c in text]
                return {"input_ids": ids, "attention_mask": [1] * len(ids)}
        return MockTokenizer()

    def test_custom_field_names(self, mock_tokenizer):
        example = {"question": "AB", "answer": "CD", "good": True}
        result = tokenize_kto(
            example, mock_tokenizer,
            prompt_field="question", response_field="answer", label_field="good"
        )

        assert "input_ids" in result
        assert "kto_label" in result
        assert result["kto_label"] is True

    def test_truncates_to_max_length(self, mock_tokenizer):
        example = {"prompt": "AB", "response": "CDEFGHIJ", "label": False}
        result = tokenize_kto(example, mock_tokenizer, max_length=5)

        assert len(result["input_ids"]) == 5
        assert len(result["labels"]) == 5

    def test_label_converts_to_bool(self, mock_tokenizer):
        # label_field value gets cast to bool
        example = {"prompt": "A", "response": "B", "label": 1}
        result = tokenize_kto(example, mock_tokenizer)
        assert result["kto_label"] is True

        example2 = {"prompt": "A", "response": "B", "label": 0}
        result2 = tokenize_kto(example2, mock_tokenizer)
        assert result2["kto_label"] is False


class TestTokenizeGRPO:
    @pytest.fixture
    def mock_tokenizer(self):
        class MockTokenizer:
            def __call__(self, text, max_length=None, truncation=False, add_special_tokens=True):
                ids = [ord(c) for c in text[:max_length]] if max_length else [ord(c) for c in text]
                return {"input_ids": ids, "attention_mask": [1] * len(ids)}
        return MockTokenizer()

    def test_produces_correct_keys(self, mock_tokenizer):
        example = {"prompt": "hello"}
        result = tokenize_grpo(example, mock_tokenizer)

        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" not in result

    def test_tokenizes_prompt_only(self, mock_tokenizer):
        example = {"prompt": "AB"}
        result = tokenize_grpo(example, mock_tokenizer)

        assert result["input_ids"] == [ord("A"), ord("B")]
        assert result["attention_mask"] == [1, 1]

    def test_respects_max_prompt_length(self, mock_tokenizer):
        example = {"prompt": "ABCDEF"}
        result = tokenize_grpo(example, mock_tokenizer, max_prompt_length=3)

        assert len(result["input_ids"]) == 3

    def test_custom_prompt_field(self, mock_tokenizer):
        example = {"question": "AB"}
        result = tokenize_grpo(example, mock_tokenizer, prompt_field="question")

        assert result["input_ids"] == [ord("A"), ord("B")]
