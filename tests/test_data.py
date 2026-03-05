"""Tests for data collators and tokenization utilities."""

import torch
import pytest
from grimoire.data.sft import SFTCollator, tokenize_sft
from grimoire.data.preference import PreferenceCollator, tokenize_preference


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
