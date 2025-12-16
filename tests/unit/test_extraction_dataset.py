"""Unit tests for extraction dataset creation and BIO label generation."""

import json
import tempfile
from pathlib import Path

import pytest

from perscit_model.extraction.data_loader import (
    BIO_LABELS,
    LABEL2ID,
    ExtractionDataLoader,
    create_extraction_dataset,
)


class TestGenerateBioLabels:
    """Test suite for generate_bio_labels function."""

    @pytest.fixture
    def loader_with_special_tokens(self, mocker, mock_tokenizer):
        """Create an ExtractionDataLoader with mocked tokenizer."""
        loader = ExtractionDataLoader()

        # Replace the tokenizer with a mock by patching the underlying attribute
        mock_tok = mocker.Mock()

        # Mock token IDs
        mock_tok.cls_token_id = 0
        mock_tok.sep_token_id = 2
        mock_tok.pad_token_id = 1

        # Mock special token conversions
        special_token_map = {
            "[BIBL_START]": 100,
            "[BIBL_END]": 101,
            "[QUOTE_START]": 102,
            "[QUOTE_END]": 103,
            "[CIT_START]": 104,
            "[CIT_END]": 105,
        }

        def convert_tokens_to_ids(token):
            return special_token_map.get(token, -1)

        mock_tok.convert_tokens_to_ids = convert_tokens_to_ids
        loader._tokenizer = mock_tok

        return loader

    def test_simple_bibl_tag(self, loader_with_special_tokens):
        """Test basic BIBL tag generates correct BIO labels."""
        # [CLS] [BIBL_START] word1 word2 [BIBL_END] [SEP]
        input_ids = [0, 100, 10, 11, 101, 2]

        labels = loader_with_special_tokens.generate_bio_labels(input_ids)

        assert labels == [
            -100,              # CLS
            -100,              # BIBL_START
            LABEL2ID["B-BIBL"],  # First token
            LABEL2ID["I-BIBL"],  # Second token
            -100,              # BIBL_END
            -100,              # SEP
        ]

    def test_simple_quote_tag(self, loader_with_special_tokens):
        """Test basic QUOTE tag generates correct BIO labels."""
        # [CLS] [QUOTE_START] word1 word2 word3 [QUOTE_END] [SEP]
        input_ids = [0, 102, 10, 11, 12, 103, 2]

        labels = loader_with_special_tokens.generate_bio_labels(input_ids)

        assert labels == [
            -100,               # CLS
            -100,               # QUOTE_START
            LABEL2ID["B-QUOTE"],  # First token
            LABEL2ID["I-QUOTE"],  # Second token
            LABEL2ID["I-QUOTE"],  # Third token
            -100,               # QUOTE_END
            -100,               # SEP
        ]

    def test_simple_cit_tag(self, loader_with_special_tokens):
        """Test basic CIT tag generates correct BIO labels."""
        # [CLS] [CIT_START] word1 [CIT_END] [SEP]
        input_ids = [0, 104, 10, 105, 2]

        labels = loader_with_special_tokens.generate_bio_labels(input_ids)

        assert labels == [
            -100,             # CLS
            -100,             # CIT_START
            LABEL2ID["B-CIT"],  # First token
            -100,             # CIT_END
            -100,             # SEP
        ]

    def test_outside_tags(self, loader_with_special_tokens):
        """Test tokens outside any tags get O label."""
        # [CLS] word1 word2 [SEP]
        input_ids = [0, 10, 11, 2]

        labels = loader_with_special_tokens.generate_bio_labels(input_ids)

        assert labels == [
            -100,        # CLS
            LABEL2ID["O"],  # word1
            LABEL2ID["O"],  # word2
            -100,        # SEP
        ]

    def test_mixed_tags_and_outside(self, loader_with_special_tokens):
        """Test mix of tagged and untagged content."""
        # [CLS] word1 [BIBL_START] word2 [BIBL_END] word3 [SEP]
        input_ids = [0, 10, 100, 11, 101, 12, 2]

        labels = loader_with_special_tokens.generate_bio_labels(input_ids)

        assert labels == [
            -100,              # CLS
            LABEL2ID["O"],        # word1 (outside)
            -100,              # BIBL_START
            LABEL2ID["B-BIBL"],   # word2
            -100,              # BIBL_END
            LABEL2ID["O"],        # word3 (outside)
            -100,              # SEP
        ]

    def test_multiple_tags_same_type(self, loader_with_special_tokens):
        """Test multiple instances of same tag type."""
        # [CLS] [BIBL_START] word1 [BIBL_END] [BIBL_START] word2 [BIBL_END] [SEP]
        input_ids = [0, 100, 10, 101, 100, 11, 101, 2]

        labels = loader_with_special_tokens.generate_bio_labels(input_ids)

        assert labels == [
            -100,              # CLS
            -100,              # BIBL_START
            LABEL2ID["B-BIBL"],   # word1
            -100,              # BIBL_END
            -100,              # BIBL_START (new tag)
            LABEL2ID["B-BIBL"],   # word2 (new B- tag)
            -100,              # BIBL_END
            -100,              # SEP
        ]

    def test_nested_tags(self, loader_with_special_tokens):
        """Test nested tags (CIT containing BIBL)."""
        # [CLS] [CIT_START] [BIBL_START] word1 [BIBL_END] [CIT_END] [SEP]
        input_ids = [0, 104, 100, 10, 101, 105, 2]

        labels = loader_with_special_tokens.generate_bio_labels(input_ids)

        # Inner tag (BIBL) takes precedence
        assert labels == [
            -100,              # CLS
            -100,              # CIT_START
            -100,              # BIBL_START
            LABEL2ID["B-BIBL"],   # word1 (BIBL, not CIT)
            -100,              # BIBL_END
            -100,              # CIT_END
            -100,              # SEP
        ]

    def test_different_tag_types_sequential(self, loader_with_special_tokens):
        """Test different tag types in sequence."""
        # [CLS] [BIBL_START] w1 [BIBL_END] [QUOTE_START] w2 [QUOTE_END] [SEP]
        input_ids = [0, 100, 10, 101, 102, 11, 103, 2]

        labels = loader_with_special_tokens.generate_bio_labels(input_ids)

        assert labels == [
            -100,               # CLS
            -100,               # BIBL_START
            LABEL2ID["B-BIBL"],    # w1
            -100,               # BIBL_END
            -100,               # QUOTE_START
            LABEL2ID["B-QUOTE"],   # w2
            -100,               # QUOTE_END
            -100,               # SEP
        ]

    def test_padding_tokens(self, loader_with_special_tokens):
        """Test that padding tokens get -100 label."""
        # [CLS] [BIBL_START] word1 [BIBL_END] [SEP] [PAD] [PAD]
        input_ids = [0, 100, 10, 101, 2, 1, 1]

        labels = loader_with_special_tokens.generate_bio_labels(input_ids)

        assert labels == [
            -100,              # CLS
            -100,              # BIBL_START
            LABEL2ID["B-BIBL"],   # word1
            -100,              # BIBL_END
            -100,              # SEP
            -100,              # PAD
            -100,              # PAD
        ]

    def test_empty_tag(self, loader_with_special_tokens):
        """Test tag with no content between START and END."""
        # [CLS] [BIBL_START] [BIBL_END] [SEP]
        input_ids = [0, 100, 101, 2]

        labels = loader_with_special_tokens.generate_bio_labels(input_ids)

        assert labels == [
            -100,  # CLS
            -100,  # BIBL_START
            -100,  # BIBL_END
            -100,  # SEP
        ]

    def test_label_count_matches_input_count(self, loader_with_special_tokens):
        """Test that output labels have same length as input_ids."""
        input_ids = [0, 100, 10, 11, 12, 101, 2]

        labels = loader_with_special_tokens.generate_bio_labels(input_ids)

        assert len(labels) == len(input_ids)


class TestStripSpecialTokens:
    """Test suite for strip_special_tokens_and_align_labels function."""

    @pytest.fixture
    def loader_with_special_tokens(self, mocker, mock_tokenizer):
        """Create an ExtractionDataLoader with mocked tokenizer."""
        loader = ExtractionDataLoader()

        # Replace the tokenizer with a mock by patching the underlying attribute
        mock_tok = mocker.Mock()

        # Mock token IDs
        mock_tok.cls_token_id = 0
        mock_tok.sep_token_id = 2
        mock_tok.pad_token_id = 1

        # Mock special token conversions
        special_token_map = {
            "[BIBL_START]": 100,
            "[BIBL_END]": 101,
            "[QUOTE_START]": 102,
            "[QUOTE_END]": 103,
            "[CIT_START]": 104,
            "[CIT_END]": 105,
        }

        def convert_tokens_to_ids(token):
            return special_token_map.get(token, -1)

        mock_tok.convert_tokens_to_ids = convert_tokens_to_ids
        loader._tokenizer = mock_tok

        return loader

    def test_strips_bibl_tokens(self, loader_with_special_tokens):
        """Test that BIBL special tokens are removed from input."""
        # [CLS] [BIBL_START] word1 word2 [BIBL_END] [SEP]
        input_ids = [0, 100, 10, 11, 101, 2]
        labels = [-100, -100, LABEL2ID["B-BIBL"], LABEL2ID["I-BIBL"], -100, -100]

        clean_ids, aligned_labels = loader_with_special_tokens.strip_special_tokens_and_align_labels(
            input_ids, labels
        )

        # Should remove [BIBL_START] (100) and [BIBL_END] (101)
        assert clean_ids == [0, 10, 11, 2]  # CLS, word1, word2, SEP
        assert aligned_labels == [-100, LABEL2ID["B-BIBL"], LABEL2ID["I-BIBL"], -100]

    def test_strips_quote_tokens(self, loader_with_special_tokens):
        """Test that QUOTE special tokens are removed from input."""
        # [CLS] [QUOTE_START] word1 word2 [QUOTE_END] [SEP]
        input_ids = [0, 102, 10, 11, 103, 2]
        labels = [-100, -100, LABEL2ID["B-QUOTE"], LABEL2ID["I-QUOTE"], -100, -100]

        clean_ids, aligned_labels = loader_with_special_tokens.strip_special_tokens_and_align_labels(
            input_ids, labels
        )

        # Should remove [QUOTE_START] (102) and [QUOTE_END] (103)
        assert clean_ids == [0, 10, 11, 2]
        assert aligned_labels == [-100, LABEL2ID["B-QUOTE"], LABEL2ID["I-QUOTE"], -100]

    def test_strips_cit_tokens(self, loader_with_special_tokens):
        """Test that CIT special tokens are removed from input."""
        # [CLS] [CIT_START] word1 [CIT_END] [SEP]
        input_ids = [0, 104, 10, 105, 2]
        labels = [-100, -100, LABEL2ID["B-CIT"], -100, -100]

        clean_ids, aligned_labels = loader_with_special_tokens.strip_special_tokens_and_align_labels(
            input_ids, labels
        )

        # Should remove [CIT_START] (104) and [CIT_END] (105)
        assert clean_ids == [0, 10, 2]
        assert aligned_labels == [-100, LABEL2ID["B-CIT"], -100]

    def test_strips_all_special_token_types(self, loader_with_special_tokens):
        """Test that all citation special tokens are removed."""
        # [CLS] [BIBL_START] w1 [BIBL_END] [QUOTE_START] w2 [QUOTE_END] [CIT_START] w3 [CIT_END] [SEP]
        input_ids = [0, 100, 10, 101, 102, 11, 103, 104, 12, 105, 2]
        labels = [
            -100,                  # CLS
            -100,                  # BIBL_START
            LABEL2ID["B-BIBL"],    # w1
            -100,                  # BIBL_END
            -100,                  # QUOTE_START
            LABEL2ID["B-QUOTE"],   # w2
            -100,                  # QUOTE_END
            -100,                  # CIT_START
            LABEL2ID["B-CIT"],     # w3
            -100,                  # CIT_END
            -100,                  # SEP
        ]

        clean_ids, aligned_labels = loader_with_special_tokens.strip_special_tokens_and_align_labels(
            input_ids, labels
        )

        # Should only have CLS, w1, w2, w3, SEP (all 6 special tokens removed)
        assert clean_ids == [0, 10, 11, 12, 2]
        assert aligned_labels == [
            -100,
            LABEL2ID["B-BIBL"],
            LABEL2ID["B-QUOTE"],
            LABEL2ID["B-CIT"],
            -100,
        ]

    def test_preserves_non_special_tokens(self, loader_with_special_tokens):
        """Test that regular tokens and their labels are preserved."""
        # [CLS] word1 [BIBL_START] word2 [BIBL_END] word3 [SEP]
        input_ids = [0, 10, 100, 11, 101, 12, 2]
        labels = [
            -100,               # CLS
            LABEL2ID["O"],      # word1 (outside)
            -100,               # BIBL_START
            LABEL2ID["B-BIBL"], # word2
            -100,               # BIBL_END
            LABEL2ID["O"],      # word3 (outside)
            -100,               # SEP
        ]

        clean_ids, aligned_labels = loader_with_special_tokens.strip_special_tokens_and_align_labels(
            input_ids, labels
        )

        # Should preserve CLS, word1, word2, word3, SEP
        assert clean_ids == [0, 10, 11, 12, 2]
        assert aligned_labels == [
            -100,
            LABEL2ID["O"],
            LABEL2ID["B-BIBL"],
            LABEL2ID["O"],
            -100,
        ]

    def test_alignment_maintained(self, loader_with_special_tokens):
        """Test that labels remain aligned with tokens after stripping."""
        # [CLS] [BIBL_START] w1 w2 w3 [BIBL_END] [SEP]
        input_ids = [0, 100, 10, 11, 12, 101, 2]
        labels = [
            -100,               # CLS
            -100,               # BIBL_START
            LABEL2ID["B-BIBL"], # w1
            LABEL2ID["I-BIBL"], # w2
            LABEL2ID["I-BIBL"], # w3
            -100,               # BIBL_END
            -100,               # SEP
        ]

        clean_ids, aligned_labels = loader_with_special_tokens.strip_special_tokens_and_align_labels(
            input_ids, labels
        )

        # Length should match
        assert len(clean_ids) == len(aligned_labels)
        # Each position should be correctly aligned
        assert clean_ids == [0, 10, 11, 12, 2]
        assert aligned_labels == [
            -100,
            LABEL2ID["B-BIBL"],
            LABEL2ID["I-BIBL"],
            LABEL2ID["I-BIBL"],
            -100,
        ]

    def test_empty_input(self, loader_with_special_tokens):
        """Test handling of empty input."""
        input_ids = []
        labels = []

        clean_ids, aligned_labels = loader_with_special_tokens.strip_special_tokens_and_align_labels(
            input_ids, labels
        )

        assert clean_ids == []
        assert aligned_labels == []

    def test_only_special_tokens(self, loader_with_special_tokens):
        """Test input containing only special tokens."""
        # [BIBL_START] [BIBL_END]
        input_ids = [100, 101]
        labels = [-100, -100]

        clean_ids, aligned_labels = loader_with_special_tokens.strip_special_tokens_and_align_labels(
            input_ids, labels
        )

        # Should be empty after removing special tokens
        assert clean_ids == []
        assert aligned_labels == []


class TestCreateExtractionDataset:
    """Test suite for create_extraction_dataset function."""

    @pytest.fixture
    def temp_jsonl_file(self):
        """Create a temporary JSONL file with extraction data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            test_data = [
                {
                    "xml_context": "<bibl>Hdt. 8.82</bibl>",
                    "filename": "file1.xml",
                },
                {
                    "xml_context": "<quote>test quote</quote>",
                    "filename": "file2.xml",
                },
                {
                    "xml_context": "plain text",
                    "filename": "file3.xml",
                },
            ]
            for item in test_data:
                f.write(json.dumps(item) + '\n')
            temp_path = f.name

        yield temp_path

        # Cleanup
        Path(temp_path).unlink()

    def test_returns_dataset(self, temp_jsonl_file, mock_tokenizer):
        """Test that create_extraction_dataset returns a Dataset."""
        from datasets import Dataset

        dataset = create_extraction_dataset(temp_jsonl_file)

        assert isinstance(dataset, Dataset)

    def test_dataset_has_correct_columns(self, temp_jsonl_file, mock_tokenizer):
        """Test that dataset has expected columns."""
        dataset = create_extraction_dataset(temp_jsonl_file)

        assert "input_ids" in dataset.column_names
        assert "attention_mask" in dataset.column_names
        assert "labels" in dataset.column_names
        assert "filename" in dataset.column_names

    def test_dataset_has_correct_length(self, temp_jsonl_file, mock_tokenizer):
        """Test that dataset has same number of examples as input."""
        dataset = create_extraction_dataset(temp_jsonl_file)

        assert len(dataset) == 3

    def test_labels_same_length_as_input_ids(self, temp_jsonl_file, mock_tokenizer):
        """Test that labels have same length as input_ids for each example."""
        dataset = create_extraction_dataset(temp_jsonl_file)

        for i in range(len(dataset)):
            assert len(dataset[i]["labels"]) == len(dataset[i]["input_ids"])

    def test_attention_mask_same_length_as_input_ids(self, temp_jsonl_file, mock_tokenizer):
        """Test that attention_mask has same length as input_ids."""
        dataset = create_extraction_dataset(temp_jsonl_file)

        for i in range(len(dataset)):
            assert len(dataset[i]["attention_mask"]) == len(dataset[i]["input_ids"])

    def test_preserves_filenames(self, temp_jsonl_file, mock_tokenizer):
        """Test that filenames are preserved in dataset."""
        dataset = create_extraction_dataset(temp_jsonl_file)

        assert dataset[0]["filename"] == "file1.xml"
        assert dataset[1]["filename"] == "file2.xml"
        assert dataset[2]["filename"] == "file3.xml"

    def test_preserves_order(self, temp_jsonl_file, mock_tokenizer):
        """Test that examples maintain input order."""
        dataset = create_extraction_dataset(temp_jsonl_file)

        # Filenames should be in same order as input
        filenames = [dataset[i]["filename"] for i in range(len(dataset))]
        assert filenames == ["file1.xml", "file2.xml", "file3.xml"]

    def test_labels_are_integers(self, temp_jsonl_file, mock_tokenizer):
        """Test that labels are integer IDs."""
        dataset = create_extraction_dataset(temp_jsonl_file)

        for i in range(len(dataset)):
            for label in dataset[i]["labels"]:
                assert isinstance(label, int)

    def test_labels_in_valid_range(self, temp_jsonl_file, mock_tokenizer):
        """Test that label IDs are in valid range."""
        dataset = create_extraction_dataset(temp_jsonl_file)

        valid_label_ids = set(LABEL2ID.values()) | {-100}

        for i in range(len(dataset)):
            for label in dataset[i]["labels"]:
                assert label in valid_label_ids

    def test_input_ids_are_integers(self, temp_jsonl_file, mock_tokenizer):
        """Test that input_ids are integers."""
        dataset = create_extraction_dataset(temp_jsonl_file)

        for i in range(len(dataset)):
            for token_id in dataset[i]["input_ids"]:
                assert isinstance(token_id, int)

    def test_custom_config_path(self, temp_jsonl_file, mock_tokenizer):
        """Test dataset creation with custom config path."""
        from datasets import Dataset

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
model_name: bert-base-uncased
max_length: 256
""")
            config_path = f.name

        try:
            dataset = create_extraction_dataset(temp_jsonl_file, config_path=config_path)

            # Should succeed and create dataset
            assert isinstance(dataset, Dataset)
            assert len(dataset) == 3
        finally:
            Path(config_path).unlink()

    def test_dataset_strips_special_tokens_from_input(self, temp_jsonl_file, mock_tokenizer):
        """Test that special tokens are NOT in the final dataset input_ids."""
        dataset = create_extraction_dataset(temp_jsonl_file)

        # Get the loader to check special token IDs
        loader = ExtractionDataLoader()

        # These are the IDs we should NOT see in input_ids
        special_token_ids = {
            loader.tokenizer.convert_tokens_to_ids("[BIBL_START]"),
            loader.tokenizer.convert_tokens_to_ids("[BIBL_END]"),
            loader.tokenizer.convert_tokens_to_ids("[QUOTE_START]"),
            loader.tokenizer.convert_tokens_to_ids("[QUOTE_END]"),
            loader.tokenizer.convert_tokens_to_ids("[CIT_START]"),
            loader.tokenizer.convert_tokens_to_ids("[CIT_END]"),
        }

        # Check that none of the special tokens appear in input_ids
        for i in range(len(dataset)):
            input_ids = dataset[i]["input_ids"]
            for token_id in input_ids:
                assert token_id not in special_token_ids, \
                    f"Found special token {token_id} in input_ids at position {i}"
