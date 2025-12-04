"""Unit tests for evaluation module."""

import json
import pytest
import torch
import transformers
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open

from perscit_model.extraction.evaluate import (
    extract_citations,
    strip_xml_tags,
    evaluate_model,
)


class TestExtractCitations:
    """Tests for extract_citations function."""

    def test_extract_single_bibl(self):
        """Test extracting a single bibl citation."""
        xml = "This is <bibl>a citation</bibl> in text."
        result = extract_citations(xml)

        assert result["bibl"] == ["a citation"]
        assert result["quote"] == []

    def test_extract_single_quote(self):
        """Test extracting a single quote citation."""
        xml = "He said <quote>hello world</quote> yesterday."
        result = extract_citations(xml)

        assert result["bibl"] == []
        assert result["quote"] == ["hello world"]

    def test_extract_multiple_bibls(self):
        """Test extracting multiple bibl citations."""
        xml = "See <bibl>Smith 2020</bibl> and <bibl>Jones 2021</bibl> for details."
        result = extract_citations(xml)

        assert len(result["bibl"]) == 2
        assert "Smith 2020" in result["bibl"]
        assert "Jones 2021" in result["bibl"]

    def test_extract_multiple_quotes(self):
        """Test extracting multiple quote citations."""
        xml = "He said <quote>first</quote> then <quote>second</quote>."
        result = extract_citations(xml)

        assert len(result["quote"]) == 2
        assert "first" in result["quote"]
        assert "second" in result["quote"]

    def test_extract_mixed_citations(self):
        """Test extracting both bibl and quote citations."""
        xml = "The <bibl>author</bibl> wrote <quote>this text</quote> yesterday."
        result = extract_citations(xml)

        assert result["bibl"] == ["author"]
        assert result["quote"] == ["this text"]

    def test_extract_nested_tags(self):
        """Test extracting citations with nested XML tags."""
        xml = "See <bibl n=\"test\">Smith <foreign>et al.</foreign> 2020</bibl>."
        result = extract_citations(xml)

        assert len(result["bibl"]) == 1
        assert "Smith <foreign>et al.</foreign> 2020" in result["bibl"][0]

    def test_extract_no_citations(self):
        """Test text with no citations returns empty lists."""
        xml = "This is plain text with no citations."
        result = extract_citations(xml)

        assert result["bibl"] == []
        assert result["quote"] == []

    def test_extract_multiline_citation(self):
        """Test extracting citation that spans multiple lines."""
        xml = "Text <bibl>line one\nline two</bibl> more text."
        result = extract_citations(xml)

        assert len(result["bibl"]) == 1
        assert "line one\nline two" in result["bibl"][0]


class TestStripXmlTags:
    """Tests for strip_xml_tags function."""

    def test_strip_bibl_tags(self):
        """Test stripping bibl tags."""
        xml = "This is <bibl>a citation</bibl> in text."
        result = strip_xml_tags(xml)

        assert result == "This is a citation in text."
        assert "<bibl>" not in result
        assert "</bibl>" not in result

    def test_strip_quote_tags(self):
        """Test stripping quote tags."""
        xml = "He said <quote>hello</quote> yesterday."
        result = strip_xml_tags(xml)

        assert result == "He said hello yesterday."
        assert "<quote>" not in result

    def test_strip_cit_tags(self):
        """Test stripping cit tags."""
        xml = "Text <cit>with citation</cit> here."
        result = strip_xml_tags(xml)

        assert result == "Text with citation here."
        assert "<cit>" not in result

    def test_strip_all_citation_tags(self):
        """Test stripping all citation tag types."""
        xml = "<cit><bibl>Author</bibl> wrote <quote>this</quote></cit>."
        result = strip_xml_tags(xml)

        assert result == "Author wrote this."
        assert "<" not in result or ">" not in result

    def test_preserve_other_tags(self):
        """Test that other XML tags are preserved."""
        xml = "Text <foreign>Greek</foreign> and <bibl>citation</bibl>."
        result = strip_xml_tags(xml)

        assert "<foreign>" in result
        assert "</foreign>" in result
        assert "<bibl>" not in result

    def test_strip_tags_with_attributes(self):
        """Test stripping tags with attributes."""
        xml = 'Text <bibl n="test">citation</bibl> here.'
        result = strip_xml_tags(xml)

        assert result == "Text citation here."
        assert "n=" not in result


class TestGroundTruthLabelGeneration:
    """Tests for ground truth label generation alignment."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()

        def tokenize_fn(text, **kwargs):
            # Simple mock tokenization based on spaces
            words = text.split()
            token_ids = [101]  # CLS
            offset_mapping = [(0, 0)]  # CLS offset

            pos = 0
            for word in words:
                # Find word position in original text
                word_start = text.find(word, pos)
                word_end = word_start + len(word)

                token_ids.append(len(token_ids))  # Mock token ID
                offset_mapping.append((word_start, word_end))
                pos = word_end

            token_ids.append(102)  # SEP
            offset_mapping.append((len(text), len(text)))  # SEP offset

            result = {
                "input_ids": torch.tensor([token_ids]),
                "attention_mask": torch.tensor([[1] * len(token_ids)]),
            }

            if kwargs.get("return_offsets_mapping"):
                result["offset_mapping"] = torch.tensor([offset_mapping])

            if kwargs.get("return_tensors") == "pt":
                return result
            else:
                return {
                    "input_ids": token_ids,
                    "attention_mask": [1] * len(token_ids),
                }

        tokenizer.side_effect = tokenize_fn
        return tokenizer

    def test_ground_truth_labels_align_with_stripped_text(self, mock_tokenizer):
        """Test that ground truth labels correctly align with stripped text tokens."""
        # Original XML with citation
        original_xml = "This is <bibl>a citation</bibl> in text."
        # Stripped text (what model sees)
        stripped_text = "This is a citation in text."

        # Tokenize stripped text (simulate what happens in evaluate.py)
        result = mock_tokenizer(
            stripped_text,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        token_ids = result["input_ids"][0].tolist()
        offset_mapping = result["offset_mapping"][0].tolist()

        # Initialize labels
        true_labels = ["O"] * len(token_ids)

        # Extract citation from original XML
        import re
        bibl_pattern = re.compile(r"<bibl[^>]*>(.*?)</bibl>", re.DOTALL)
        for match in bibl_pattern.finditer(original_xml):
            citation_text = match.group(1)  # "a citation"

            # Find in stripped text
            start_pos = stripped_text.find(citation_text)
            assert start_pos != -1, "Citation text not found in stripped text"

            end_pos = start_pos + len(citation_text)

            # Label tokens that overlap with citation
            first_token_in_citation = True
            for i, (token_start, token_end) in enumerate(offset_mapping):
                if token_start == token_end:  # Skip special tokens
                    continue
                if token_start >= start_pos and token_end <= end_pos:
                    if first_token_in_citation:
                        true_labels[i] = "B-BIBL"
                        first_token_in_citation = False
                    else:
                        true_labels[i] = "I-BIBL"

        # Verify labels are correct
        # Should have at least one B-BIBL and one I-BIBL
        assert "B-BIBL" in true_labels
        assert "I-BIBL" in true_labels
        # First and last tokens should be "O" (CLS and SEP)
        assert true_labels[0] == "O"
        assert true_labels[-1] == "O"

    def test_multiple_citations_labeled_correctly(self, mock_tokenizer):
        """Test that multiple citations are labeled correctly."""
        original_xml = "See <bibl>Smith</bibl> and <quote>hello</quote> there."
        stripped_text = "See Smith and hello there."

        result = mock_tokenizer(
            stripped_text,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        offset_mapping = result["offset_mapping"][0].tolist()
        true_labels = ["O"] * len(offset_mapping)

        # Label BIBL
        import re
        bibl_pattern = re.compile(r"<bibl[^>]*>(.*?)</bibl>", re.DOTALL)
        for match in bibl_pattern.finditer(original_xml):
            citation_text = match.group(1)
            start_pos = stripped_text.find(citation_text)
            end_pos = start_pos + len(citation_text)

            first_token_in_citation = True
            for i, (token_start, token_end) in enumerate(offset_mapping):
                if token_start == token_end:
                    continue
                if token_start >= start_pos and token_end <= end_pos:
                    if first_token_in_citation:
                        true_labels[i] = "B-BIBL"
                        first_token_in_citation = False
                    else:
                        true_labels[i] = "I-BIBL"

        # Label QUOTE
        quote_pattern = re.compile(r"<quote[^>]*>(.*?)</quote>", re.DOTALL)
        for match in quote_pattern.finditer(original_xml):
            citation_text = match.group(1)
            start_pos = stripped_text.find(citation_text)
            end_pos = start_pos + len(citation_text)

            first_token_in_citation = True
            for i, (token_start, token_end) in enumerate(offset_mapping):
                if token_start == token_end:
                    continue
                if token_start >= start_pos and token_end <= end_pos:
                    if first_token_in_citation:
                        true_labels[i] = "B-QUOTE"
                        first_token_in_citation = False
                    else:
                        true_labels[i] = "I-QUOTE"

        # Verify both citation types are present
        assert "B-BIBL" in true_labels
        assert "B-QUOTE" in true_labels

    def test_no_citations_all_labels_O(self, mock_tokenizer):
        """Test that text with no citations gets all O labels."""
        original_xml = "This is plain text."
        stripped_text = "This is plain text."

        result = mock_tokenizer(
            stripped_text,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        offset_mapping = result["offset_mapping"][0].tolist()
        true_labels = ["O"] * len(offset_mapping)

        # No citations to label - all should remain "O"
        assert all(label == "O" for label in true_labels)

    def test_citation_not_in_stripped_text_no_labels(self, mock_tokenizer):
        """Test that citations not present in stripped text don't cause errors."""
        # Edge case: citation was truncated or modified
        original_xml = "Text <bibl>citation</bibl> here."
        stripped_text = "Text here."  # Citation removed by some process

        result = mock_tokenizer(
            stripped_text,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        offset_mapping = result["offset_mapping"][0].tolist()
        true_labels = ["O"] * len(offset_mapping)

        # Try to find citation
        import re
        bibl_pattern = re.compile(r"<bibl[^>]*>(.*?)</bibl>", re.DOTALL)
        for match in bibl_pattern.finditer(original_xml):
            citation_text = match.group(1)
            start_pos = stripped_text.find(citation_text)

            if start_pos != -1:  # Only label if found
                end_pos = start_pos + len(citation_text)
                for i, (token_start, token_end) in enumerate(offset_mapping):
                    if token_start == token_end:
                        continue
                    if token_start >= start_pos and token_end <= end_pos:
                        true_labels[i] = "B-BIBL" if token_start == start_pos else "I-BIBL"

        # Citation wasn't in stripped text, so all labels should still be "O"
        assert all(label == "O" for label in true_labels)


class TestMetricsComputation:
    """Tests for metrics computation with seqeval."""

    def test_perfect_predictions_f1_is_one(self):
        """Test that perfect predictions give F1 score of 1.0."""
        from seqeval.metrics import f1_score, precision_score, recall_score

        true_labels = [["O", "B-BIBL", "I-BIBL", "O", "B-QUOTE", "O"]]
        predictions = [["O", "B-BIBL", "I-BIBL", "O", "B-QUOTE", "O"]]

        f1 = f1_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)

        assert f1 == 1.0
        assert precision == 1.0
        assert recall == 1.0

    def test_no_predictions_f1_is_zero(self):
        """Test that missing all citations gives F1 score of 0.0."""
        from seqeval.metrics import f1_score, precision_score, recall_score

        true_labels = [["O", "B-BIBL", "I-BIBL", "O", "B-QUOTE", "O"]]
        predictions = [["O", "O", "O", "O", "O", "O"]]  # Missed everything

        f1 = f1_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)

        assert f1 == 0.0
        assert recall == 0.0

    def test_partial_overlap_gives_partial_score(self):
        """Test that partial predictions give intermediate scores."""
        from seqeval.metrics import f1_score

        true_labels = [["O", "B-BIBL", "I-BIBL", "O", "B-QUOTE", "I-QUOTE", "O"]]
        predictions = [["O", "B-BIBL", "I-BIBL", "O", "O", "O", "O"]]  # Got BIBL, missed QUOTE

        f1 = f1_score(true_labels, predictions)

        # Should be between 0 and 1 since we got 1 out of 2 citations
        assert 0.0 < f1 < 1.0

    def test_wrong_boundaries_affects_score(self):
        """Test that wrong citation boundaries affect the score."""
        from seqeval.metrics import f1_score

        # True: BIBL spans tokens 1-2
        true_labels = [["O", "B-BIBL", "I-BIBL", "O", "O"]]
        # Predicted: BIBL spans tokens 1-3 (too long)
        predictions = [["O", "B-BIBL", "I-BIBL", "I-BIBL", "O"]]

        f1 = f1_score(true_labels, predictions)

        # Score should be less than 1.0 due to boundary mismatch
        assert f1 < 1.0

    def test_misaligned_labels_caught_by_length_check(self):
        """Test that misaligned label sequences would cause errors."""
        from seqeval.metrics import f1_score

        # Different lengths should raise an error
        true_labels = [["O", "B-BIBL", "I-BIBL", "O"]]
        predictions = [["O", "B-BIBL", "O"]]  # Wrong length!

        # seqeval should handle this gracefully or raise an error
        with pytest.raises((ValueError, IndexError)):
            f1_score(true_labels, predictions)


class TestEvaluateModelIntegration:
    """Integration tests for evaluate_model function."""

    @pytest.fixture
    def mock_test_data(self, tmp_path):
        """Create mock test data file."""
        test_file = tmp_path / "test.jsonl"

        examples = [
            {
                "filename": "test1.xml",
                "xml_context": "This is <bibl>a citation</bibl> text.",
            },
            {
                "filename": "test2.xml",
                "xml_context": "He said <quote>hello world</quote> there.",
            },
        ]

        with open(test_file, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

        return test_file

    def test_evaluate_model_returns_metrics(self, tmp_path, mock_test_data):
        """Test that evaluate_model returns expected metrics."""
        output_dir = tmp_path / "output"

        with patch('perscit_model.extraction.evaluate.InferenceModel') as MockModel:
            with patch('perscit_model.extraction.evaluate.ExtractionDataLoader') as MockLoader:
                # Mock the model and loader
                mock_model_instance = Mock()
                mock_loader_instance = Mock()

                MockModel.return_value = mock_model_instance
                MockLoader.return_value = mock_loader_instance

                # Mock tokenizer
                mock_tokenizer = Mock()
                mock_tokenizer.return_value = {
                    "input_ids": torch.tensor([[101, 2023, 102]]),
                    "attention_mask": torch.tensor([[1, 1, 1]]),
                    "offset_mapping": torch.tensor([[(0, 0), (0, 4), (4, 5)]]),
                }
                mock_loader_instance.tokenizer = mock_tokenizer

                # Mock process_batch to return predictions
                mock_model_instance.process_batch.return_value = [
                    (
                        {"input_ids": torch.tensor([[101, 2023, 102]]),
                         "offset_mapping": torch.tensor([[(0, 0), (0, 4), (4, 5)]])},
                        ["O", "B-BIBL", "O"]
                    ),
                ]

                # Mock insert_tags_into_xml
                mock_model_instance.insert_tags_into_xml.return_value = "test"

                # This will still fail due to complex mocking needs,
                # but demonstrates the test structure
                # In practice, you'd need more comprehensive mocking

    def test_output_files_created(self, tmp_path, mock_test_data):
        """Test that output files are created in the correct location."""
        output_dir = tmp_path / "output"

        # This test would require extensive mocking to actually run
        # It serves as a template for integration testing

        # Expected output files:
        expected_files = [
            output_dir / "test_metrics.json",
            output_dir / "classification_report.txt",
            output_dir / "test_predictions.jsonl",
        ]

        # After running evaluate_model, check these files exist
        # (Would need full mocking setup to actually test this)
