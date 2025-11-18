"""Unit tests for ExtractionDataLoader."""

import json
from pathlib import Path
import tempfile

import pytest

from perscit_model.extraction.data_loader import ExtractionDataLoader, ExtractionData


class TestExtractionDataLoader:
    """Test suite for ExtractionDataLoader."""

    @pytest.fixture
    def temp_jsonl_file(self):
        """Create a temporary JSONL file with extraction task data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            test_data = [
                {
                    "xml_context": "<bibl>Hdt. 8.82</bibl> some context",
                    "filename": "file1.xml",
                    "bibl": "Hdt. 8.82",
                    "urn": "urn:cts:greekLit:tlg0016.tlg001:8.82"
                },
                {
                    "xml_context": "<quote>τᾶς πολυχρύσου</quote> <bibl>Soph. OT 151</bibl>",
                    "filename": "file2.xml",
                    "bibl": "Soph. OT 151",
                    "urn": "urn:cts:greekLit:tlg0011.tlg004:151"
                },
                {
                    "xml_context": "plain text <cit>citation</cit> more text",
                    "filename": "file3.xml",
                    "bibl": "Eur. Med. 1",
                    "urn": "urn:cts:greekLit:tlg0006.tlg012:1"
                },
            ]
            for item in test_data:
                f.write(json.dumps(item) + '\n')
            temp_path = f.name

        yield temp_path

        # Cleanup
        Path(temp_path).unlink()

    def test_initialization(self, mock_tokenizer):
        """Test that ExtractionDataLoader initializes correctly."""
        loader = ExtractionDataLoader()

        assert loader.model_name == "microsoft/deberta-v3-base"
        assert loader.max_length == 512
        assert loader._tokenizer is None

    def test_inherits_from_shared_loader(self, mock_tokenizer):
        """Test that ExtractionDataLoader inherits SharedDataLoader methods."""
        loader = ExtractionDataLoader()

        # Should have inherited methods
        assert hasattr(loader, 'load_jsonl')
        assert hasattr(loader, 'tokenize_text')
        assert hasattr(loader, 'tokenizer')

    def test_call_returns_generator(self, temp_jsonl_file, mock_tokenizer):
        """Test that __call__ returns a generator."""
        loader = ExtractionDataLoader()
        result = loader(temp_jsonl_file)

        # Check it's a generator
        assert hasattr(result, '__iter__')
        assert hasattr(result, '__next__')

    def test_call_yields_extraction_data(self, temp_jsonl_file, mock_tokenizer):
        """Test that __call__ yields ExtractionData instances."""
        loader = ExtractionDataLoader()
        data = list(loader(temp_jsonl_file))

        # Should yield 3 items
        assert len(data) == 3

        # All should be ExtractionData instances
        for item in data:
            assert isinstance(item, ExtractionData)

    def test_extraction_data_has_correct_fields(self, temp_jsonl_file, mock_tokenizer):
        """Test that ExtractionData has xml_context and filename fields."""
        loader = ExtractionDataLoader()
        data = list(loader(temp_jsonl_file))

        first_item = data[0]

        # Should have xml_context as BatchEncoding
        assert "input_ids" in first_item.xml_context
        assert "attention_mask" in first_item.xml_context

        # Should have filename as string
        assert isinstance(first_item.filename, str)
        assert first_item.filename == "file1.xml"

    def test_only_tokenizes_xml_context(self, temp_jsonl_file, mock_tokenizer):
        """Test that only xml_context is tokenized, not other fields."""
        loader = ExtractionDataLoader()
        data = list(loader(temp_jsonl_file))

        first_item = data[0]

        # xml_context should be tokenized (BatchEncoding)
        assert "input_ids" in first_item.xml_context

        # filename should NOT be tokenized (just string)
        assert isinstance(first_item.filename, str)
        assert not hasattr(first_item.filename, 'input_ids')

    def test_handles_missing_filename(self, mock_tokenizer):
        """Test that loader handles missing filename field."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            # Data without filename
            test_data = {"xml_context": "test context", "bibl": "test"}
            f.write(json.dumps(test_data) + '\n')
            temp_path = f.name

        try:
            loader = ExtractionDataLoader()
            data = list(loader(temp_path))

            # Should default to empty string
            assert data[0].filename == ""
        finally:
            Path(temp_path).unlink()

    def test_tokenized_context_correct_length(self, temp_jsonl_file, mock_tokenizer):
        """Test that tokenized context respects max_length."""
        loader = ExtractionDataLoader(max_length=128)
        data = list(loader(temp_jsonl_file))

        # All should be padded/truncated to 128
        for item in data:
            assert len(item.xml_context["input_ids"][0]) == 128

    def test_preserves_order(self, temp_jsonl_file, mock_tokenizer):
        """Test that data is yielded in file order."""
        loader = ExtractionDataLoader()
        data = list(loader(temp_jsonl_file))

        assert data[0].filename == "file1.xml"
        assert data[1].filename == "file2.xml"
        assert data[2].filename == "file3.xml"

    def test_memory_efficient_streaming(self, temp_jsonl_file, mock_tokenizer):
        """Test that loader processes one item at a time."""
        loader = ExtractionDataLoader()
        gen = loader(temp_jsonl_file)

        # Get first item without exhausting generator
        first_item = next(gen)
        assert isinstance(first_item, ExtractionData)
        assert first_item.filename == "file1.xml"

        # Should still have more items
        second_item = next(gen)
        assert second_item.filename == "file2.xml"

    def test_handles_greek_text(self, temp_jsonl_file, mock_tokenizer):
        """Test that loader handles Greek text correctly."""
        loader = ExtractionDataLoader()
        data = list(loader(temp_jsonl_file))

        # Second item has Greek text
        greek_item = data[1]
        assert "input_ids" in greek_item.xml_context
        # Should have non-zero tokens (not empty)
        assert len(greek_item.xml_context["input_ids"][0]) > 0

    def test_custom_config_path(self, mock_tokenizer):
        """Test loader with custom config path."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
model_name: bert-base-uncased
max_length: 256
""")
            config_path = f.name

        try:
            loader = ExtractionDataLoader(config_path=config_path)
            assert loader.model_name == "bert-base-uncased"
            assert loader.max_length == 256
        finally:
            Path(config_path).unlink()
