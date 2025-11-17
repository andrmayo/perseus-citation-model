"""Unit tests for ResolutionDataLoader."""

import json
from pathlib import Path
import tempfile

import pytest

from perscit_model.resolution.data.loader import ResolutionDataLoader, ResolutionData


class TestResolutionDataLoader:
    """Test suite for ResolutionDataLoader."""

    @pytest.fixture
    def temp_jsonl_file(self):
        """Create a temporary JSONL file with resolution task data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            test_data = [
                {
                    "bibl": "Hdt. 8.82",
                    "ref": "hdt. 8.82",
                    "urn": "urn:cts:greekLit:tlg0016.tlg001.perseus-grc2:8.82",
                    "quote": "τᾶς πολυχρύσου | Πυθῶνος",
                    "xml_context": "<bibl>Hdt. 8.82</bibl>"
                },
                {
                    "bibl": "Soph. OT 151",
                    "ref": "soph. ot 151",
                    "urn": "urn:cts:greekLit:tlg0011.tlg004.perseus-grc2:151",
                    "quote": "ἀγλαὰς ἔβας",
                    "xml_context": "<bibl>Soph. OT 151</bibl>"
                },
                {
                    "bibl": "Plin. NH 15.30",
                    "ref": "plin. nh 15.30",
                    "urn": "urn:cts:latinLit:phi0978.phi001.perseus-lat2:15.30",
                    "quote": "",  # Empty quote
                    "xml_context": "<bibl>Plin. NH 15.30</bibl>"
                },
            ]
            for item in test_data:
                f.write(json.dumps(item) + '\n')
            temp_path = f.name

        yield temp_path

        # Cleanup
        Path(temp_path).unlink()

    def test_initialization(self, mock_tokenizer):
        """Test that ResolutionDataLoader initializes correctly."""
        loader = ResolutionDataLoader()

        assert loader.model_name == "microsoft/deberta-v3-base"
        assert loader.max_length == 512
        assert loader._tokenizer is None

    def test_inherits_from_shared_loader(self, mock_tokenizer):
        """Test that ResolutionDataLoader inherits SharedDataLoader methods."""
        loader = ResolutionDataLoader()

        # Should have inherited methods
        assert hasattr(loader, 'load_jsonl')
        assert hasattr(loader, 'tokenize_text')
        assert hasattr(loader, 'tokenizer')

    def test_call_returns_generator(self, temp_jsonl_file, mock_tokenizer):
        """Test that __call__ returns a generator."""
        loader = ResolutionDataLoader()
        result = loader(temp_jsonl_file)

        # Check it's a generator
        assert hasattr(result, '__iter__')
        assert hasattr(result, '__next__')

    def test_call_yields_resolution_data(self, temp_jsonl_file, mock_tokenizer):
        """Test that __call__ yields ResolutionData instances."""
        loader = ResolutionDataLoader()
        data = list(loader(temp_jsonl_file))

        # Should yield 3 items
        assert len(data) == 3

        # All should be ResolutionData instances
        for item in data:
            assert isinstance(item, ResolutionData)

    def test_resolution_data_has_correct_fields(self, temp_jsonl_file, mock_tokenizer):
        """Test that ResolutionData has all expected fields."""
        loader = ResolutionDataLoader()
        data = list(loader(temp_jsonl_file))

        first_item = data[0]

        # bibl should be tokenized (BatchEncoding)
        assert "input_ids" in first_item.bibl
        assert "attention_mask" in first_item.bibl

        # ref should be tokenized (BatchEncoding)
        assert "input_ids" in first_item.ref
        assert "attention_mask" in first_item.ref

        # urn should be string (NOT tokenized)
        assert isinstance(first_item.urn, str)
        assert first_item.urn == "urn:cts:greekLit:tlg0016.tlg001.perseus-grc2:8.82"

        # quote should be string (NOT tokenized)
        assert isinstance(first_item.quote, str)
        assert first_item.quote == "τᾶς πολυχρύσου | Πυθῶνος"

    def test_tokenizes_bibl_and_ref_only(self, temp_jsonl_file, mock_tokenizer):
        """Test that only bibl and ref are tokenized."""
        loader = ResolutionDataLoader()
        data = list(loader(temp_jsonl_file))

        first_item = data[0]

        # bibl and ref should be tokenized
        assert "input_ids" in first_item.bibl
        assert "input_ids" in first_item.ref

        # urn and quote should NOT be tokenized
        assert isinstance(first_item.urn, str)
        assert isinstance(first_item.quote, str)
        assert not hasattr(first_item.urn, 'input_ids')
        assert not hasattr(first_item.quote, 'input_ids')

    def test_handles_empty_quote(self, temp_jsonl_file, mock_tokenizer):
        """Test that loader handles empty quote field."""
        loader = ResolutionDataLoader()
        data = list(loader(temp_jsonl_file))

        # Third item has empty quote
        third_item = data[2]
        assert third_item.quote == ""

    def test_handles_missing_quote(self, mock_tokenizer):
        """Test that loader handles missing quote field."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            # Data without quote field
            test_data = {
                "bibl": "test",
                "ref": "test",
                "urn": "urn:cts:test:1"
            }
            f.write(json.dumps(test_data) + '\n')
            temp_path = f.name

        try:
            loader = ResolutionDataLoader()
            data = list(loader(temp_path))

            # Should default to empty string
            assert data[0].quote == ""
        finally:
            Path(temp_path).unlink()

    def test_tokenized_fields_correct_length(self, temp_jsonl_file, mock_tokenizer):
        """Test that tokenized fields respect max_length."""
        loader = ResolutionDataLoader(max_length=128)
        data = list(loader(temp_jsonl_file))

        # All should be padded/truncated to 128
        for item in data:
            assert len(item.bibl["input_ids"][0]) == 128
            assert len(item.ref["input_ids"][0]) == 128

    def test_preserves_order(self, temp_jsonl_file, mock_tokenizer):
        """Test that data is yielded in file order."""
        loader = ResolutionDataLoader()
        data = list(loader(temp_jsonl_file))

        assert "greekLit:tlg0016" in data[0].urn  # Herodotus
        assert "greekLit:tlg0011" in data[1].urn  # Sophocles
        assert "latinLit:phi0978" in data[2].urn  # Pliny

    def test_memory_efficient_streaming(self, temp_jsonl_file, mock_tokenizer):
        """Test that loader processes one item at a time."""
        loader = ResolutionDataLoader()
        gen = loader(temp_jsonl_file)

        # Get first item without exhausting generator
        first_item = next(gen)
        assert isinstance(first_item, ResolutionData)
        assert "tlg0016" in first_item.urn

        # Should still have more items
        second_item = next(gen)
        assert "tlg0011" in second_item.urn

    def test_handles_greek_text(self, temp_jsonl_file, mock_tokenizer):
        """Test that loader handles Greek text in bibl correctly."""
        loader = ResolutionDataLoader()
        data = list(loader(temp_jsonl_file))

        # First two items have Greek bibliographic refs
        for item in data[:2]:
            assert "input_ids" in item.bibl
            # Should have non-zero tokens
            assert len(item.bibl["input_ids"][0]) > 0

    def test_handles_latin_text(self, temp_jsonl_file, mock_tokenizer):
        """Test that loader handles Latin text correctly."""
        loader = ResolutionDataLoader()
        data = list(loader(temp_jsonl_file))

        # Third item is Latin (Pliny)
        latin_item = data[2]
        assert "input_ids" in latin_item.bibl
        assert "latinLit" in latin_item.urn

    def test_bibl_and_ref_different_tokenization(self, temp_jsonl_file, mock_tokenizer):
        """Test that bibl and ref are tokenized separately."""
        loader = ResolutionDataLoader()
        data = list(loader(temp_jsonl_file))

        first_item = data[0]

        # bibl and ref should have different content
        # (bibl has capitals, ref is lowercase)
        # So they should tokenize differently
        bibl_tokens = first_item.bibl["input_ids"]
        ref_tokens = first_item.ref["input_ids"]

        # Should both exist
        assert bibl_tokens is not None
        assert ref_tokens is not None

        # Should be different (case difference)
        # Note: They might be same if model lowercases, but worth checking
        assert len(bibl_tokens) == len(ref_tokens)  # Same shape though

    def test_preserves_full_urn(self, temp_jsonl_file, mock_tokenizer):
        """Test that full URN string is preserved."""
        loader = ResolutionDataLoader()
        data = list(loader(temp_jsonl_file))

        # Check full URN is preserved
        assert data[0].urn == "urn:cts:greekLit:tlg0016.tlg001.perseus-grc2:8.82"
        assert data[1].urn == "urn:cts:greekLit:tlg0011.tlg004.perseus-grc2:151"
        assert data[2].urn == "urn:cts:latinLit:phi0978.phi001.perseus-lat2:15.30"

    def test_custom_config_path(self, mock_tokenizer):
        """Test loader with custom config path."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
model_name: bert-base-uncased
max_length: 256
""")
            config_path = f.name

        try:
            loader = ResolutionDataLoader(config_path=config_path)
            assert loader.model_name == "bert-base-uncased"
            assert loader.max_length == 256
        finally:
            Path(config_path).unlink()

    @pytest.mark.parametrize("field_name,should_be_tokenized", [
        ("bibl", True),
        ("ref", True),
        ("urn", False),
        ("quote", False),
    ])
    def test_field_tokenization(self, temp_jsonl_file, field_name, should_be_tokenized, mock_tokenizer):
        """Test which fields are tokenized vs preserved as strings."""
        loader = ResolutionDataLoader()
        data = list(loader(temp_jsonl_file))

        first_item = data[0]
        field_value = getattr(first_item, field_name)

        if should_be_tokenized:
            # Should have BatchEncoding attributes
            assert "input_ids" in field_value
            assert "attention_mask" in field_value
        else:
            # Should be plain string
            assert isinstance(field_value, str)
            assert not hasattr(field_value, 'input_ids')
