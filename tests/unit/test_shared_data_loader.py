"""Unit tests for SharedDataLoader."""

import json
from pathlib import Path
import tempfile

import pytest

from perscit_model.shared.data_loader import SharedDataLoader


class TestSharedDataLoader:
    """Test suite for SharedDataLoader."""

    @pytest.fixture
    def temp_jsonl_file(self):
        """Create a temporary JSONL file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            # Write test data
            test_data = [
                {"xml_context": "test context 1", "bibl": "Hdt. 1.1", "urn": "urn:cts:greekLit:tlg0016.tlg001:1.1"},
                {"xml_context": "test context 2", "bibl": "Soph. OT 1", "urn": "urn:cts:greekLit:tlg0011.tlg004:1"},
                {"xml_context": "test context 3", "bibl": "Eur. Med. 1", "urn": "urn:cts:greekLit:tlg0006.tlg012:1"},
            ]
            for item in test_data:
                f.write(json.dumps(item) + '\n')
            temp_path = f.name

        yield temp_path

        # Cleanup
        Path(temp_path).unlink()

    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary YAML config file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
model_name: microsoft/deberta-v3-base
max_length: 128
learning_rate: 2e-5
num_train_epochs: 3
""")
            temp_path = f.name

        yield temp_path

        # Cleanup
        Path(temp_path).unlink()

    def test_init_with_defaults(self):
        """Test initialization with default config."""
        loader = SharedDataLoader()

        assert loader.model_name == "microsoft/deberta-v3-base"
        assert loader.max_length == 512
        assert loader._tokenizer is None

    def test_init_with_explicit_params(self):
        """Test initialization with explicit parameters."""
        loader = SharedDataLoader(
            model_name="bert-base-uncased",
            max_length=256
        )

        assert loader.model_name == "bert-base-uncased"
        assert loader.max_length == 256

    def test_init_with_config_path(self, temp_config_file):
        """Test initialization from YAML config file."""
        loader = SharedDataLoader(config_path=temp_config_file)

        assert loader.model_name == "microsoft/deberta-v3-base"
        assert loader.max_length == 128

    def test_init_explicit_params_override_config(self, temp_config_file):
        """Test that explicit params override config values."""
        loader = SharedDataLoader(
            model_name="custom-model",
            config_path=temp_config_file
        )

        assert loader.model_name == "custom-model"
        assert loader.max_length == 128  # From config

    def test_from_yaml_with_path(self, temp_config_file):
        """Test loading config from YAML file."""
        model_name, max_length = SharedDataLoader.from_yaml(temp_config_file)

        assert model_name == "microsoft/deberta-v3-base"
        assert max_length == 128

    def test_from_yaml_with_none(self):
        """Test loading config with None uses default config."""
        model_name, max_length = SharedDataLoader.from_yaml(None)

        assert model_name == "microsoft/deberta-v3-base"
        assert max_length == 512

    def test_load_jsonl_is_generator(self, temp_jsonl_file):
        """Test that load_jsonl returns a generator."""
        loader = SharedDataLoader()
        result = loader.load_jsonl(temp_jsonl_file)

        # Check it's a generator
        assert hasattr(result, '__iter__')
        assert hasattr(result, '__next__')

    def test_load_jsonl_yields_correct_data(self, temp_jsonl_file):
        """Test that load_jsonl yields correct parsed data."""
        loader = SharedDataLoader()
        data = list(loader.load_jsonl(temp_jsonl_file))

        assert len(data) == 3
        assert data[0]["bibl"] == "Hdt. 1.1"
        assert data[1]["bibl"] == "Soph. OT 1"
        assert data[2]["bibl"] == "Eur. Med. 1"

    def test_load_jsonl_memory_efficient(self, temp_jsonl_file):
        """Test that load_jsonl doesn't load all data at once."""
        loader = SharedDataLoader()
        gen = loader.load_jsonl(temp_jsonl_file)

        # Get first item
        first_item = next(gen)
        assert first_item["bibl"] == "Hdt. 1.1"

        # Generator should still have more items
        second_item = next(gen)
        assert second_item["bibl"] == "Soph. OT 1"

    def test_tokenizer_lazy_loading(self, mocker):
        """Test that tokenizer is loaded lazily."""
        # Mock AutoTokenizer
        mock_tokenizer = mocker.Mock()
        mocker.patch(
            'perscit_model.shared.data_loader.AutoTokenizer.from_pretrained',
            return_value=mock_tokenizer
        )

        loader = SharedDataLoader()

        # Initially None
        assert loader._tokenizer is None

        # Access tokenizer
        tokenizer = loader.tokenizer

        # Now loaded
        assert tokenizer is mock_tokenizer
        assert loader._tokenizer is mock_tokenizer

    def test_tokenizer_cached(self, mocker):
        """Test that tokenizer is cached after first load."""
        # Mock AutoTokenizer
        mock_tokenizer = mocker.Mock()
        mock_from_pretrained = mocker.patch(
            'perscit_model.shared.data_loader.AutoTokenizer.from_pretrained',
            return_value=mock_tokenizer
        )

        loader = SharedDataLoader()

        # Access twice
        tokenizer1 = loader.tokenizer
        tokenizer2 = loader.tokenizer

        # Should be same object
        assert tokenizer1 is tokenizer2
        # Should only call from_pretrained once
        assert mock_from_pretrained.call_count == 1

    def test_tokenize_text_returns_batch_encoding(self, mocker):
        """Test that tokenize_text returns BatchEncoding."""
        # Mock tokenizer callable
        mock_tokenizer = mocker.Mock()
        mock_tokenizer.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]]
        }
        mocker.patch(
            'perscit_model.shared.data_loader.AutoTokenizer.from_pretrained',
            return_value=mock_tokenizer
        )

        loader = SharedDataLoader()
        result = loader.tokenize_text("test text")

        # Check it has expected keys
        assert "input_ids" in result
        assert "attention_mask" in result

    def test_tokenize_text_uses_default_max_length(self, mocker):
        """Test that tokenize_text uses loader's max_length by default."""
        # Mock tokenizer callable
        mock_tokenizer = mocker.Mock()
        mock_tokenizer.return_value = {
            "input_ids": [[1] * 128],
            "attention_mask": [[1] * 128]
        }
        mocker.patch(
            'perscit_model.shared.data_loader.AutoTokenizer.from_pretrained',
            return_value=mock_tokenizer
        )

        loader = SharedDataLoader(max_length=128)
        result = loader.tokenize_text("test text")

        # Check tokenizer was called with correct max_length
        mock_tokenizer.assert_called_once()
        call_kwargs = mock_tokenizer.call_args[1]
        assert call_kwargs['max_length'] == 128

    def test_tokenize_text_with_custom_max_length(self, mocker):
        """Test that tokenize_text accepts custom max_length."""
        # Mock tokenizer callable
        mock_tokenizer = mocker.Mock()
        mock_tokenizer.return_value = {
            "input_ids": [[1] * 64],
            "attention_mask": [[1] * 64]
        }
        mocker.patch(
            'perscit_model.shared.data_loader.AutoTokenizer.from_pretrained',
            return_value=mock_tokenizer
        )

        loader = SharedDataLoader(max_length=512)
        result = loader.tokenize_text("test text", max_length=64)

        # Check tokenizer was called with custom max_length (64, not 512)
        mock_tokenizer.assert_called_once()
        call_kwargs = mock_tokenizer.call_args[1]
        assert call_kwargs['max_length'] == 64

    def test_tokenize_text_truncation(self, mocker):
        """Test that truncation is enabled."""
        # Mock tokenizer callable
        mock_tokenizer = mocker.Mock()
        mock_tokenizer.return_value = {
            "input_ids": [[1] * 16],
            "attention_mask": [[1] * 16]
        }
        mocker.patch(
            'perscit_model.shared.data_loader.AutoTokenizer.from_pretrained',
            return_value=mock_tokenizer
        )

        loader = SharedDataLoader(max_length=16)
        long_text = " ".join(["word"] * 100)
        result = loader.tokenize_text(long_text)

        # Check tokenizer was called with truncation=True
        mock_tokenizer.assert_called_once()
        call_kwargs = mock_tokenizer.call_args[1]
        assert call_kwargs['truncation'] is True

    def test_tokenize_text_padding(self, mocker):
        """Test that padding is set correctly."""
        # Mock tokenizer callable
        mock_tokenizer = mocker.Mock()
        mock_tokenizer.return_value = {
            "input_ids": [[1] * 32],
            "attention_mask": [[1] * 32]
        }
        mocker.patch(
            'perscit_model.shared.data_loader.AutoTokenizer.from_pretrained',
            return_value=mock_tokenizer
        )

        loader = SharedDataLoader(max_length=32)
        result = loader.tokenize_text("short")

        # Check tokenizer was called with padding="max_length"
        mock_tokenizer.assert_called_once()
        call_kwargs = mock_tokenizer.call_args[1]
        assert call_kwargs['padding'] == "max_length"

    def test_tokenize_text_returns_tensors(self, mocker):
        """Test that return_tensors is set to pt."""
        # Mock tokenizer callable
        mock_tokenizer = mocker.Mock()
        mock_tokenizer.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]]
        }
        mocker.patch(
            'perscit_model.shared.data_loader.AutoTokenizer.from_pretrained',
            return_value=mock_tokenizer
        )

        loader = SharedDataLoader()
        result = loader.tokenize_text("test")

        # Check tokenizer was called with return_tensors="pt"
        mock_tokenizer.assert_called_once()
        call_kwargs = mock_tokenizer.call_args[1]
        assert call_kwargs['return_tensors'] == "pt"

    def test_trim_offset_whitespace_leading_space(self):
        """Test trimming offset with leading whitespace."""
        text = "cf. Thuc. III.38"
        # Offset (3, 7) represents " Thu" in text
        start, end = SharedDataLoader.trim_offset_whitespace(text, 3, 7)

        assert start == 4  # After the space
        assert end == 7
        assert text[start:end] == "Thu"

    def test_trim_offset_whitespace_trailing_space(self):
        """Test trimming offset with trailing whitespace."""
        text = "word   next"
        # Offset (0, 7) represents "word   " (with trailing spaces)
        start, end = SharedDataLoader.trim_offset_whitespace(text, 0, 7)

        assert start == 0
        assert end == 4  # After "word", before spaces
        assert text[start:end] == "word"

    def test_trim_offset_whitespace_both_sides(self):
        """Test trimming offset with both leading and trailing whitespace."""
        text = "text  word  next"
        # Offset (4, 11) represents "  word  "
        start, end = SharedDataLoader.trim_offset_whitespace(text, 4, 11)

        assert start == 6  # After leading spaces
        assert end == 10   # Before trailing spaces
        assert text[start:end] == "word"

    def test_trim_offset_whitespace_no_whitespace(self):
        """Test trimming offset with no whitespace."""
        text = "cf.Thuc"
        # Offset (3, 7) represents "Thuc" (no spaces)
        start, end = SharedDataLoader.trim_offset_whitespace(text, 3, 7)

        assert start == 3
        assert end == 7
        assert text[start:end] == "Thuc"

    def test_trim_offset_whitespace_all_whitespace(self):
        """Test trimming offset that contains only whitespace."""
        text = "word     next"
        # Offset (4, 9) represents "     " (all spaces)
        start, end = SharedDataLoader.trim_offset_whitespace(text, 4, 9)

        # Should return start == end (empty range)
        assert start == end
        assert start == 9

    def test_trim_offset_whitespace_newlines_and_tabs(self):
        """Test trimming offset with various whitespace characters."""
        text = "word\t\n  content  \t\nnext"
        # Offset (4, 17) represents "\t\n  content  \t\n"
        start, end = SharedDataLoader.trim_offset_whitespace(text, 4, 17)

        assert start == 8   # After "\t\n  "
        assert end == 15    # Before "  \t\n"
        assert text[start:end] == "content"

    def test_trim_offset_whitespace_empty_range(self):
        """Test trimming an empty offset range."""
        text = "some text"
        # Empty range (5, 5)
        start, end = SharedDataLoader.trim_offset_whitespace(text, 5, 5)

        assert start == 5
        assert end == 5

    def test_trim_offset_whitespace_real_tokenizer_example(self):
        """Test with a real example from DeBERTa tokenizer offset_mapping."""
        # Simulates: "cf. Thuc. III.38"
        # DeBERTa tokenizer gives offset (3, 7) for token "Thu" but includes leading space
        text = "cf. Thuc. III.38"
        start, end = SharedDataLoader.trim_offset_whitespace(text, 3, 7)

        # Should exclude the space at position 3
        assert text[start:end] == "Thu"
        assert start == 4
        assert end == 7
