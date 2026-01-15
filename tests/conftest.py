"""Shared pytest fixtures and configuration."""

from pathlib import Path

import pytest


# Cache for the real tokenizer (loaded once per test session)
_cached_tokenizer = None


@pytest.fixture(scope="session")
def cached_tokenizer():
    """Load the real DeBERTa tokenizer once and cache it for all tests.

    This avoids the ~2-3 second overhead of loading the tokenizer for each test.
    """
    global _cached_tokenizer
    if _cached_tokenizer is None:
        from transformers import AutoTokenizer

        from perscit_model.extraction.data_loader import SPECIAL_TOKENS
        from perscit_model.shared.data_loader import DEFAULT_CONFIG
        from perscit_model.shared.training_utils import TrainingConfig

        config = TrainingConfig.from_yaml(DEFAULT_CONFIG)
        _cached_tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        _cached_tokenizer.add_tokens(SPECIAL_TOKENS, special_tokens=True)

    return _cached_tokenizer


@pytest.fixture
def mock_tokenizer(mocker):
    """Create a mock tokenizer that returns realistic BatchEncoding-like data."""
    import torch

    mock_tok = mocker.Mock()

    # Configure the mock to return object with both dict and attribute access
    def tokenize_side_effect(text, **kwargs):
        max_length = kwargs.get('max_length', 512)
        # Simple mock: return list of token IDs based on text length
        num_tokens = min(len(text.split()), max_length)

        # Create a mock BatchEncoding that supports both dict and attribute access
        mock_encoding = mocker.Mock()
        input_ids_tensor = torch.tensor([[1] * num_tokens + [0] * (max_length - num_tokens)])
        attention_mask_tensor = torch.tensor([[1] * num_tokens + [0] * (max_length - num_tokens)])

        data_dict = {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor
        }

        # Support dict-style access
        mock_encoding.__getitem__ = lambda self, key: data_dict[key]
        mock_encoding.__contains__ = lambda self, key: key in data_dict

        # Support attribute-style access
        mock_encoding.input_ids = input_ids_tensor
        mock_encoding.attention_mask = attention_mask_tensor

        return mock_encoding

    mock_tok.side_effect = tokenize_side_effect

    # Mock AutoTokenizer.from_pretrained to return our mock
    mocker.patch(
        'perscit_model.shared.data_loader.AutoTokenizer.from_pretrained',
        return_value=mock_tok
    )

    return mock_tok


@pytest.fixture
def use_cached_tokenizer(mocker, cached_tokenizer):
    """Patch AutoTokenizer.from_pretrained to return the cached tokenizer.

    Use this fixture for tests that need the real tokenizer behavior but want
    to avoid the overhead of loading it multiple times.
    """
    mocker.patch(
        "perscit_model.shared.data_loader.AutoTokenizer.from_pretrained",
        return_value=cached_tokenizer,
    )
    return cached_tokenizer


@pytest.fixture(scope="module")
def real_tagger():
    """Create a CitationTagger instance with a real model.

    This is expensive to load, so we use module scope to share it across tests.
    Tests using this fixture will actually run inference.
    Uses CUDA if available for better performance, otherwise falls back to CPU.
    """
    import torch
    from perscit_model.xml_processing.tagger import CitationTagger

    model_path = Path(__file__).parent.parent / "outputs" / "models" / "extraction"

    if not model_path.exists():
        pytest.skip(f"Model not found at {model_path}")

    # Use CUDA if available for faster inference, otherwise CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tagger = CitationTagger(model_path=str(model_path), device=device)

    return tagger
