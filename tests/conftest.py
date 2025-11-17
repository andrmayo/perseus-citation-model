"""Shared pytest fixtures and configuration."""

import pytest


@pytest.fixture
def mock_tokenizer(mocker):
    """Create a mock tokenizer that returns realistic BatchEncoding-like data."""
    mock_tok = mocker.Mock()

    # Configure the mock to return dict with input_ids and attention_mask
    def tokenize_side_effect(text, **kwargs):
        max_length = kwargs.get('max_length', 512)
        # Simple mock: return list of token IDs based on text length
        num_tokens = min(len(text.split()), max_length)
        return {
            "input_ids": [[1] * num_tokens + [0] * (max_length - num_tokens)],
            "attention_mask": [[1] * num_tokens + [0] * (max_length - num_tokens)]
        }

    mock_tok.side_effect = tokenize_side_effect

    # Mock AutoTokenizer.from_pretrained to return our mock
    mocker.patch(
        'perscit_model.shared.data_loader.AutoTokenizer.from_pretrained',
        return_value=mock_tok
    )

    return mock_tok
