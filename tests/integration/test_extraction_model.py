"""Unit tests for extraction model creation and configuration.

Note: These tests use real models (not mocks) because we need to test actual embedding resizing.
They will be slower but more accurate.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from perscit_model.extraction.model import create_model, load_model_from_checkpoint
from perscit_model.extraction.data_loader import BIO_LABELS, LABEL2ID, ID2LABEL, SPECIAL_TOKENS, ExtractionDataLoader


@pytest.fixture(scope="module")
def loader():
    """Create a data loader once for all tests in this module."""
    return ExtractionDataLoader()


@pytest.fixture(scope="module")
def model(loader):
    """Create a model once for all tests in this module."""
    return create_model(loader.tokenizer)


class TestCreateModel:
    """Test suite for create_model function."""

    def test_model_has_correct_num_labels(self, model):
        """Test that model is created with correct number of labels."""
        assert model.config.num_labels == len(BIO_LABELS)

    def test_model_has_label_mappings(self, model):
        """Test that model has correct label2id and id2label mappings."""
        assert model.config.label2id == LABEL2ID
        assert model.config.id2label == ID2LABEL

    def test_model_embeddings_resized(self, model, loader):
        """Test that model embeddings are resized for special tokens."""
        embedding_size = model.get_input_embeddings().weight.shape[0]
        assert embedding_size == loader.tokenizer.vocab_size

    def test_new_embeddings_initialized_to_mean(self, model):
        """Test that new token embeddings are initialized to mean of existing embeddings."""
        embeddings = model.get_input_embeddings().weight

        # Get the last len(SPECIAL_TOKENS) embeddings (the new ones)
        new_embeddings = embeddings[-len(SPECIAL_TOKENS):, :]

        # Get old embeddings (everything except new ones)
        old_embeddings = embeddings[:-len(SPECIAL_TOKENS), :]

        # Calculate mean of old embeddings
        expected_mean = old_embeddings.mean(dim=0)

        # All new embeddings should be equal to this mean
        for i in range(len(SPECIAL_TOKENS)):
            # Use allclose for floating point comparison
            assert torch.allclose(new_embeddings[i], expected_mean, rtol=1e-5, atol=1e-7)

    def test_all_new_embeddings_identical(self, model):
        """Test that all new special token embeddings are identical (all set to same mean)."""
        embeddings = model.get_input_embeddings().weight
        new_embeddings = embeddings[-len(SPECIAL_TOKENS):, :]

        # All new embeddings should be identical
        first_embedding = new_embeddings[0]
        for i in range(1, len(SPECIAL_TOKENS)):
            assert torch.allclose(new_embeddings[i], first_embedding)

    def test_embeddings_are_trainable(self, model):
        """Test that new embeddings are not frozen (requires_grad=True)."""
        embeddings = model.get_input_embeddings().weight
        assert embeddings.requires_grad is True

    def test_new_embeddings_have_correct_dimension(self, model, loader):
        """Test that new embeddings have same dimension as existing embeddings."""
        embeddings = model.get_input_embeddings().weight
        vocab_size, hidden_dim = embeddings.shape

        # Check that we have the right vocab size
        assert vocab_size == loader.tokenizer.vocab_size

        # Check that hidden dim is reasonable (DeBERTa base is 768)
        assert hidden_dim > 0

    def test_model_is_in_eval_mode_by_default(self, model):
        """Test that model is created in eval mode by default."""
        assert model.training is False


class TestLoadModelFromCheckpoint:
    """Test suite for load_model_from_checkpoint function."""

    def test_load_from_checkpoint_returns_model(self, loader, tmp_path):
        """Test that loading from checkpoint returns a model."""
        # Create and save a model
        model = create_model(loader.tokenizer)

        checkpoint_path = tmp_path / "test_checkpoint"
        model.save_pretrained(checkpoint_path)

        # Load from checkpoint
        loaded_model = load_model_from_checkpoint(checkpoint_path)

        assert loaded_model is not None
        assert loaded_model.config.num_labels == len(BIO_LABELS)

    def test_loaded_model_has_same_config(self, loader, tmp_path):
        """Test that loaded model preserves configuration."""
        # Create and save a model
        model = create_model(loader.tokenizer)

        checkpoint_path = tmp_path / "test_checkpoint"
        model.save_pretrained(checkpoint_path)

        # Load from checkpoint
        loaded_model = load_model_from_checkpoint(checkpoint_path)

        # Should have same label mappings
        assert loaded_model.config.label2id == LABEL2ID
        assert loaded_model.config.id2label == ID2LABEL
        assert loaded_model.config.num_labels == len(BIO_LABELS)
