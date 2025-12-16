"""Model definition for citation extraction (token classification as <cit>, <quote>, or <bibl>)."""

from pathlib import Path

import torch
from transformers import (
    AutoModelForTokenClassification,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from perscit_model.extraction.data_loader import (
    BIO_LABELS,
    ID2LABEL,
    LABEL2ID,
    SPECIAL_TOKENS,
)
from perscit_model.shared.data_loader import DEFAULT_CONFIG
from perscit_model.shared.training_utils import TrainingConfig


def create_model(
    tokenizer: PreTrainedTokenizerBase,
    config_path: Path | str | None = None,
) -> PreTrainedModel:
    """Create a token classification model for citation extraction from Perseus XML documents.

    Loads model_name from YAML config file specified in shared/data_loader.py.
        ARGS:
            tokenizer: Tokenizer with special tokens (so embeddings can be resized)
            config_path: Path to YAML config file (default None will load DEFAULT_CONFIG)

        Returns:
            Token classification model with correct label mappings of subtype AutoModelForTokenClassification.
    """

    # Load config
    if config_path is None:
        config_path = DEFAULT_CONFIG
    config = TrainingConfig.from_yaml(config_path)

    model = AutoModelForTokenClassification.from_pretrained(
        config.model_name,
        num_labels=len(BIO_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,  # In case of checkpoint size mismatch
    )

    # Resize embeddings to include special tokens
    # Use len(tokenizer) instead of vocab_size since vocab_size doesn't include added tokens
    model.resize_token_embeddings(len(tokenizer))

    # Set the new embeddings to mean of existing embeddings
    # This is more to stabilize loss early on in training
    # than to improve final performance

    with torch.no_grad():
        old_embeddings = model.get_input_embeddings().weight[: -len(SPECIAL_TOKENS), :]
        mean_embedding = old_embeddings.mean(dim=0)
        model.get_input_embeddings().weight[-len(SPECIAL_TOKENS) :, :] = mean_embedding

    return model


def load_model_from_checkpoint(
    checkpoint_path: Path | str,
) -> AutoModelForTokenClassification:
    """
    Load a trained model from a checkpoint.

    Args:
        checkpoint_path: Path to saved model checkpoint

    Returns:
        Loaded token classification model
    """
    return AutoModelForTokenClassification.from_pretrained(checkpoint_path)
