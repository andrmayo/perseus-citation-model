"""Model definition for citation extraction (token classification as <bibl> or <quote>)."""

from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from safetensors import safe_open
from torch import ByteTensor, Tensor
from torch.nn import Embedding
from torch.nn.utils.rnn import pad_sequence
from torchcrf import CRF
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForTokenClassification,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import TokenClassifierOutput

from perscit_model.extraction.data_loader import (
    BIO_LABELS,
    ID2LABEL,
    LABEL2ID,
    SPECIAL_TOKENS,
)
from perscit_model.shared.data_loader import DEFAULT_CONFIG
from perscit_model.shared.training_utils import TrainingConfig


class TokenClassificationWithCRF(PreTrainedModel):
    """Generic transformer model with CRF layer for token classification.

    This model adds a CRF layer on top of any AutoModel encoder to enforce valid BIO
    tag transitions. Written mainly with DeBERTa in mind.
    """

    def __init__(self, config: PretrainedConfig, batch_first: bool = True):
        """
        Args:
            config: most straightforwardly, use AutoConfig.from_pretrained(), providing
                model_path, num_labels, id2label, and label2id
            batch_first: for intializing CRF, True if first dimension of encoder outputs
                is batch size
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        # Load the base encoder
        self.encoder = AutoModel.from_config(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=batch_first)

        # Initialize weights
        self.post_init()

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        labels: Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Pass through transformer (encoder)
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # CRF requires first token to be valid (not -100)
            # Extract only valid positions, pad them, then compute CRF loss

            # Find valid positions (vectorized)
            valid_mask = labels != -100  # [batch, seq_len]

            # Extract valid tokens for each sequence (still need loop for variable lengths)
            batch_logits = [
                logits[i][valid_mask[i]]
                for i in range(logits.size(0))
                if valid_mask[i].any()
            ]
            batch_labels = [
                labels[i][valid_mask[i]]
                for i in range(labels.size(0))
                if valid_mask[i].any()
            ]

            if batch_logits:
                # Pad sequences (transpose because pad_sequence expects [seq_len, batch, ...])
                padded_logits = pad_sequence(
                    batch_logits, batch_first=True, padding_value=0.0
                )
                padded_labels = pad_sequence(
                    batch_labels, batch_first=True, padding_value=0
                )

                # Create mask efficiently
                lengths = torch.tensor(
                    [len(seq) for seq in batch_logits], device=logits.device
                )
                max_len = padded_logits.size(1)
                padded_mask = torch.arange(max_len, device=logits.device).expand(
                    len(lengths), max_len
                ) < lengths.unsqueeze(1)

                # Now first position is always valid - CRF requirement satisfied
                loss = -self.crf(
                    padded_logits,
                    padded_labels,
                    mask=padded_mask,
                    reduction="token_mean",
                )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def decode(self, logits: Tensor, attention_mask: Tensor) -> list[list[int]]:
        """Decode using Viterbi algorithm to get best tag sequence.

        Args:
            logits: model output logits[batch_size, seq_len, num_labels]
            attention_mask: attention mask[batch_size, seq_len]

        Returns:
            List of best tag sequences
        """

        # crf.decode uses legacy tensor types
        if attention_mask is not None and not isinstance(attention_mask, ByteTensor):
            attention_mask = torch.ByteTensor(attention_mask)
        return self.crf.decode(logits, mask=attention_mask)

    def get_input_embeddings(self):
        """Get input embeddings from the encoder."""
        return self.encoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        """Set input embeddings in the encoder."""
        self.encoder.set_input_embeddings(value)


def create_model(
    tokenizer: PreTrainedTokenizerBase,
    config_path: Path | str | None = None,
    pretrained_model_path: Path | str | None = None,
    use_crf: bool = True,
) -> PreTrainedModel:
    """Create a token classification model for citation extraction from Perseus XML documents.

    Loads model_name from YAML config file specified in shared/data_loader.py.
        ARGS:
            tokenizer: Tokenizer with special tokens (so embeddings can be resized)
            config_path: Path to YAML config file (default None will load DEFAULT_CONFIG)
            pretrained_model_path: Optional path to a pretrained model checkpoint to initialize from.
                If provided, loads weights from this checkpoint instead of the base model.
                Useful for curriculum learning (load Phase 1 weights but start Phase 2 training from epoch 0).
            use_crf: If True, add CRF layer on top of encoder model

        Returns:
            Token classification model with correct label mappings of subtype AutoModelForTokenClassification.
    """

    # Load config
    if config_path is None:
        config_path = DEFAULT_CONFIG
    config = TrainingConfig.from_yaml(config_path)

    # Determine model path to load from
    if pretrained_model_path is not None:
        # Load from pretrained checkpoint
        model_path = str(pretrained_model_path)
    else:
        # Load from base model
        model_path = config.model_name

    if use_crf:
        model_config = AutoConfig.from_pretrained(
            model_path,
            num_labels=len(BIO_LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )

        model = TokenClassificationWithCRF.from_pretrained(
            model_path,
            config=model_config,
            ignore_mismatched_sizes=True,
        )
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            num_labels=len(BIO_LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,  # In case of checkpoint size mismatch
        )

    # Resize embeddings to include special tokens
    # Use len(tokenizer) instead of vocab_size since vocab_size doesn't include added tokens
    # Do this only if loading from base model (not from pretrained checkpoint which already has special tokens)
    if pretrained_model_path is None:
        model.resize_token_embeddings(len(tokenizer))

        # Set the new embeddings to mean of existing embeddings
        # This is more to stabilize loss early on in training
        # than to improve final performance

        with torch.no_grad():
            embeddings_module = model.get_input_embeddings()
            if embeddings_module is None:
                raise RuntimeError(
                    """Model has no input embeddings layer. This should not happen after
                    resize_token_embeddings() was called. Check model architecture."""
                )
            # Cast to Embedding to satisfy type checker
            embeddings = cast(Embedding, embeddings_module)
            old_embeddings = embeddings.weight[: -len(SPECIAL_TOKENS), :]
            mean_embedding = old_embeddings.mean(dim=0)
            embeddings.weight[-len(SPECIAL_TOKENS) :, :] = mean_embedding

    return model


def load_model_from_checkpoint(
    checkpoint_path: Path | str,
) -> TokenClassificationWithCRF | AutoModelForTokenClassification:
    """
    Load a trained model from a checkpoint.
    Detects whether model is encoder+CRF or just encoder.

    Args:
        checkpoint_path: Path to saved model checkpoint

    Returns:
        Loaded token classification model
    """

    if isinstance(checkpoint_path, str):
        checkpoint_path = Path(checkpoint_path)

    # Try to find the weights file
    safetensors_path = checkpoint_path / "model.safetensors"
    pytorch_path = checkpoint_path / "pytorch_model.bin"

    if not (safetensors_path.exists() or pytorch_path.exists()):
        raise FileNotFoundError(
            f"load_model_from_checkpoint() found no tensor file at {checkpoint_path}"
        )

    has_crf = False
    if safetensors_path.exists():
        with safe_open(safetensors_path, framework="pt") as f:
            keys = list(f.keys())
            has_crf = any("crf." in key for key in keys)
    elif pytorch_path.exists():
        # check .bin file
        state_dict = torch.load(pytorch_path, map_location="cpu")
        has_crf = any("crf." in key for key in state_dict.keys())

    if has_crf:
        return TokenClassificationWithCRF.from_pretrained(checkpoint_path)
    return AutoModelForTokenClassification.from_pretrained(checkpoint_path)
