import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Callable, cast

import torch
import transformers
from torch import IntTensor

from perscit_model.extraction.data_loader import (
    ID2LABEL,
    ExtractionDataLoader,
)
from perscit_model.extraction.model import load_model_from_checkpoint

MODEL_TRAIN_DIR = Path(__file__).parent.parent.parent.parent / "outputs" / "extraction"
MODEL_SAVE_DIR = (
    Path(__file__).parent.parent.parent.parent / "outputs" / "models" / "extraction"
)


class InferenceModel:
    def __init__(
        self,
        model_path: str | Path | None = None,
        last_trained=False,
        device: str | None = None,
        **kwargs,
    ):
        self.model = cast(
            torch.nn.Module, InferenceModel.load_model(model_path, last_trained)
        )
        self.loader = ExtractionDataLoader(**kwargs)

        # Move model to GPU if available
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

    @staticmethod
    def load_model(
        path: str | Path | None = None, last_trained=False
    ) -> transformers.AutoModelForTokenClassification:
        def check_path(p: str | Path) -> Path:
            if isinstance(p, str):
                try:
                    p = Path(p)
                except Exception as e:
                    print(f"Error parsing string {p} as path: {e}", file=sys.stderr)
                    sys.exit(1)
            if not p.exists() and p.is_dir():
                raise FileNotFoundError(f"No directory found at {p}")
            return p

        if last_trained:
            if path is None:
                path = MODEL_TRAIN_DIR

            path = check_path(path)
            model_dirs = list(path.glob("final-model*"))

            if not model_dirs:
                raise FileNotFoundError(
                    f"No final models with directory name 'final-model*' found at {path}"
                )

            model_path = max(model_dirs, key=lambda x: x.stat().st_mtime)

        else:
            if path is None:
                path = MODEL_SAVE_DIR
            model_path = check_path(path)
        return load_model_from_checkpoint(model_path)

    def insert_tags_into_xml(
        self,
        xml: str | list[str],
        encoding: transformers.BatchEncoding,
        labels: list[str] | list[list[str]],
        existing_citations: None | list[tuple[int, int, str, str]] = None,
    ) -> list[str] | str:
        """
        Convert input text and predicted BIO labels to XML-tagged text.
        Handles arbitrary numbers of text inputs as lists.

        Args:
            xml: (list of) string representation(s) of xml
            encoding: BatchEncoding object with offset_mapping and token ids
            labels: list of labels from prediction, or list of lists for batch
            existing_citations: irrelevant if None, otherwise list of (start, stop, label)
                for citations already known from original XML that need to be interwoven
                with predictions. Or can take a list of tuples of ((start, stop, label), attrs)
                where attrs is a string with the attributes for the corresponding citation element.
                This could get rewritten as a more elegant polymorphism.

        Returns:
            A string or list of strings of texts with XML tags (<bibl>, <quote>, <cit>) inserted from prediction
        """
        try:
            offset_mapping = cast(IntTensor, encoding["offset_mapping"])
        except KeyError:
            msg = """
                BatchEncoding object passed to xml must have 'offset_mappping' key.
                Make sure that tokenizer call has return_offsets_mapping=True.
            """
            raise KeyError(msg)

        if isinstance(xml, str):
            tokens = cast(IntTensor, encoding["input_ids"])[0]
            if isinstance(labels, list) and isinstance(labels[0], list):
                labels = labels[0]
            try:
                tokens = cast(IntTensor, tokens)
                offset = offset_mapping[0]
                with_tags = self._insert_tags(
                    xml,
                    tokens,
                    cast(IntTensor, offset),
                    cast(list[str], labels),
                    existing_citations,
                )
            except Exception as e:
                raise e

            return with_tags

        def int_tens(x):
            return cast(IntTensor, x)

        tokens = int_tens(encoding["input_ids"])
        try:
            # Use 'spawn' instead of 'fork' to avoid CUDA initialization issues
            with ProcessPoolExecutor(mp_context=multiprocessing.get_context('spawn')) as executor:
                with_tags = list(
                    executor.map(
                        self._insert_tags,
                        xml,
                        tokens,
                        offset_mapping,
                        labels,
                        repeat(existing_citations),
                    )
                )
        except Exception as e:
            raise e
        return with_tags

    def _insert_tags(
        self,
        xml: str,
        tokens: IntTensor,
        offset_mapping: IntTensor,
        labels: list[str],
        existing_citations: None | list[tuple[int, int, str, str]],
    ) -> str:
        """Inserts XML tags for a single text."""

        special_token_ids = {
            getattr(self.loader.tokenizer, "cls_token_id", None),
            getattr(self.loader.tokenizer, "sep_token_id", None),
            getattr(self.loader.tokenizer, "pad_token_id", None),
        }

        special_token_ids.discard(None)

        # filter tokens
        token_spans = [
            (offset[0].item(), offset[1].item(), label)
            for token_id, offset, label in zip(tokens, offset_mapping, labels)
            if token_id.item() not in special_token_ids
            and offset[0].item() != offset[1].item()
        ]

        # build entities according to citation labels
        entities = []
        current_entity = None

        # add in existing citations first, if provided
        # this assumes existing_citations is already sorted
        prior_entities = []
        if existing_citations:
            prior_entities = [
                {"type": tag_type.upper(), "start": start, "end": end, "attrs": attrs}
                for start, end, tag_type, attrs in existing_citations
            ]

        # so can we keep track whether to skip a prediction because it overlaps with existing citation
        citation_idx = 0  # pointer into (sorted)
        n_existing_citations = len(prior_entities)
        skipping_predicted_entity = False

        for start, end, label in token_spans:
            while (
                citation_idx < n_existing_citations
                and start >= prior_entities[citation_idx]["end"]
            ):
                citation_idx += 1
            # Extract label type
            label_type = label[2:] if label.startswith(("B-", "I-")) else None

            should_skip = False

            # check overlap or contiguity between existing citation and prediction
            if citation_idx < n_existing_citations:
                prior_start, prior_end, prior_type = (
                    prior_entities[citation_idx]["start"],
                    prior_entities[citation_idx]["end"],
                    prior_entities[citation_idx]["type"],
                )

                overlap = start < prior_end and end > prior_start
                if overlap:
                    should_skip = True

                # contiguity check
                if label_type and prior_end == start and label_type == prior_type:
                    should_skip = True

            # Also check contiguity with previous citation tagged in XML already
            if citation_idx > 0 and label_type:
                prev_end = prior_entities[citation_idx - 1]["end"]
                prev_type = prior_entities[citation_idx - 1]["type"]
                if prev_end == start and label_type == prev_type:
                    should_skip = True

            if should_skip:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                # Mark that we're now skipping a predicted entity
                if label.startswith("B-"):
                    skipping_predicted_entity = True
                continue

            if skipping_predicted_entity:
                if label == "O" or label.startswith("B-"):
                    skipping_predicted_entity = False
                else:
                    continue

            if label == "O":
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
            elif label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "type": label[2:],
                    "start": start,
                    "end": end,
                    "attrs": "",
                }
            elif label.startswith("I-"):
                if current_entity:
                    current_entity["end"] = end
                else:
                    current_entity = {
                        "type": label[2:],
                        "start": start,
                        "end": end,
                        "attrs": "",
                    }

        if current_entity:
            entities.append(current_entity)

        entities += prior_entities
        entities.sort(key=lambda x: x["start"])

        # Trim leading/trailing whitespace from entity boundaries
        # (tokenizer offset_mapping includes spaces in ranges)
        for entity in entities:
            start, end = entity["start"], entity["end"]
            start, end = ExtractionDataLoader.trim_offset_whitespace(xml, start, end)
            entity["start"] = start
            entity["end"] = end

        # build text segments
        segments = []
        last_pos = 0

        for entity in entities:
            if last_pos < entity["start"]:
                segments.append(xml[last_pos : entity["start"]])

            tag = entity["type"].lower()
            # entity["attrs"] should already include any necessary whitespace
            attr_string = entity["attrs"]
            segments.append(
                f"<{tag}{attr_string}>{xml[entity['start'] : entity['end']]}</{tag}>"
            )
            last_pos = entity["end"]

        if last_pos < len(xml):
            segments.append(xml[last_pos:])

        return "".join(segments)

    def process_text(
        self, text: str, **kwargs
    ) -> tuple[transformers.BatchEncoding, list[str]]:
        """
        Run inference on plain text (without <bibl>, <cit>, or <quote> tags).

        Args:
            text: Plain text without XML citation tags
            **kwargs: Additional arguments for model/loader

        Returns:
            Tuple of tokens as BatchEncoding and List of BIO labels for each token,

        Note:
            Input should be plain text. If you have XML with tags, strip them first.
            The model predicts where <bibl>, <quote>, and <cit> tags should be.
        """

        # Tokenize plain text (no special token conversion during inference)
        inputs = self.loader.tokenize_text(text, return_offsets_mapping=True)
        labels = self.predict(inputs, **kwargs)

        return inputs, labels

    def process_batch(
        self, texts: list[str], batch_size: int = 32, **kwargs
    ) -> list[tuple[transformers.BatchEncoding, list[str]]]:
        """
        Process multiple texts in batches (efficient GPU utilization).

        Args:
            texts: list of plain text strings
            batch_size: number of texts to process at once
            **kwargs: additional arguments for model

        Returns:
            List of (encoding, labels) tuples for each text
        """

        results = []
        max_length = getattr(
            getattr(self.model, "config", None), "max_position_embeddings", None
        )

        if max_length is None:
            raise AttributeError(
                "Unable to get max embedding size from model with model.config.max_position_embeddings"
            )

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            inputs = self.loader.tokenizer(
                batch,
                padding=True,
                return_tensors="pt",
                return_offsets_mapping=True,
            )

            seq_length = cast(torch.Tensor, inputs["input_ids"]).shape[1]
            if seq_length > max_length:
                msg = f"""
                Input text too long: {seq_length} tokens (max: {max_length}).
                Split your text into smaller chunks.
                """
                raise ValueError(msg)

            # Move inputs to device and run prediction
            inputs_on_device = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
                if k != "offset_mapping"
            }
            call_model = cast(Callable, self.model)
            with torch.no_grad():
                outputs = call_model(**inputs_on_device, **kwargs)

            # Decode predictions
            predictions = outputs.logits.argmax(dim=-1).tolist()
            labels_batch = [[ID2LABEL[p] for p in preds] for preds in predictions]

            # Store results
            for j, labels in enumerate(labels_batch):
                single_encoding = {k: v[j : j + 1] for k, v in inputs.items()}
                results.append((transformers.BatchEncoding(single_encoding), labels))

        return results

    def predict(
        self,
        inputs: transformers.BatchEncoding,
        **kwargs,
    ) -> list[str]:
        """
        Run inference on plain text (without <bibl>, <cit>, or <quote> tags).

        Args:
            inputs: BatchEncoding object with input_ids.
            **kwargs: Additional arguments for model/loader

        Returns:
            List of BIO labels for each token

        Note:
            Input should be tokenized text. If you have XML with tags, strip them first.
            The model predicts where <bibl>, <quote>, and <cit> tags should be.
        """
        # Move inputs to device
        inputs_on_device = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
            if k != "offset_mapping"
        }
        call_model = cast(Callable, self.model)
        with torch.no_grad():
            outputs = call_model(**inputs_on_device, **kwargs)
        logits = outputs.logits
        predictions = logits.argmax(dim=-1).squeeze().tolist()
        labels = [ID2LABEL[p] for p in predictions]
        return labels
