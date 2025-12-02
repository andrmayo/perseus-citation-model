import sys
from pathlib import Path
from typing import Callable, cast

import transformers
from torch import IntTensor

from perscit_model.extraction.model import load_model_from_checkpoint
from perscit_model.extraction.data_loader import (
    ExtractionDataLoader,
    ID2LABEL,
)

import torch

MODEL_TRAIN_DIR = Path(__file__).parent.parent.parent.parent / "outputs" / "extraction"
MODEL_SAVE_DIR = (
    Path(__file__).parent.parent.parent.parent / "outputs" / "models" / "extraction"
)


class InferenceModel:
    def __init__(
        self, model_path: str | Path | None = None, last_trained=False, **kwargs
    ):
        self.model = cast(
            transformers.AutoModelForTokenClassification,
            InferenceModel.load_model(model_path, last_trained),
        )
        self.loader = ExtractionDataLoader(**kwargs)

    @staticmethod
    def load_model(path: str | Path | None = None, last_trained=False):
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
    ) -> list[str] | str:
        """
        Convert input text and predicted BIO labels to XML-tagged text.
        Handles arbitrary numbers of text inputs as lists.

        Args:
            encoding: BatchEncoding object with offset_mapping and token ids
            labels: list of labels from prediction

        Returns:
            List of strings of texts with XML tags (<bibl>, <quote>, <cit>) inserted from prediction
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
                    xml, tokens, cast(IntTensor, offset), cast(list[str], labels)
                )
            except Exception as e:
                raise e

            return with_tags

        def int_tens(x):
            return cast(IntTensor, x)

        tokens = int_tens(encoding["input_ids"])
        try:
            with_tags = [
                self._insert_tags(x, int_tens(tok), int_tens(om), cast(list[str], lbl))
                for x, tok, om, lbl in zip(xml, tokens, offset_mapping, labels)
            ]
        except Exception as e:
            raise e
        return with_tags

    def _insert_tags(
        self,
        xml: str,
        tokens: IntTensor,
        offset_mapping: IntTensor,
        labels: list[str],
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

        for start, end, label in token_spans:
            if label == "0":
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
            elif label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {"type": label[2:], "start": start, "end": end}
            elif label.startswith("I-"):
                if current_entity:
                    current_entity["end"] = end
                else:
                    current_entity = {"type": label[2:], "start": start, "end": end}

        if current_entity:
            entities.append(current_entity)

        # build text segments
        segments = []
        last_pos = 0

        for entity in entities:
            if last_pos < entity["start"]:
                segments.append(xml[last_pos : entity["start"]])

            tag = entity["type"].lower()
            segments.append(f"<{tag}>{xml[entity['start'] : entity['end']]}</{tag}>")
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

    def predict(
        self,
        inputs: transformers.BatchEncoding,
        **kwargs,
    ) -> list[str]:
        """
        Run inference on plain text (without <bibl>, <cit>, or <quote> tags).

        Args:
            text: Plain text without XML citation tags
            **kwargs: Additional arguments for model/loader

        Returns:
            List of BIO labels for each token

        Note:
            Input should be plain text. If you have XML with tags, strip them first.
            The model predicts where <bibl>, <quote>, and <cit> tags should be.
        """
        call_model = cast(Callable, self.model)
        with torch.no_grad():
            outputs = call_model(
                **{k: v for k, v in inputs.items() if k != "offset_mapping"}, **kwargs
            )
        logits = outputs.logits
        predictions = logits.argmax(dim=-1).squeeze().tolist()
        labels = [ID2LABEL[p] for p in predictions]
        return labels
