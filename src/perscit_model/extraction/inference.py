import inspect
import sys
from pathlib import Path
from typing import Callable, cast

from perscit_model.extraction.model import load_model_from_checkpoint
from perscit_model.extraction.data_loader import (
    ExtractionDataLoader,
    parse_xml_to_bio,
    ID2LABEL,
)

import torch
from transformers import AutoModelForTokenClassification

MODEL_TRAIN_DIR = Path(__file__).parent.parent.parent.parent / "outputs" / "extraction"
MODEL_SAVE_DIR = (
    Path(__file__).parent.parent.parent.parent / "outputs" / "models" / "extraction"
)


def get_model(path: str | Path | None = None, last_trained=False):
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


def predict(text: str, **kwargs):
    loader_args = {
        k: v
        for k, v in kwargs.items()
        if k in inspect.signature(ExtractionDataLoader).parameters
    }
    model = cast(Callable, get_model())
    pred_args = {k: v for k, v in kwargs.items() if k not in loader_args}
    loader = ExtractionDataLoader(**loader_args)
    inputs = loader.tokenize_text(text)
    with torch.no_grad():
        outputs = model(**inputs, **pred_args)
    logits = outputs.logits
    predictions = logits.argmax(dim=-1).squeeze().tolist()
    labels = [ID2LABEL[p] for p in predictions]
    return labels
