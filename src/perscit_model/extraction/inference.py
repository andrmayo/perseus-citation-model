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

MODEL_DIR = Path(__file__).parent.parent.parent.parent / "outputs" / "extraction"


def get_model(path: str | Path = MODEL_DIR):
    if isinstance(path, str):
        try:
            path = Path(path)
        except Exception as e:
            print(f"Error parsing string {path} as path: {e}", file=sys.stderr)
            sys.exit(1)
    if not path.exists() and path.is_dir():
        raise FileNotFoundError(f"No directory found at {path}")

    model_dirs = path.glob("final-model*")

    if not model_dirs:
        raise FileNotFoundError(
            f"No final models with directory name 'final-model*' found at {path}"
        )

    latest_model_path = max(model_dirs, key=lambda x: x.stat().st_mtime)

    return load_model_from_checkpoint(latest_model_path)


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
