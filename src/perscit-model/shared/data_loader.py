import json
from pathlib import Path

import transformers
from datasets import Dataset
from transformers import AutoTokenizer


def load_jsonl(filepath: Path) -> list[dict]:
    data = []
    with open(filepath) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def get_deberta_tokenizer() -> transformers.DebertaTokenizer:
    return AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")


def tokenize_text(
    text: str, tokenizer: transformers.PreTrainedTokenizerBase, max_length: int = 512
):
    return tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
