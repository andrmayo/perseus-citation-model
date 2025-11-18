"""Data loader for URN resolution task."""

from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import transformers

from perscit_model.shared.data_loader import SharedDataLoader


@dataclass
class ResolutionData:
    """Tokenized data for URN resolution task."""

    bibl: transformers.BatchEncoding
    ref: transformers.BatchEncoding
    urn: str
    quote: str


class ResolutionDataLoader(SharedDataLoader):
    """Data loader for URN resolution task - tokenizes citation text fields."""

    def __call__(self, filepath: Path | str) -> Generator[ResolutionData, None, None]:
        """
        Load and tokenize data for URN resolution.

        Args:
            filepath: Path to JSONL file

        Yields:
            ResolutionData instances with tokenized bibl and ref
        """
        for item in self.load_jsonl(filepath):
            yield ResolutionData(
                bibl=self.tokenize_text(item["bibl"]),
                ref=self.tokenize_text(item["ref"]),
                urn=item["urn"],
                quote=item.get("quote", ""),
            )
