"""Data loader for tag extraction task."""

from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import transformers

from perscit_model.shared.data_loader import SharedDataLoader


@dataclass
class ExtractionData:
    """Tokenized data for tag extraction task."""

    xml_context: transformers.BatchEncoding
    filename: str


class ExtractionDataLoader(SharedDataLoader):
    """Data loader for tag extraction task - only tokenizes xml_context."""

    def __call__(self, filepath: Path | str) -> Generator[ExtractionData, None, None]:
        """
        Load and tokenize data for tag extraction.

        Args:
            filepath: Path to JSONL file

        Yields:
            ExtractionData instances with tokenized xml_context
        """
        for item in self.load_jsonl(filepath):
            yield ExtractionData(
                xml_context=self.tokenize_text(item["xml_context"]),
                filename=item.get("filename", ""),
            )
