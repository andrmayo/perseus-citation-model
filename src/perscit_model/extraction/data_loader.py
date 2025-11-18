"""Data loader for tag extraction task."""

from dataclasses import dataclass
from pathlib import Path
from typing import cast, Generator

import transformers
from bs4 import BeautifulSoup
from datasets import Dataset

from perscit_model.shared.data_loader import SharedDataLoader


SPECIAL_TAGS = ["<bibl>", "</bibl>", "<quote>", "</quote>", "<cit>", "</cit>"]
SPECIAL_TOKENS = [
    "[BIBL_START]",
    "[BIBL_END]",
    "[QUOTE_START]",
    "[QUOTE_END]",
    "[CIT_START]",
    "[CIT_END]",
]

# BIO label definitions
BIO_LABELS = ["O", "B-BIBL", "I-BIBL", "B-QUOTE", "I-QUOTE", "B-CIT", "I-CIT"]
LABEL2ID = {label: idx for idx, label in enumerate(BIO_LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(BIO_LABELS)}


@dataclass
class ExtractionData:
    """Tokenized data for tag extraction task."""

    xml_context: transformers.BatchEncoding
    filename: str


class ExtractionDataLoader(SharedDataLoader):
    """Data loader for tag extraction task - only tokenizes xml_context."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_special_tokens(SPECIAL_TOKENS)

    def __call__(self, filepath: Path | str) -> Generator[ExtractionData, None, None]:
        """
        Load and tokenize data for tag extraction.

        Args:
            filepath: Path to JSONL file

        Yields:
            ExtractionData instances with tokenized xml_context
        """
        for item in self.load_jsonl(filepath):
            cleaned_text = parse_xml_to_bio(item["xml_context"])
            yield ExtractionData(
                xml_context=self.tokenize_text(cleaned_text),
                filename=item.get("filename", ""),
            )


def parse_xml_to_bio(xml_context: str) -> str:
    """
    Replace XML citation tags with special tokens for DeBERTa tokenizer.

    Pipeline:
    1. Parse with BeautifulSoup to repair malformed XML
    2. Remove attributes from citation tags (bibl, quote, cit)
    3. Reconstruct repaired XML string
    4. Replace citation tags with special tokens (surrounded by spaces)

    Converts citation tags to special tokens:
    - <bibl> → [BIBL_START]
    - </bibl> → [BIBL_END]
    - <quote> → [QUOTE_START]
    - </quote> → [QUOTE_END]
    - <cit> → [CIT_START]
    - </cit> → [CIT_END]

    Other tags (e.g., <title>, <author>) are preserved in the output.

    The special tokens should be added to the DeBERTa tokenizer's vocabulary
    so they won't be split during tokenization. BIO labels are then generated
    in the dataset creation step based on positions relative to these markers.

    Note on nested tags:
        For nested structures like <cit><bibl>text</bibl></cit>, both sets of
        markers will be present: [CIT_START] [BIBL_START] text [BIBL_END] [CIT_END].
        The model learns to handle these nested structures.

    Note on tags orphaned in excerpting:
        BeautifulSoup's lxml parser will handle malformed XML by ignoring orphaned
        closing tags and auto-closing orphaned opening tags. This may result in
        missing or extra markers, which the model must learn to be robust to.

    Args:
        xml_context: XML snippet (may be malformed from excerpting)

    Returns:
        - processed_text: Text with citation tags replaced by special tokens
    """

    # Parse with BeautifulSoup to repair malformed XML
    # and remove attributes from citation tags
    soup = BeautifulSoup(xml_context, "lxml")
    for tag in soup.find_all(["bibl", "quote", "cit"]):
        tag.attrs = {}

    # Reconstruct repaired XML
    cleaned_xml = str(soup)

    # Replace citation tags with special tokens (with spaces for tokenization)
    for tag, token in zip(SPECIAL_TAGS, SPECIAL_TOKENS):
        cleaned_xml = cleaned_xml.replace(tag, f" {token} ")

    return cleaned_xml


def generate_bio_labels(input_ids: list[int], tokenizer) -> list[int]:
    """
    Generate BIO labels from tokenized input containing special tokens.

    Tracks state as we scan through tokens:
    - When we see [TAG_START], we enter that tag
    - First real token after [TAG_START] gets B-TAG
    - Subsequent tokens get I-TAG
    - When we see [TAG_END], we exit that tag
    - Outside any tag: O
    - Special tokens (CLS, SEP, PAD, and our markers): -100

    Args:
        input_ids: List of token IDs from tokenizer
        tokenizer: The tokenizer (to get special token IDs)

    Returns:
        List of label IDs (same length as input_ids)
    """
    # Get special token IDs
    special_token_ids = {
        tokenizer.convert_tokens_to_ids("[BIBL_START]"): ("BIBL", "start"),
        tokenizer.convert_tokens_to_ids("[BIBL_END]"): ("BIBL", "end"),
        tokenizer.convert_tokens_to_ids("[QUOTE_START]"): ("QUOTE", "start"),
        tokenizer.convert_tokens_to_ids("[QUOTE_END]"): ("QUOTE", "end"),
        tokenizer.convert_tokens_to_ids("[CIT_START]"): ("CIT", "start"),
        tokenizer.convert_tokens_to_ids("[CIT_END]"): ("CIT", "end"),
    }

    labels = []
    current_tag = None  # None, "BIBL", "QUOTE", or "CIT"
    first_token_of_tag = False

    for token_id in input_ids:
        # Check if it's a special token (CLS, SEP, PAD)
        if token_id in [
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id,
        ]:
            labels.append(-100)
            continue

        # Check if it's one of our custom special tokens
        if token_id in special_token_ids:
            tag_type, position = special_token_ids[token_id]
            if position == "start":
                current_tag = tag_type
                first_token_of_tag = True
            else:  # position == "end"
                current_tag = None
                first_token_of_tag = False
            labels.append(-100)  # Special tokens get -100
            continue

        # Regular token - assign BIO label based on state
        if current_tag is None:
            labels.append(LABEL2ID["O"])
        elif first_token_of_tag:
            labels.append(LABEL2ID[f"B-{current_tag}"])
            first_token_of_tag = False
        else:
            labels.append(LABEL2ID[f"I-{current_tag}"])

    return labels


def create_extraction_dataset(
    jsonl_path: Path | str,
    config_path: Path | str | None = None,
) -> Dataset:
    """
    Create a HuggingFace Dataset for BIO tag extraction.

    Pipeline:
    1. Load data using ExtractionDataLoader (handles XML parsing, tokenization)
    2. Generate BIO labels from special token positions
    3. Create Dataset with input_ids, attention_mask, labels, filename

    Args:
        jsonl_path: Path to JSONL file with xml_context field
        config_path: Optional path to YAML config

    Returns:
        HuggingFace Dataset with tokenized inputs and BIO labels
    """
    loader = ExtractionDataLoader(config_path=config_path)

    def generate_entries() -> Generator[dict]:
        for entry in loader(jsonl_path):
            # Extract data from BatchEncoding (shape is [1, seq_len])
            input_ids = entry.xml_context.input_ids[0].tolist()
            attention_mask = entry.xml_context.attention_mask[0].tolist()

            # Generate BIO labels from special token positions
            labels = generate_bio_labels(input_ids, loader.tokenizer)

            yield {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "filename": entry.filename,
            }

    # Create HuggingFace Dataset
    return cast(Dataset, Dataset.from_generator(generate_entries))
