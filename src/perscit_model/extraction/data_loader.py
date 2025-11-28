"""Data loader for tag extraction task."""

import multiprocessing
import warnings
from pathlib import Path
from typing import Callable, cast, Generator

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


class ExtractionDataLoader(SharedDataLoader):
    """Data loader for tag extraction task - only tokenizes xml_context."""

    special_tags = ["bibl", "quote", "cit"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_special_tokens(SPECIAL_TOKENS)

    def __call__(self, filepath: Path | str) -> Generator[dict, None, None]:
        """
        Load and tokenize data for tag extraction.

        Args:
            filepath: Path to JSONL file

        Yields:
            Dicts with xml_context and filename
        """
        for item in self.load_jsonl(filepath):
            yield {
                "xml_context": item["xml_context"],
                "filename": item.get("filename", ""),
            }

    @classmethod
    def parse_xml_to_bio(cls, xml_context: str) -> str:
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
        for tag in soup.find_all(cls.special_tags):
            tag.attrs = {}

        # Reconstruct repaired XML
        cleaned_xml = str(soup)

        # Replace citation tags with special tokens (with spaces for tokenization)
        for tag, token in zip(SPECIAL_TAGS, SPECIAL_TOKENS):
            cleaned_xml = cleaned_xml.replace(tag, f" {token} ")

        return cleaned_xml

    def generate_bio_labels(self, input_ids: list[int]) -> list[int]:
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
        # Address typing issues with Pyright and HuggingFace tokenizer
        annotated_convert_tokens_to_ids = cast(
            Callable, self.tokenizer.convert_tokens_to_ids
        )
        # Get special token IDs
        special_token_ids = {
            annotated_convert_tokens_to_ids("[BIBL_START]"): ("BIBL", "start"),
            annotated_convert_tokens_to_ids("[BIBL_END]"): ("BIBL", "end"),
            annotated_convert_tokens_to_ids("[QUOTE_START]"): ("QUOTE", "start"),
            annotated_convert_tokens_to_ids("[QUOTE_END]"): ("QUOTE", "end"),
            annotated_convert_tokens_to_ids("[CIT_START]"): ("CIT", "start"),
            annotated_convert_tokens_to_ids("[CIT_END]"): ("CIT", "end"),
        }

        labels = []
        current_tag = None  # None, "BIBL", "QUOTE", or "CIT"
        first_token_of_tag = False

        for token_id in input_ids:
            # Check if it's a special token (CLS, SEP, PAD)
            if token_id in [
                self.tokenizer.cls_token_id,
                self.tokenizer.sep_token_id,
                self.tokenizer.pad_token_id,
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

    def strip_special_tokens_and_align_labels(
        self, input_ids: list[int], labels: list[int]
    ) -> tuple[list[int], list[int]]:
        """
        Remove special citation tokens from input while keeping labels aligned.

        During training, we use special tokens ([BIBL_START], etc.) to generate labels,
        but we don't want the model to see them in the input. This function removes
        the special tokens while keeping the remaining labels aligned.

        Args:
            input_ids: Token IDs including special citation tokens
            labels: BIO labels (with -100 for special tokens)

        Returns:
            Tuple of (clean_input_ids, aligned_labels) without special tokens
        """
        # Address typing issues with Pyright and HuggingFace tokenizer
        annotated_convert_tokens_to_ids = cast(
            Callable, self.tokenizer.convert_tokens_to_ids
        )

        special_token_ids = {
            annotated_convert_tokens_to_ids("[BIBL_START]"),
            annotated_convert_tokens_to_ids("[BIBL_END]"),
            annotated_convert_tokens_to_ids("[QUOTE_START]"),
            annotated_convert_tokens_to_ids("[QUOTE_END]"),
            annotated_convert_tokens_to_ids("[CIT_START]"),
            annotated_convert_tokens_to_ids("[CIT_END]"),
        }

        clean_input_ids = []
        aligned_labels = []

        for token_id, label in zip(input_ids, labels):
            # Skip special citation tokens
            if token_id not in special_token_ids:
                clean_input_ids.append(token_id)
                aligned_labels.append(label)

        return clean_input_ids, aligned_labels


def create_extraction_dataset(
    jsonl_path: Path | str,
    config_path: Path | str | None = None,
    num_proc: int | None = None,
) -> Dataset:
    """
    Create a HuggingFace Dataset for BIO tag extraction.

    Pipeline:
    1. Parse XML and convert tags to special tokens ([BIBL_START], etc.)
    2. Tokenize text WITH special tokens
    3. Generate BIO labels based on special token positions
    4. STRIP special tokens from input (so model doesn't see the answer)
    5. Align labels with cleaned input
    6. Create Dataset with clean input_ids, attention_mask, labels, filename

    This ensures the model learns to predict citation boundaries from context alone,
    not from seeing the special tokens that mark the boundaries.

    Args:
        jsonl_path: Path to JSONL file with xml_context field
        config_path: Optional path to YAML config
        num_proc: Optional number of processes for parallel tokenization
        (1 = sequential),
        defaults to number of threads available on system

    Returns:
        HuggingFace Dataset with tokenized inputs and BIO labels (no special tokens in input)
    """
    # Suppress tokenizer warning about byte fallback in fast tokenizers
    # This warning appears once per process in parallel tokenization
    warnings.filterwarnings(
        "ignore",
        message=".*byte fallback.*",
        category=UserWarning,
        module="transformers.convert_slow_tokenizer",
    )

    loader = ExtractionDataLoader(config_path=config_path)

    # if parallel requested
    if num_proc is None:
        num_proc = multiprocessing.cpu_count()

    def path_loader():
        return loader(jsonl_path)

    dataset = cast(Dataset, Dataset.from_generator(path_loader))

    def process_entries(entries: dict[str, list]) -> dict[str, list]:
        # Extract data from BatchEncoding (shape is [1, seq_len])
        extraction_entries = [
            {
                "xml_context": loader.tokenize_text(
                    ExtractionDataLoader.parse_xml_to_bio(entry_content)
                ),
                "filename": entry_filename,
            }
            for entry_content, entry_filename in zip(
                entries["xml_context"], entries["filename"]
            )
        ]

        input_ids_with_special = [
            entry["xml_context"].input_ids[0].tolist() for entry in extraction_entries
        ]

        # Generate BIO labels from special token positions
        labels_with_special = [
            loader.generate_bio_labels(entry_input_ids)
            for entry_input_ids in input_ids_with_special
        ]

        # Strip special tokens from input and align labels
        cleaned_data = [
            loader.strip_special_tokens_and_align_labels(ids, labs)
            for ids, labs in zip(input_ids_with_special, labels_with_special)
        ]

        input_ids = [clean_ids for clean_ids, _ in cleaned_data]
        labels = [aligned_labs for _, aligned_labs in cleaned_data]

        # Rebuild attention masks for cleaned inputs (all 1s up to sequence length)
        attention_mask = [[1] * len(ids) for ids in input_ids]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "filename": entries["filename"],
        }

    msg = "Tokenizing and labelling tokens"
    return dataset.map(
        process_entries, num_proc=num_proc, batched=True, batch_size=1000, desc=msg
    )
