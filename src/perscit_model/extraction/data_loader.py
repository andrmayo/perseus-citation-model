"""Data loader for tag extraction task."""

import logging
import multiprocessing
import random
import re
import warnings
from pathlib import Path
from typing import Callable, Generator, cast

from datasets import Dataset

from perscit_model.shared.data_loader import SharedDataLoader

logger = logging.getLogger(__name__)

SPECIAL_TAGS = ["<bibl>", "</bibl>", "<quote>", "</quote>"]
SPECIAL_TOKENS = [
    "[BIBL_START]",
    "[BIBL_END]",
    "[QUOTE_START]",
    "[QUOTE_END]",
]

# BIO label definitions
# Note: CIT tags are not included - they are structural wrappers in source XML
# that should not be predicted by the model
BIO_LABELS = ["O", "B-BIBL", "I-BIBL", "B-QUOTE", "I-QUOTE"]
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
            # Handle both xml_context (snippets) and window_text (full doc windows)
            if "xml_context" in item:
                content = item["xml_context"]
            elif "window_text" in item:
                content = item["window_text"]
            else:
                raise KeyError(
                    f"Expected 'xml_context' or 'window_text' field in data, got: {list(item.keys())}"
                )

            yield {
                "xml_context": content,
                "filename": item.get("filename", ""),
            }

    @classmethod
    def parse_xml_to_bio(
        cls, xml_content: str, tag_retention_prob: float | None = None
    ) -> str:
        """
        Replace XML citation tags with special tokens for DeBERTa tokenizer.

        Pipeline:
        1. Strip all <cit> tags (they are structural wrappers we don't predict)
        2. If tag_retention_prob == 0.0:
            - Strip attributes from all bibl/quote tags
            - Convert all bibl/quote tags to special tokens
        3. If tag_retention_prob > 0.0 (Phase 3):
            - For each bibl/quote tag pair, randomly decide:
              * Keep: preserve original tag WITH attributes
              * Convert: strip attributes and replace with special tokens

        Converts citation tags to special tokens:
        - <bibl> → [BIBL_START]
        - </bibl> → [BIBL_END]
        - <quote> → [QUOTE_START]
        - </quote> → [QUOTE_END]

        Note: <cit> tags are NOT converted - they are completely stripped from the
        training data. CIT tags are structural wrappers in source documents that
        should not be predicted by the model.

        Other tags (e.g., <title>, <author>) are preserved in the output.

        The special tokens should be added to the DeBERTa tokenizer's vocabulary
        so they won't be split during tokenization. BIO labels are then generated
        in the dataset creation step based on positions relative to these markers.

        Note on tags orphaned in excerpting:
            This regex-based approach doesn't validate or repair XML structure, so
            orphaned tags in excerpts will be converted directly to markers. This may
            result in unpaired markers (e.g., [BIBL_START] without [BIBL_END]), which
            the model must learn to be robust to. The only kind of invalid XML this
            method handles properly is XML malformed by excerpting.

        Args:
            xml_context: XML snippet (may be malformed from excerpting)
            tag_retention_prob: Probability (0.0-1.0) that a bibl/quote tag pair is kept
                as-is with attributes instead of converted to special tokens.
                Default 0.0 means convert all tags (Phase 1 & 2 behavior).

        Returns:
            - processed_text: Text with bibl/quote tags replaced by special tokens
                (and some tags optionally kept as-is if tag_retention_prob > 0).
                All <cit> tags are stripped.
        """
        if tag_retention_prob is None:
            tag_retention_prob = 0.0

        # FIRST: Always strip all <cit> tags (with or without attributes)
        # These are structural wrappers we never want in training data
        xml_content = re.sub(r"<cit(?:\s+[^>]*)?>", "", xml_content)
        xml_content = re.sub(r"</cit>", "", xml_content)

        if tag_retention_prob == 0.0:
            # Strip attributes from bibl/quote tags
            cleaned_xml = re.sub(r"<(bibl|quote)\s+[^>]*>", r"<\1>", xml_content)

            # Replace bibl/quote tags with special tokens
            for tag, token in zip(SPECIAL_TAGS, SPECIAL_TOKENS):
                cleaned_xml = cleaned_xml.replace(tag, token)

            return cleaned_xml

        # Phase 3: Randomly keep some bibl/quote tags, convert others
        # Note: Only processing bibl/quote now (cit already stripped above)
        pattern = r"<(/?)(bibl|quote)(?:\s+[^>]*)?>"

        tag_matches = []
        for match in re.finditer(pattern, xml_content):
            is_closing = bool(match.group(1))
            tag_name = match.group(2)
            full_tag = match.group(0)
            tag_matches.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "is_closing": is_closing,
                    "tag_name": tag_name,
                    "full_tag": full_tag,
                    "keep": None,  # decide when we match pairs
                }
            )

        # match opening and closing tags into pairs using a stack
        stack = []
        for tag_info in tag_matches:
            if not tag_info["is_closing"]:
                # Opening tag: make random decisions and push to stack
                tag_info["keep"] = random.random() < tag_retention_prob
                stack.append(tag_info)
            else:
                # Closing tag: find matching opening tag and use same decision
                if stack and stack[-1]["tag_name"] == tag_info["tag_name"]:
                    tag_info["keep"] = stack[-1]["keep"]
                    stack.pop()
                else:  # orphaned closing tag
                    tag_info["keep"] = random.random() < tag_retention_prob

        # Build results by replacing tags based on keep decision
        result = []
        last_end = 0
        for tag_info in tag_matches:
            prev_end, last_end = last_end, tag_info["end"]
            if tag_info["keep"]:
                # Keep the original tag with attributes - no replacement needed
                result.extend(
                    [xml_content[prev_end : tag_info["start"]], tag_info["full_tag"]]
                )
                continue
            tag_name = tag_info["tag_name"]
            is_closing = tag_info["is_closing"]
            spec_token = (
                f"[{tag_name.upper()}_END]"
                if is_closing
                else f"[{tag_name.upper()}_START]"
            )
            result.extend([xml_content[prev_end : tag_info["start"]], spec_token])
        result.append(xml_content[last_end:])

        return "".join(result)

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
        # Get special token IDs (only BIBL and QUOTE - CIT tags are stripped from training data)
        special_token_ids = {
            annotated_convert_tokens_to_ids("[BIBL_START]"): ("BIBL", "start"),
            annotated_convert_tokens_to_ids("[BIBL_END]"): ("BIBL", "end"),
            annotated_convert_tokens_to_ids("[QUOTE_START]"): ("QUOTE", "start"),
            annotated_convert_tokens_to_ids("[QUOTE_END]"): ("QUOTE", "end"),
        }

        labels = []
        current_tag = None  # None, "BIBL", or "QUOTE"
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

        # Only BIBL and QUOTE special tokens (CIT tags are stripped from training data)
        special_token_ids = {
            annotated_convert_tokens_to_ids("[BIBL_START]"),
            annotated_convert_tokens_to_ids("[BIBL_END]"),
            annotated_convert_tokens_to_ids("[QUOTE_START]"),
            annotated_convert_tokens_to_ids("[QUOTE_END]"),
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
    tag_retention_prob: float | None = None,
) -> Dataset:
    """
    Create a HuggingFace Dataset for BIO tag extraction.

    Pipeline:
    1. Parse XML and convert tags to special tokens ([BIBL_START], etc.)
       - For Phase 3: Randomly keep some tags as-is with attributes based on tag_retention_prob
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
        tag_retention_prob: Probability (0.0-1.0) of keeping citation tags as-is
        instead of converting to special tokens. Default 0.0 (Phase 1 & 2 behavior).
        For Phase 3, use 0.3-0.5 to teach model to handle existing citations.

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
        # This is a bit hacky, but is meant to deal with different data format
        if "xml_context" in entries.keys():
            xml_key = "xml_context"
        elif "window_text" in entries.keys():
            xml_key = "window_text"
        else:
            raise KeyError
        extraction_entries = [
            {
                "xml_string": loader.tokenize_text(
                    ExtractionDataLoader.parse_xml_to_bio(
                        entry_content, tag_retention_prob=tag_retention_prob
                    )
                ),
                "filename": entry_filename,
            }
            for entry_content, entry_filename in zip(
                entries[xml_key], entries["filename"]
            )
        ]

        input_ids_with_special = [
            entry["xml_string"].input_ids[0].tolist() for entry in extraction_entries
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
