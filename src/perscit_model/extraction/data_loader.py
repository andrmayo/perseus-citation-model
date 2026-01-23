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

# All special tokens including CIT (used in preprocessing, stripped during training)
ALL_SPECIAL_TOKENS = [
    "[CIT_START]",
    "[CIT_END]",
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

    # Tags that model predicts (cit tags are structural wrappers, not predicted)
    special_tags = ["bibl", "quote"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add all special tokens including CIT (CIT tokens are stripped during training)
        self.add_special_tokens(ALL_SPECIAL_TOKENS)

    def __call__(self, filepath: Path | str) -> Generator[dict, None, None]:
        """
        Load and tokenize data for tag extraction.

        Args:
            filepath: Path to JSONL file

        Yields:
            Dicts with xml_context, tag_attributes (if present), and filename
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

            result = {
                "xml_context": content,
                "filename": item.get("filename", ""),
            }

            # Include tag_attributes if present (for Phase 3 training)
            if "tag_attributes" in item:
                result["tag_attributes"] = item["tag_attributes"]

            yield result

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

    @classmethod
    def process_special_tokens(
        cls,
        text: str,
        tag_attributes: list[str] | None = None,
        tag_retention_prob: float | None = None,
    ) -> str:
        """
        Process text that already has special tokens from preprocessing.

        This method handles window_text from cp_xml_with_cits.py which already has
        [CIT_START], [CIT_END], [BIBL_START], [BIBL_END], [QUOTE_START], [QUOTE_END].

        Pipeline:
        1. Strip all [CIT_START] and [CIT_END] tokens (structural wrappers, not predicted)
        2. If tag_retention_prob == 0.0: keep BIBL/QUOTE tokens as-is
        3. If tag_retention_prob > 0.0 (Phase 3): randomly restore some BIBL/QUOTE
           tags with their original attributes from tag_attributes

        Args:
            text: Text with special tokens already present
            tag_attributes: List of attribute strings aligned with special tokens
                in the text (empty string for closing tags or tags without attributes)
            tag_retention_prob: Probability (0.0-1.0) that a bibl/quote tag pair is
                restored to original XML format with attributes. Default 0.0.

        Returns:
            Processed text with CIT tokens stripped and BIBL/QUOTE tokens either
            kept as special tokens or restored to XML tags based on tag_retention_prob
        """
        if tag_retention_prob is None:
            tag_retention_prob = 0.0

        # Always strip CIT tokens (structural wrappers)
        text = text.replace("[CIT_START]", "")
        text = text.replace("[CIT_END]", "")

        # For Phase 2, just return with CIT stripped
        if tag_retention_prob == 0.0 or tag_attributes is None:
            return text

        # Phase 3: Randomly restore some BIBL/QUOTE tags with attributes
        # Find all special token positions and their attributes
        token_pattern = re.compile(r"\[(BIBL|QUOTE)_(START|END)\]")

        # Build list of matches with their attributes
        matches = []

        # Re-scan original text to get attribute alignment
        # (before CIT stripping, attributes include CIT)
        original_attr_idx = 0
        for token in ALL_SPECIAL_TOKENS:
            start = 0
            while True:
                # Need to search in original text pattern
                pos = text.find(token, start)
                if pos == -1:
                    break
                if token in ("[CIT_START]", "[CIT_END]"):
                    # CIT tokens were already stripped, skip their attributes
                    original_attr_idx += 1
                start = pos + len(token)

        # Now process the text with CIT already stripped
        # Re-find BIBL/QUOTE tokens
        matches = list(token_pattern.finditer(text))

        if not matches or tag_attributes is None:
            return text

        # Filter attributes to only BIBL/QUOTE (skip CIT attributes)
        # The original tag_attributes list has attributes for ALL tokens in order
        # We need to extract just BIBL/QUOTE attributes
        bibl_quote_attrs = []
        # Parse original text to identify which attributes belong to BIBL/QUOTE
        # Note: We need the original text before CIT stripping to align attributes
        # Since CIT tokens are stripped, we need to count them from tag_attributes length
        # Assumption: tag_attributes has one entry per special token in order
        for i, attr in enumerate(tag_attributes):
            # We can infer token type from attribute position
            # This is a simplification - ideally we'd track token types explicitly
            bibl_quote_attrs.append(attr)

        # Match opening and closing tags for paired decisions
        stack: list[tuple[int, str, bool]] = []  # (match_idx, tag_name, keep)
        keep_decisions: dict[int, bool] = {}

        for i, match in enumerate(matches):
            tag_name = match.group(1)  # BIBL or QUOTE
            position = match.group(2)  # START or END

            if position == "START":
                keep = random.random() < tag_retention_prob
                stack.append((i, tag_name, keep))
                keep_decisions[i] = keep
            else:  # END
                if stack and stack[-1][1] == tag_name:
                    # Matching closing tag - use same decision
                    _, _, keep = stack.pop()
                    keep_decisions[i] = keep
                else:
                    # Orphaned closing tag
                    keep_decisions[i] = random.random() < tag_retention_prob

        # Build result
        result_parts = []
        last_end = 0

        for i, match in enumerate(matches):
            result_parts.append(text[last_end : match.start()])

            tag_name = match.group(1).lower()  # bibl or quote
            position = match.group(2)  # START or END

            if keep_decisions.get(i, False) and i < len(bibl_quote_attrs):
                # Restore to XML tag with attributes
                attrs = bibl_quote_attrs[i]
                if position == "START":
                    if attrs:
                        result_parts.append(f"<{tag_name} {attrs}>")
                    else:
                        result_parts.append(f"<{tag_name}>")
                else:
                    result_parts.append(f"</{tag_name}>")
            else:
                # Keep as special token
                result_parts.append(match.group(0))

            last_end = match.end()

        result_parts.append(text[last_end:])
        return "".join(result_parts)

    @staticmethod
    def pad_and_mask_sequence(
        input_ids: list[int],
        labels: list[int],
        max_length: int,
        pad_token_id: int,
        loss_margin_fraction: float = 0.3,
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Pad sequence to max_length with centering and mask edge labels.

        Centers short sequences and masks labels outside the reliable center
        region to match inference behavior (sliding window with center-only predictions).

        The masking scheme (for max_length=512, loss_margin_fraction=0.0):
        - Masked edge: 128 tokens (max_length // 4)
        - Center with loss: 256 tokens (max_length // 2)
        - Masked edge: 128 tokens (max_length // 4)

        With loss_margin_fraction=0.5, extends loss into half the edge margin:
        - Masked edge: 64 tokens
        - Center with loss: 384 tokens
        - Masked edge: 64 tokens

        Args:
            input_ids: Token IDs to pad
            labels: BIO labels to pad and mask
            max_length: Target sequence length
            pad_token_id: Token ID to use for padding
            loss_margin_fraction: Fraction of edge margin to include in loss (0.0-1.0).
                0.0 = only center, 0.3 = extend into 30% of edge (default), 1.0 = full sequence

        Returns:
            Tuple of (padded_input_ids, padded_labels, attention_mask)
        """
        center_length = max_length // 2
        edge_margin = max_length // 4

        # Calculate how much of the edge margin to include in loss
        loss_margin = int(edge_margin * loss_margin_fraction)
        masked_margin = edge_margin - loss_margin

        seq_len = len(input_ids)

        # Truncate if necessary
        if seq_len > max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
            seq_len = max_length

        # Calculate padding (center the sequence)
        pad_len = max_length - seq_len
        front_pad = pad_len // 2 if pad_len > 0 else 0
        end_pad = pad_len - front_pad if pad_len > 0 else 0

        # Pad input ids
        padded_ids = [pad_token_id] * front_pad + input_ids + [pad_token_id] * end_pad

        # Pad labels (padding positions get -100)
        padded_labels = [-100] * front_pad + labels + [-100] * end_pad

        # Mask left edge: from end of front padding to masked_margin
        left_mask_end = min(masked_margin, front_pad + seq_len)
        if left_mask_end > front_pad:
            padded_labels[front_pad:left_mask_end] = [-100] * (left_mask_end - front_pad)

        # Mask right edge: from (masked_margin + center_length + 2*loss_margin) to end
        # This simplifies to: from (max_length - masked_margin) to end of real tokens
        right_mask_start = max_length - masked_margin
        right_mask_end = max_length - end_pad
        if right_mask_start < right_mask_end:
            padded_labels[right_mask_start:right_mask_end] = [-100] * (
                right_mask_end - right_mask_start
            )

        # Attention mask: 1 for real tokens, 0 for padding
        attn_mask = [0] * front_pad + [1] * seq_len + [0] * end_pad

        return padded_ids, padded_labels, attn_mask

    def generate_bio_labels(self, input_ids: list[int]) -> list[int]:
        """
        Generate BIO labels from tokenized input containing special tokens.

        Tracks state as we scan through tokens:
        - When we see [TAG_START], we enter that tag (BIBL/QUOTE only, not CIT)
        - First real token after [TAG_START] gets B-TAG
        - Subsequent tokens get I-TAG
        - When we see [TAG_END], we exit that tag
        - Outside any tag: O
        - Special tokens (CLS, SEP, PAD, CIT markers, and BIBL/QUOTE markers): -100

        Note: CIT tokens are structural wrappers and don't affect labeling.
        They are simply ignored (labeled -100 and will be stripped).

        Args:
            input_ids: List of token IDs from tokenizer

        Returns:
            List of label IDs (same length as input_ids)
        """
        # Address typing issues with Pyright and HuggingFace tokenizer
        annotated_convert_tokens_to_ids = cast(
            Callable, self.tokenizer.convert_tokens_to_ids
        )
        # BIBL and QUOTE tokens affect BIO labeling
        label_token_ids = {
            annotated_convert_tokens_to_ids("[BIBL_START]"): ("BIBL", "start"),
            annotated_convert_tokens_to_ids("[BIBL_END]"): ("BIBL", "end"),
            annotated_convert_tokens_to_ids("[QUOTE_START]"): ("QUOTE", "start"),
            annotated_convert_tokens_to_ids("[QUOTE_END]"): ("QUOTE", "end"),
        }
        # CIT tokens are stripped but don't affect labeling
        cit_token_ids = {
            annotated_convert_tokens_to_ids("[CIT_START]"),
            annotated_convert_tokens_to_ids("[CIT_END]"),
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

            # Check if it's a CIT token (structural wrapper, doesn't affect labeling)
            if token_id in cit_token_ids:
                labels.append(-100)
                continue

            # Check if it's a BIBL/QUOTE token (affects labeling)
            if token_id in label_token_ids:
                tag_type, position = label_token_ids[token_id]
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
        all citation special tokens (CIT, BIBL, QUOTE) while keeping the remaining
        labels aligned.

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

        # All citation special tokens to strip (CIT, BIBL, QUOTE)
        special_token_ids = {
            annotated_convert_tokens_to_ids("[CIT_START]"),
            annotated_convert_tokens_to_ids("[CIT_END]"),
            annotated_convert_tokens_to_ids("[BIBL_START]"),
            annotated_convert_tokens_to_ids("[BIBL_END]"),
            annotated_convert_tokens_to_ids("[QUOTE_START]"),
            annotated_convert_tokens_to_ids("[QUOTE_END]"),
        }

        clean_input_ids = []
        aligned_labels = []

        for token_id, label in zip(input_ids, labels):
            # Skip all citation special tokens
            if token_id not in special_token_ids:
                clean_input_ids.append(token_id)
                aligned_labels.append(label)

        return clean_input_ids, aligned_labels


def create_extraction_dataset(
    jsonl_path: Path | str,
    config_path: Path | str | None = None,
    num_proc: int | None = None,
    tag_retention_prob: float | None = None,
    loss_margin_fraction: float = 0.3,
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
            (1 = sequential), defaults to number of threads available on system
        tag_retention_prob: Probability (0.0-1.0) of keeping citation tags as-is
            instead of converting to special tokens. Default 0.0 (Phase 1 & 2 behavior).
            For Phase 3, use 0.3-0.5 to teach model to handle existing citations.
        loss_margin_fraction: Fraction of edge margin to include in loss (0.0-1.0).
            0.0 = only center, 0.3 = extend into 30% of edge (default), 0.5 = half edge

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

        # Check if data has special tokens (new format) or raw XML (old format)
        # by looking at first entry
        first_content = entries[xml_key][0] if entries[xml_key] else ""
        has_special_tokens = (
            "[CIT_START]" in first_content or "[BIBL_START]" in first_content
        )

        # Get tag_attributes if present (for Phase 3)
        tag_attrs_list = entries.get("tag_attributes", [None] * len(entries[xml_key]))

        extraction_entries = []
        for entry_content, entry_filename, tag_attrs in zip(
            entries[xml_key], entries["filename"], tag_attrs_list
        ):
            if has_special_tokens:
                # New format: special tokens already present, process them
                processed = ExtractionDataLoader.process_special_tokens(
                    entry_content,
                    tag_attributes=tag_attrs,
                    tag_retention_prob=tag_retention_prob,
                )
            else:
                # Old format: raw XML, convert to special tokens
                processed = ExtractionDataLoader.parse_xml_to_bio(
                    entry_content, tag_retention_prob=tag_retention_prob
                )
            extraction_entries.append(
                {
                    "xml_string": loader.tokenize_text(processed),
                    "filename": entry_filename,
                }
            )

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

        # Pad all sequences to max_length and mask edge labels
        pad_token_id = loader.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer has no pad_token_id configured")

        padded_input_ids = []
        padded_labels = []
        attention_mask = []

        for ids, labs in zip(input_ids, labels):
            p_ids, p_labs, attn = ExtractionDataLoader.pad_and_mask_sequence(
                ids, labs, loader.max_length, cast(int, pad_token_id),
                loss_margin_fraction=loss_margin_fraction,
            )
            padded_input_ids.append(p_ids)
            padded_labels.append(p_labs)
            attention_mask.append(attn)

        # Note: filename is excluded because DataCollatorForTokenClassification
        # cannot handle string columns, and remove_unused_columns may be False
        # when using augmented training datasets
        return {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "labels": padded_labels,
        }

    msg = "Tokenizing and labelling tokens"

    # Remove all original columns - only keep model-compatible tensor columns
    columns_to_remove = dataset.column_names

    return dataset.map(
        process_entries,
        num_proc=num_proc,
        batched=True,
        batch_size=1000,
        desc=msg,
        remove_columns=columns_to_remove,
    )


def create_augmented_extraction_dataset(
    jsonl_path: Path | str,
    config_path: Path | str | None = None,
    strip_tags: list[str] | None = None,
    tag_retention_prob: float = 0.5,
    token_deletion_prob: float = 0.05,
    loss_margin_fraction: float = 0.3,
) -> Dataset:
    """
    Create a dataset with on-the-fly augmentation using TrainingTransform.

    Unlike create_extraction_dataset() which preprocesses all data upfront,
    this function uses set_transform() for lazy evaluation. Each access to
    the dataset applies fresh random augmentation.

    Args:
        jsonl_path: Path to JSONL file with window_text or xml_context field
        config_path: Optional path to YAML config
        strip_tags: List of tag names to randomly strip (e.g., ["author", "title"])
        tag_retention_prob: Probability of keeping each specified tag (default 0.5)
        token_deletion_prob: Probability of deleting each token (default 0.1)
        loss_margin_fraction: Fraction of edge margin to include in loss (0.0-1.0).
            0.0 = only center, 0.3 = extend into 30% of edge (default), 0.5 = half edge

    Returns:
        HuggingFace Dataset with set_transform applied
    """
    if strip_tags is None:
        strip_tags = []

    loader = ExtractionDataLoader(config_path=config_path)

    # Load raw data without preprocessing
    def load_raw():
        for item in loader(jsonl_path):
            yield item

    dataset = cast(Dataset, Dataset.from_generator(load_raw))

    # Create transform and apply it
    transform = TrainingTransform(
        loader=loader,
        strip_tags=strip_tags,
        tag_retention_prob=tag_retention_prob,
        token_deletion_prob=token_deletion_prob,
        loss_margin_fraction=loss_margin_fraction,
    )

    dataset.set_transform(transform)

    logger.info(
        f"Created augmented dataset with {len(dataset)} examples "
        f"(strip_tags={strip_tags}, tag_retention={tag_retention_prob}, "
        f"token_deletion={token_deletion_prob}, loss_margin_fraction={loss_margin_fraction})"
    )

    return dataset


class TrainingTransform:
    """
    Transform applied at dataset access time for data augmentation.

    Augmentations:
    1. Random retention of specified tags (default to 50%).
    2. Random token deletion (defaults to 10%).
    """

    def __init__(
        self,
        loader: ExtractionDataLoader,
        strip_tags: list[str],
        tag_retention_prob: float = 0.5,
        token_deletion_prob: float = 0.05,
        loss_margin_fraction: float = 0.3,
    ):
        """
        Initialize the training transform.

        Args:
            loader: ExtractionDataLoader instance for tokenization and label generation.
            tag_retention_prob: Probability of keeping <author>, <title>, tags
            token_deletion_prob: Probability of deleting each token.
            loss_margin_fraction: Fraction of edge margin to include in loss (0.0-1.0).
        """
        self.loader = loader
        self.tag_retention_prob = tag_retention_prob
        self.token_deletion_prob = token_deletion_prob
        self.strip_tags = strip_tags
        self.loss_margin_fraction = loss_margin_fraction

        self._citation_token_ids: set[int] | None = None

    @property
    def citation_token_ids(self) -> set[int]:
        if self._citation_token_ids is None:
            convert = cast(Callable, self.loader.tokenizer.convert_tokens_to_ids)
            self._citation_token_ids = {
                convert(spec_tok) for spec_tok in ALL_SPECIAL_TOKENS
            }
        return self._citation_token_ids

    def process_internal_tags(self, text: str) -> str:
        """Randomly retain or strip specified tags"""

        for tag in self.strip_tags:
            opening_pattern = re.compile(f"<{tag}[^>]*>")
            closing_pattern = re.compile(f"</{tag}>")

            matches = list(opening_pattern.finditer(text))
            for match in reversed(matches):
                if random.random() > self.tag_retention_prob:
                    open_start, open_end = match.span()

                    closing_match = closing_pattern.search(text, open_end)
                    if closing_match:
                        text = (
                            text[: closing_match.start()] + text[closing_match.end() :]
                        )

                    text = text[:open_start] + text[open_end:]

        return text

    def random_token_deletion(
        self, input_ids: list[int], labels: list[int]
    ) -> tuple[list[int], list[int]]:
        """Randomly delete tokens while preserving structure.structure

        Preserves:
            - Special tokens(CLS, SEP, PAD)
            - B- labels (entity start) to maintain valid BIO sequences
        """

        new_ids = []
        new_labels = []

        special_ids = {
            cast(int, self.loader.tokenizer.cls_token_id)
            if isinstance(self.loader.tokenizer.cls_token_id, int)
            else None,
            cast(int, self.loader.tokenizer.sep_token_id)
            if isinstance(self.loader.tokenizer.sep_token_id, int)
            else None,
            cast(int, self.loader.tokenizer.pad_token_id)
            if isinstance(self.loader.tokenizer.pad_token_id, int)
            else None,
        }
        special_ids.discard(None)

        for token_id, label in zip(input_ids, labels):
            if token_id in special_ids:
                new_ids.append(token_id)
                new_labels.append(label)
                continue

            if label in ID2LABEL and ID2LABEL[label][0] == "B":
                new_ids.append(token_id)
                new_labels.append(label)
                continue

            if random.random() > self.token_deletion_prob:
                new_ids.append(token_id)
                new_labels.append(label)

        return new_ids, new_labels

    def pad_and_mask(
        self, input_ids: list[int], labels: list[int]
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Pad sequence to max_length and mask edge labels.

        Centers short sequences and masks labels outside the reliable center
        region + a small margin to approximate inference behavior.

        Returns:
            Tuple of (padded_input_ids, padded_labels, attention_mask)
        """
        pad_token_id = self.loader.tokenizer.pad_token_id
        if not isinstance(pad_token_id, int):
            raise ValueError("Tokenizer does not have a valid pad_token_id configured")

        return ExtractionDataLoader.pad_and_mask_sequence(
            input_ids, labels, self.loader.max_length, pad_token_id,
            loss_margin_fraction=self.loss_margin_fraction,
        )

    def __call__(self, examples: dict[str, list]) -> dict[str, list]:
        """Transform a batch of examples.

        Note: filename is intentionally excluded from output because
        DataCollatorForTokenClassification cannot handle string columns.
        """

        text_key = "window_text" if "window_text" in examples else "xml_context"

        results: dict[str, list] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

        for text in examples[text_key]:
            text = self.process_internal_tags(text)

            text = ExtractionDataLoader.process_special_tokens(text)

            encoding = self.loader.tokenize_text(text)
            input_ids = encoding.input_ids[0].tolist()

            labels = self.loader.generate_bio_labels(input_ids)

            input_ids, labels = self.loader.strip_special_tokens_and_align_labels(
                input_ids, labels
            )
            input_ids, labels = self.random_token_deletion(input_ids, labels)
            input_ids, labels, attn_mask = self.pad_and_mask(input_ids, labels)

            results["input_ids"].append(input_ids)
            results["attention_mask"].append(attn_mask)
            results["labels"].append(labels)

        return results
