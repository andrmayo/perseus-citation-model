"""
This script does two things:
    1. It counts citation relevant tags (bibl, cit, quote) in a given dir,
        and copies all those with a count above a certain threshold to a target dir.
    2. It computes a few statistics about the citations, and prints them to stdout
        and a yaml file in project_root/outputs/metrics/xml_cit_stats.yaml.

    Args:
        src_dir: directory containing XML files
        target_dir (optional): directory to copy XML files that pass threshold test to
"""

import bisect
import json
import math
import random
import re
import sys
from io import BytesIO
from pathlib import Path
from typing import Callable, cast
from xml import sax

import yaml
from lxml import etree
from tqdm import tqdm
from torch import Tensor

from perscit_model.shared.data_loader import SharedDataLoader
from perscit_model.shared.training_utils import TrainingConfig
from perscit_model.shared.xml_utils import CitXMLHandler

config = TrainingConfig.from_yaml(
    Path(__file__).parent.parent / "configs/extraction/baseline.yaml"
)

random.seed(config.seed)

INCLUSION_THRESHOLD = 100
DEFAULT_OUTPUT = Path(__file__).parent.parent / "cit_data/xml_files"
WINDOW_SIZE = 512

cit_tag_retention_prob = 0.1

# Special tokens for citation tags (must match what's added to tokenizer)
SPECIAL_TOKENS = [
    "[CIT_START]",
    "[CIT_END]",
    "[BIBL_START]",
    "[BIBL_END]",
    "[QUOTE_START]",
    "[QUOTE_END]",
]

# Regex pattern for matching citation tags with optional attributes
# Captures: (1) optional slash for closing, (2) tag name, (3) attributes if present
TAG_PATTERN = re.compile(r"<(/?)(cit|bibl|quote)\b([^>]*)>")


def replace_tags_with_special_tokens(
    xml_text: str, retention_prob: float = 0.0
) -> tuple[str, list[str]]:
    """
    Replace citation tags with special tokens and extract attributes.

    Args:
        xml_text: XML text with citation tags
        retention_prob: Probability (0.0-1.0) of keeping bibl/quote tags as-is
            instead of converting to special tokens. CIT tags are always converted.
            When a tag is kept, both opening and closing tags are preserved.
            Handles nested tags correctly using a stack.

    Returns:
        Tuple of (processed_text, tag_attributes) where:
        - processed_text has tags replaced with special tokens
        - tag_attributes is a list of attribute strings in order of appearance,
            not including kept tags
            (empty string for tags without attributes or closing tags)
    """
    tag_attributes: list[str] = []
    result_parts: list[str] = []
    last_end = 0

    kept_open_tags: list[str] = []
    keep = False

    for match in TAG_PATTERN.finditer(xml_text):
        # Add text before this tag
        result_parts.append(xml_text[last_end : match.start()])

        is_closing = match.group(1) == "/"
        tag_name = match.group(2).upper()
        attrs = match.group(3).strip()
        last_end = match.end()

        # check if closing tag of kept open tag
        if is_closing and kept_open_tags and match.group(2) == kept_open_tags[-1]:
            keep = True
            kept_open_tags.pop()
        elif match.group(2) != "cit" and not is_closing:
            keep = random.random() < retention_prob
            if keep:
                kept_open_tags.append(match.group(2))

        if keep:
            keep = False
            result_parts.append(match.group(0))
            continue

        # Determine special token
        if is_closing:
            special_token = f"[{tag_name}_END]"
            # Closing tags have no attributes
            tag_attributes.append("")
        else:
            special_token = f"[{tag_name}_START]"
            # Store attributes (empty string if none)
            tag_attributes.append(attrs)

        result_parts.append(special_token)

    # Add remaining text after last tag
    result_parts.append(xml_text[last_end:])

    return "".join(result_parts), tag_attributes


def find_special_token_indices(
    text: str, special_tokens: list[str]
) -> list[tuple[int, int, str]]:
    """
    Find all special token positions in text.

    Returns:
        List of (start, end, token) tuples in order of appearance
    """
    indices = []
    for token in special_tokens:
        start = 0
        while True:
            pos = text.find(token, start)
            if pos == -1:
                break
            indices.append((pos, pos + len(token), token))
            start = pos + len(token)
    # Sort by position
    indices.sort(key=lambda x: x[0])
    return indices


def count_special_tokens_in_range(
    token_indices: list[tuple[int, int, str]], start_char: int, end_char: int
) -> int:
    """Count special tokens that fall within a character range."""
    count = 0
    for tok_start, _, _ in token_indices:
        # Token is in range if it starts within the range
        if start_char <= tok_start < end_char:
            count += 1
    return count


def get_attributes_in_range(
    token_indices: list[tuple[int, int, str]],
    all_attributes: list[str],
    start_char: int,
    end_char: int,
) -> list[str]:
    """Get attributes for special tokens that fall within a character range."""
    # Binary search to find first token in range
    starts = [t[0] for t in token_indices]
    left = bisect.bisect_left(starts, start_char)

    attrs = []
    for i in range(left, len(token_indices)):
        tok_start = token_indices[i][0]
        if tok_start >= end_char:
            break
        attrs.append(all_attributes[i])
    return attrs


if __name__ == "__main__":
    try:
        path = Path(sys.argv[1])
    except IndexError:
        raise IndexError("Please provide path to dir with XML files")

    if not path.exists():
        print("Please provide path to dir with XML files")
        raise FileNotFoundError

    if not path.is_dir():
        print("Please provide path to dir with XML files")
        raise NotADirectoryError

    # Set output path for json and jsonl files, in provided
    if len(sys.argv) >= 3:
        output_dir = Path(sys.argv[2])
    else:
        output_dir = DEFAULT_OUTPUT

    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "window_data.jsonl"

    saxHandler = CitXMLHandler()
    saxParser = sax.make_parser()
    saxParser.setContentHandler(saxHandler)

    files_for_training: list[Path] = []

    # only include mappings for included files
    file_to_idx = {}

    # retain more cit tags for training on dicts
    if "dict" in path.name:
        cit_tag_retention_prob = 0.2

    print(f"<bibl> and <quote> tags kept at rate of {cit_tag_retention_prob}")

    for i, xml_file in enumerate(path.glob("*.xml")):
        saxHandler.filename = str(xml_file)
        try:
            saxParser.parse(xml_file)
        except Exception as e:
            print(f"failed to parse {xml_file}: {e}")
            continue
        cit_elt_count = 0
        for k, v in saxHandler.counts.items():
            if k in ("cit", "bibl", "quote"):
                cit_elt_count += v
        print(f"file {xml_file} has {cit_elt_count} cit, bibl, or quote elements")
        if cit_elt_count >= INCLUSION_THRESHOLD:
            file_to_idx[str(xml_file)] = i
            print(f"including file {xml_file}")
            files_for_training.append(xml_file)
        else:
            print(f"excluding file {xml_file}")

    cit_prop_avg = 0
    bibl_prop_avg = 0
    quote_prop_avg = 0

    # proportion averages weighted by character count of doc
    cit_prop_w = 0
    bibl_prop_w = 0
    quote_prop_w = 0

    n_included = len(files_for_training)

    cit_counts = []
    bibl_counts = []
    quote_counts = []

    cit_total = 0
    bibl_total = 0
    quote_total = 0

    # for getting medians
    cit_props = []
    bibl_props = []
    quote_props = []
    totals = []

    n_chars = 0

    for idx in file_to_idx.values():
        n_chars += saxHandler.doc_counts[idx]["char_count"]

    for file, idx in file_to_idx.items():
        doc_counts = saxHandler.doc_counts[idx]
        tag_count = 0
        for k, v in doc_counts.items():
            if k in ("cit", "bibl", "quote"):
                tag_count += v

        cit_proportion = doc_counts["cit"] / tag_count
        bibl_proportion = doc_counts["bibl"] / tag_count
        quote_proportion = doc_counts["quote"] / tag_count

        cit_counts.append(doc_counts["cit"])
        bibl_counts.append(doc_counts["bibl"])
        quote_counts.append(doc_counts["quote"])

        cit_props.append(cit_proportion)
        bibl_props.append(bibl_proportion)
        quote_props.append(quote_proportion)

        cit_total += doc_counts["cit"]
        bibl_total += doc_counts["bibl"]
        quote_total += doc_counts["quote"]

        totals.append(tag_count)

        cit_prop_avg += cit_proportion
        bibl_prop_avg += bibl_proportion
        quote_prop_avg += quote_proportion

        char_weight = doc_counts["char_count"] / n_chars

        cit_prop_w += cit_proportion * char_weight
        bibl_prop_w += bibl_proportion * char_weight
        quote_prop_w += quote_proportion * char_weight

    cit_prop_avg /= n_included
    bibl_prop_avg /= n_included
    quote_prop_avg /= n_included

    mean_relevant_tags = sum(totals) / n_included

    totals.sort()

    cit_counts.sort()
    bibl_counts.sort()
    quote_counts.sort()

    cit_props.sort()
    bibl_props.sort()
    quote_props.sort()

    stats = {
        "n_included_files": n_included,
        "n_excluded_files": len(list(path.glob("*.xml"))) - n_included,
        "threshold": INCLUSION_THRESHOLD,
        "total_relevant_tags": sum(totals),
        "mean_relevant_tags": mean_relevant_tags,
        "median_relevant_tags": totals[n_included // 2]
        if n_included % 2 == 1
        else (totals[n_included // 2] + totals[n_included // 2 - 1]) / 2,
        "min_relevant_tags": min(totals),
        "max_relevant_tags": max(totals),
        "std_dev_relevant_tags": math.sqrt(
            sum((x - mean_relevant_tags) ** 2 for x in totals) / n_included
        ),
        "total_cit_tag_count": cit_total,
        "total_bibl_tag_count": bibl_total,
        "total_quote_tag_count": quote_total,
        "mean_cit_tag_count": cit_total / n_included,
        "mean_bibl_tag_count": bibl_total / n_included,
        "mean_quote_tag_count": quote_total / n_included,
        "median_cit_tag_count": cit_counts[n_included // 2]
        if n_included % 2 == 1
        else (cit_counts[n_included // 2] + cit_counts[n_included // 2 - 1]) / 2,
        "median_bibl_tag_count": bibl_counts[n_included // 2]
        if n_included % 2 == 1
        else (bibl_counts[n_included // 2] + bibl_counts[n_included // 2 - 1]) / 2,
        "median_quote_tag_count": quote_counts[n_included // 2]
        if n_included % 2 == 1
        else (quote_counts[n_included // 2] + quote_counts[n_included // 2 - 1]) / 2,
        "max_cit_tag_count": max(cit_counts),
        "min_cit_tag_count": min(cit_counts),
        "max_bibl_tag_count": max(bibl_counts),
        "min_bibl_tag_count": min(bibl_counts),
        "max_quote_tag_count": max(quote_counts),
        "min_quote_tag_count": min(quote_counts),
        "mean_cit_tag_proportion": cit_prop_avg,
        "mean_bibl_tag_proportion": bibl_prop_avg,
        "mean_quote_tag_proportion": quote_prop_avg,
        "median_cit_tag_proportion": cit_props[n_included // 2]
        if n_included % 2 == 1
        else (cit_props[n_included // 2] + cit_props[n_included // 2 - 1]) / 2,
        "median_bibl_tag_proportion": bibl_props[n_included // 2]
        if n_included % 2 == 1
        else (bibl_props[n_included // 2] + bibl_props[n_included // 2 - 1]) / 2,
        "median_quote_tag_proportion": quote_props[n_included // 2]
        if n_included % 2 == 1
        else (quote_props[n_included // 2] + quote_props[n_included // 2 - 1]) / 2,
        "max_cit_tag_proportion": max(cit_props),
        "min_cit_tag_proportion": min(cit_props),
        "max_bibl_tag_proportion": max(bibl_props),
        "min_bibl_tag_proportion": min(bibl_props),
        "max_quote_tag_proportion": max(quote_props),
        "min_quote_tag_proportion": min(quote_props),
        "char_weighted_cit_tag_proportion": cit_prop_w,
        "char_weighted_bibl_tag_proportion": bibl_prop_w,
        "char_weighted_quote_tag_proportion": quote_prop_w,
    }

    stats_output_path = output_dir / "xml_cit_stats.yaml"
    with open(stats_output_path, "w") as f:
        yaml.dump(stats, f, sort_keys=False)

    # Now, iterate through files_for_training
    # Add special tokens to tokenizer so they're treated as single tokens
    data_loader = SharedDataLoader(special_tokens=SPECIAL_TOKENS)

    # Get token IDs for special tokens to count them efficiently
    special_token_ids = set(
        cast(Callable, data_loader.tokenizer.convert_tokens_to_ids)(SPECIAL_TOKENS)
    )

    # Parser that doesn't try to load external DTDs or network entities
    parser = etree.XMLParser(load_dtd=False, no_network=True, recover=True)

    with open(output_path, "w") as f:
        for file in tqdm(files_for_training):
            xml_bytes = file.read_bytes()
            tree = etree.parse(BytesIO(xml_bytes), parser)
            encoding = tree.docinfo.encoding
            del tree
            xml_text = xml_bytes.decode(
                encoding=encoding if isinstance(encoding, str) else "utf-8"
            )
            del xml_bytes

            # Replace tags with special tokens and get attributes
            processed_text, all_attributes = replace_tags_with_special_tokens(
                xml_text, cit_tag_retention_prob
            )

            # Find special token positions in processed text (for attribute alignment)
            token_indices = find_special_token_indices(processed_text, SPECIAL_TOKENS)

            # Tokenize the processed text
            tokenization = data_loader.tokenizer(
                processed_text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            tokens = tokenization["input_ids"]
            offset_mapping = tokenization["offset_mapping"]

            assert isinstance(tokens, list) and (
                len(tokens) == 0 or isinstance(tokens[0], int)
            )

            # Create windows of WINDOW_SIZE tokens AFTER stripping special tokens
            # Since each special token is exactly 1 token, we extend by the count
            n_tokens = len(tokens)
            i = 0

            while i < n_tokens:
                # Start with base window
                window_end = min(i + WINDOW_SIZE, n_tokens)

                # Count special tokens and extend to compensate
                # Repeat until no new special tokens in extension
                while window_end < n_tokens:
                    special_count = sum(
                        1 for t in tokens[i:window_end] if t in special_token_ids
                    )
                    target_end = min(i + WINDOW_SIZE + special_count, n_tokens)
                    if target_end == window_end:
                        break  # No more extension needed
                    window_end = target_end

                window_tokens = tokens[i:window_end]
                window_offsets = cast(Tensor, offset_mapping)[i:window_end]

                # Get character range for this window
                # Find first non-empty offset for start
                start_char = 0
                for start_off, end_off in window_offsets:
                    if start_off != end_off:
                        start_char = start_off
                        break

                # Find last non-empty offset for end
                end_char = len(processed_text)
                for start_off, end_off in reversed(window_offsets):
                    if start_off != end_off:
                        end_char = end_off
                        break

                # Extract window text
                window_text = processed_text[start_char:end_char]

                # Get attributes for special tokens in this window
                window_attributes = get_attributes_in_range(
                    token_indices, all_attributes, start_char, end_char
                )

                json.dump(
                    {
                        "window_text": window_text,
                        "tag_attributes": window_attributes,
                        "filename": str(file.name),
                    },
                    f,
                )
                f.write("\n")

                # Slide by half window for overlap
                i += WINDOW_SIZE // 2
