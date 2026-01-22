"""Check token lengths of windows in window_data.jsonl after stripping special tokens."""

import json
import sys
from pathlib import Path

from perscit_model.shared.data_loader import SharedDataLoader

# Special tokens that get stripped during training
SPECIAL_TOKENS = [
    "[CIT_START]",
    "[CIT_END]",
    "[BIBL_START]",
    "[BIBL_END]",
    "[QUOTE_START]",
    "[QUOTE_END]",
]


def strip_special_tokens(text: str) -> str:
    """Strip special tokens from text."""
    for token in SPECIAL_TOKENS:
        text = text.replace(token, "")
    return text


if __name__ == "__main__":
    loader = SharedDataLoader(special_tokens=SPECIAL_TOKENS)

    jsonl_path = Path("cit_data/xml_files/window_data.jsonl")

    if not jsonl_path.exists():
        print(f"File not found: {jsonl_path}")
        sys.exit(1)

    raw_lengths = []
    stripped_lengths = []

    # Sample every 100th line to speed up analysis
    sample_rate = 100

    print(f"Analyzing {jsonl_path} (sampling every {sample_rate}th line)...")

    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if i % sample_rate != 0:
                continue

            data = json.loads(line)
            window_text = data.get("window_text", "")

            # Check if this is new format (has special tokens) or old format
            has_special_tokens = (
                "[CIT_START]" in window_text or "[BIBL_START]" in window_text
            )

            # Raw token count (with special tokens)
            raw_tokens = loader.tokenizer(window_text, add_special_tokens=False)[
                "input_ids"
            ]
            raw_lengths.append(len(raw_tokens))

            # Stripped token count
            if has_special_tokens:
                stripped_text = strip_special_tokens(window_text)
                stripped_tokens = loader.tokenizer(
                    stripped_text, add_special_tokens=False
                )["input_ids"]
                stripped_lengths.append(len(stripped_tokens))
            else:
                # Old format - no special tokens to strip
                stripped_lengths.append(len(raw_tokens))

            if len(raw_lengths) % 500 == 0:
                print(f"  Processed {len(raw_lengths)} samples...")

    print(f"\n=== RESULTS ({len(raw_lengths)} samples) ===")
    print(
        f"Raw tokens: min={min(raw_lengths)}, max={max(raw_lengths)}, mean={sum(raw_lengths) / len(raw_lengths):.1f}"
    )
    print(
        f"After stripping: min={min(stripped_lengths)}, max={max(stripped_lengths)}, mean={sum(stripped_lengths) / len(stripped_lengths):.1f}"
    )

    # Distribution analysis
    within_10 = sum(1 for x in stripped_lengths if 502 <= x <= 522)
    within_20 = sum(1 for x in stripped_lengths if 492 <= x <= 532)
    within_50 = sum(1 for x in stripped_lengths if 462 <= x <= 562)
    under_400 = sum(1 for x in stripped_lengths if x < 400)
    over_550 = sum(1 for x in stripped_lengths if x > 550)

    print("\nDistribution (after stripping):")
    print(
        f"  Within ±10 of 512: {within_10}/{len(stripped_lengths)} ({100 * within_10 / len(stripped_lengths):.1f}%)"
    )
    print(
        f"  Within ±20 of 512: {within_20}/{len(stripped_lengths)} ({100 * within_20 / len(stripped_lengths):.1f}%)"
    )
    print(
        f"  Within ±50 of 512: {within_50}/{len(stripped_lengths)} ({100 * within_50 / len(stripped_lengths):.1f}%)"
    )
    print(
        f"  Under 400 tokens: {under_400}/{len(stripped_lengths)} ({100 * under_400 / len(stripped_lengths):.1f}%)"
    )
    print(
        f"  Over 550 tokens: {over_550}/{len(stripped_lengths)} ({100 * over_550 / len(stripped_lengths):.1f}%)"
    )

    # Check format
    sample_line = json.loads(open(jsonl_path).readline())
    if "tag_attributes" in sample_line:
        print("\nData format: NEW (has tag_attributes)")
    else:
        print("\nData format: OLD (no tag_attributes)")

    if "[CIT_START]" in sample_line.get(
        "window_text", ""
    ) or "[BIBL_START]" in sample_line.get("window_text", ""):
        print("Special tokens: YES (has [CIT_START]/[BIBL_START])")
    else:
        print("Special tokens: NO (raw XML tags)")
