#!/usr/bin/env python3
"""Count <author>, <title>, and [BIBL_START] occurrences in JSONL files."""

import argparse
import json
import re
from pathlib import Path


def count_tags(directory: str) -> dict[str, int]:
    """Count tag occurrences in all JSONL files in directory."""
    counts = {
        "<author>": 0,
        "<title>": 0,
        "[BIBL_START]": 0,
    }

    dir_path = Path(directory)
    jsonl_files = list(dir_path.glob("*.jsonl"))

    if not jsonl_files:
        print(f"No .jsonl files found in {directory}")
        return counts

    for jsonl_file in jsonl_files:
        file_counts = {"<author>": 0, "<title>": 0, "[BIBL_START]": 0}
        line_count = 0

        with open(jsonl_file) as f:
            for line in f:
                line_count += 1
                example = json.loads(line)

                # Check both possible text fields
                text = example.get("xml_context", "") or example.get("window_text", "")

                # Count occurrences
                file_counts["<author>"] += len(re.findall(r"<author[^>]*>", text))
                file_counts["<title>"] += len(re.findall(r"<title[^>]*>", text))
                file_counts["[BIBL_START]"] += text.count("[BIBL_START]")

        print(f"{jsonl_file.name}: {line_count} examples")
        print(f"  <author>: {file_counts['<author>']}")
        print(f"  <title>: {file_counts['<title>']}")
        print(f"  [BIBL_START]: {file_counts['[BIBL_START]']}")
        print()

        for key in counts:
            counts[key] += file_counts[key]

    return counts


def main():
    parser = argparse.ArgumentParser(description="Count tags in JSONL files")
    parser.add_argument("directory", help="Directory containing JSONL files")
    args = parser.parse_args()

    counts = count_tags(args.directory)

    print("=" * 40)
    print("TOTAL:")
    print(f"  <author>: {counts['<author>']}")
    print(f"  <title>: {counts['<title>']}")
    print(f"  [BIBL_START]: {counts['[BIBL_START]']}")


if __name__ == "__main__":
    main()
