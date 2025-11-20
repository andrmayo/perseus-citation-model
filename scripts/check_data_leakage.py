#!/usr/bin/env python3
"""
Check for data leakage between JSONL data partitions.

This script checks for similar xml_context fields across different JSONL files
in a directory, using both exact matches and fuzzy text similarity.

Usage:
    python scripts/check_data_leakage.py <target_dir>
    python scripts/check_data_leakage.py model_data/extraction
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

try:
    from difflib import SequenceMatcher
except ImportError:
    print("Error: difflib not available (should be in stdlib)")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    tqdm = None


def similarity_ratio(text1: str, text2: str) -> float:
    """Calculate similarity ratio between two strings (0.0 to 1.0)."""
    # Quick length filter - if lengths differ by >20%, similarity will be < 0.8
    len1, len2 = len(text1), len(text2)
    if abs(len1 - len2) / max(len1, len2) > 0.2:
        return 0.0

    return SequenceMatcher(None, text1, text2).ratio()


def load_jsonl_contexts(file_path: Path) -> List[Tuple[str, str, int]]:
    """
    Load xml_context fields from a JSONL file.

    Returns:
        List of tuples: (xml_context, filename, line_number)
    """
    contexts = []
    with open(file_path) as f:
        for i, line in enumerate(f, start=1):
            try:
                data = json.loads(line)
                xml_context = data.get("xml_context", "")
                filename = data.get("filename", "unknown")
                contexts.append((xml_context, filename, i))
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {i} in {file_path}: {e}")
    return contexts


def check_exact_duplicates(
    splits: Dict[str, List[Tuple[str, str, int]]]
) -> Dict[Tuple[str, str], List[str]]:
    """
    Check for exact duplicate xml_contexts across splits.

    Returns:
        Dict mapping (split1, split2) -> list of duplicate contexts
    """
    # Build index: context -> list of (split_name, source_filename, line_num)
    context_index = defaultdict(list)

    for split_name, contexts in splits.items():
        for xml_context, source_file, line_num in contexts:
            context_index[xml_context].append((split_name, source_file, line_num))

    # Find contexts that appear in multiple splits
    cross_split_duplicates = defaultdict(list)

    for xml_context, occurrences in context_index.items():
        splits_with_context = set(occ[0] for occ in occurrences)
        if len(splits_with_context) > 1:
            # This context appears in multiple splits
            for i, (split1, _, _) in enumerate(occurrences):
                for split2, _, _ in occurrences[i+1:]:
                    if split1 != split2:
                        key = tuple(sorted([split1, split2]))
                        cross_split_duplicates[key].append(xml_context)

    return cross_split_duplicates


def check_fuzzy_similarity(
    splits: Dict[str, List[Tuple[str, str, int]]],
    threshold: float = 0.9
) -> List[Dict]:
    """
    Check for highly similar (but not identical) xml_contexts across splits.

    Args:
        splits: Dictionary mapping split name to list of contexts
        threshold: Similarity threshold (0.0-1.0). Higher = more similar required.

    Returns:
        List of similarity findings with details
    """
    findings = []
    split_names = list(splits.keys())

    print(f"\nChecking fuzzy similarity (threshold={threshold})...")
    print("This may take a while for large datasets...")

    # Compare each pair of splits
    for i, split1 in enumerate(split_names):
        for split2 in split_names[i+1:]:
            print(f"  Comparing {split1} vs {split2}...", end="\r")

            contexts1 = splits[split1]
            contexts2 = splits[split2]

            # Sample if too large (for performance)
            max_comparisons = 100000  # Reduced from 10M to 100k for speed
            if len(contexts1) * len(contexts2) > max_comparisons:
                print(f"\n  Warning: {split1} and {split2} are large. Sampling for performance...")
                # Sample proportionally
                import random
                sample_size = int(max_comparisons ** 0.5)  # e.g., sqrt(100k) = 316
                contexts1 = random.sample(contexts1, min(sample_size, len(contexts1)))
                contexts2 = random.sample(contexts2, min(sample_size, len(contexts2)))
                print(f"  Sampled to {len(contexts1)} x {len(contexts2)} = {len(contexts1) * len(contexts2)} comparisons")

            # Compare all pairs with progress bar
            if tqdm:
                iterator = tqdm(
                    contexts1,
                    desc=f"  {split1} vs {split2}",
                    unit=" examples",
                    leave=False
                )
            else:
                iterator = contexts1

            for i, (ctx1, file1, line1) in enumerate(iterator):
                # Fallback progress if no tqdm
                if not tqdm and i % max(1, len(contexts1) // 10) == 0:
                    pct = (i / len(contexts1)) * 100
                    print(f"  Comparing {split1} vs {split2}... {pct:.0f}%", end="\r")

                for ctx2, file2, line2 in contexts2:
                    if ctx1 != ctx2:  # Skip exact matches (already found)
                        sim = similarity_ratio(ctx1, ctx2)
                        if sim >= threshold:
                            findings.append({
                                "split1": split1,
                                "split2": split2,
                                "similarity": sim,
                                "context1_preview": ctx1[:200],
                                "context2_preview": ctx2[:200],
                                "file1": file1,
                                "file2": file2,
                                "line1": line1,
                                "line2": line2,
                            })

    print(" " * 80, end="\r")  # Clear progress line
    return findings


def main():
    parser = argparse.ArgumentParser(
        description="Check for data leakage between JSONL data partitions"
    )
    parser.add_argument(
        "target_dir",
        type=Path,
        help="Directory containing JSONL files (e.g., model_data/extraction)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.9,
        help="Similarity threshold for fuzzy matching (0.0-1.0, default: 0.9)",
    )
    parser.add_argument(
        "--skip-fuzzy",
        action="store_true",
        help="Skip fuzzy similarity check (faster, only check exact duplicates)",
    )

    args = parser.parse_args()

    # Find all JSONL files
    target_dir = args.target_dir
    if not target_dir.exists():
        print(f"Error: Directory {target_dir} does not exist")
        sys.exit(1)

    jsonl_files = list(target_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"Error: No .jsonl files found in {target_dir}")
        sys.exit(1)

    print(f"Found {len(jsonl_files)} JSONL files in {target_dir}:")
    for f in jsonl_files:
        print(f"  - {f.name}")
    print()

    # Load all splits
    splits = {}
    for jsonl_file in jsonl_files:
        split_name = jsonl_file.stem  # e.g., "train", "val", "test"
        print(f"Loading {split_name}...", end=" ")
        contexts = load_jsonl_contexts(jsonl_file)
        splits[split_name] = contexts
        print(f"{len(contexts)} examples")

    print()

    # Check 1: Exact duplicates
    print("=" * 80)
    print("CHECK 1: Exact Duplicate xml_contexts Across Splits")
    print("=" * 80)

    exact_dups = check_exact_duplicates(splits)

    if exact_dups:
        print("\n⚠️  EXACT DUPLICATES FOUND!")
        for (split1, split2), dup_contexts in exact_dups.items():
            print(f"\n{split1} ↔ {split2}: {len(dup_contexts)} duplicate contexts")
            print(f"  First 3 examples:")
            for ctx in list(dup_contexts)[:3]:
                print(f"    {ctx[:100]}...")
    else:
        print("\n✅ No exact duplicates found across splits")

    # Check 2: Fuzzy similarity
    if not args.skip_fuzzy:
        print("\n" + "=" * 80)
        print(f"CHECK 2: Fuzzy Similarity (threshold={args.similarity_threshold})")
        print("=" * 80)

        similar = check_fuzzy_similarity(splits, threshold=args.similarity_threshold)

        if similar:
            print(f"\n⚠️  SIMILAR CONTEXTS FOUND: {len(similar)} cases")
            print(f"\nTop 10 most similar:")
            # Sort by similarity
            similar.sort(key=lambda x: x["similarity"], reverse=True)
            for i, finding in enumerate(similar[:10], 1):
                print(f"\n  {i}. {finding['split1']} ↔ {finding['split2']}")
                print(f"     Similarity: {finding['similarity']:.3f}")
                print(f"     Files: {finding['file1']} (line {finding['line1']}) vs")
                print(f"            {finding['file2']} (line {finding['line2']})")
                print(f"     Context 1: {finding['context1_preview']}...")
                print(f"     Context 2: {finding['context2_preview']}...")
        else:
            print(f"\n✅ No similar contexts found above {args.similarity_threshold} threshold")
    else:
        print("\nSkipping fuzzy similarity check (--skip-fuzzy)")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    leakage_found = bool(exact_dups) or (not args.skip_fuzzy and bool(similar))

    if leakage_found:
        print("❌ DATA LEAKAGE DETECTED")
        print("\nRecommendations:")
        print("  1. Review the flagged examples above")
        print("  2. Re-run data splitting with different seed or strategy")
        print("  3. Check for duplicate source files with different names")
        sys.exit(1)
    else:
        print("✅ NO DATA LEAKAGE DETECTED")
        print("\nAll splits appear to be properly separated.")
        sys.exit(0)


if __name__ == "__main__":
    main()
