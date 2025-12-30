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

import math
import sys
import yaml
from pathlib import Path
from xml import sax

from perscit_model.shared.xml_utils import CitXMLHandler

INCLUSION_THRESHOLD = 100
DEFAULT_OUTPUT = Path(__file__).parent.parent / "cit_data/xml_files"
WINDOW_SIZE = 512


if __name__ == "__main__":
    try:
        path = Path(sys.argv[1])
    except IndexError:
        print("Please provide path to dir with XML files")
        raise IndexError

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

    with open(output_path, "a", encoding="utf-8") as f:
        saxHandler = CitXMLHandler(f, WINDOW_SIZE)
        saxParser = sax.make_parser()
        saxParser.setContentHandler(saxHandler)

        files_for_training = []

        # only include mappings for included files
        file_to_idx = {}

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
        yaml.dump(stats, f)
