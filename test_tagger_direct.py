#!/usr/bin/env python3
"""Test the tagger's sliding window approach directly."""

import json
from pathlib import Path
from perscit_model.xml_processing.tagger import CitationTagger
from perscit_model.extraction.data_loader import ID2LABEL


def main():
    print("Loading tagger...")
    tagger = CitationTagger('outputs/models/extraction')

    # Get a validation example
    with open('model_data/extraction/phase_2/val.jsonl') as f:
        val_examples = [json.loads(line) for line in f]

    # Use example 6 which has citations at positions 315-419
    example = val_examples[6]
    window_text = example['window_text']

    # Strip citation tags (like tagger.process_xml_file does with remove_existing_citations=True)
    from perscit_model.extraction.evaluate import strip_xml_tags
    window_text = strip_xml_tags(window_text)

    print(f"Input text length: {len(window_text)} chars")
    print(f"First 500 chars:\n{window_text[:500]}...")

    # Tokenize the full text (like tagger does)
    encoding = tagger.loader.tokenizer(
        window_text,
        truncation=False,
        padding=False,
        return_tensors="pt",
        return_offsets_mapping=True,
    )

    input_ids = encoding.input_ids[0]
    attention_mask = encoding.attention_mask[0]

    print(f"\nTotal tokens: {len(input_ids)}")

    # Use tagger's _get_labels method
    labels = tagger._get_labels(input_ids, attention_mask)

    print(f"Total labels: {len(labels)}")
    non_o = [(i, l) for i, l in enumerate(labels) if l != 'O']
    print(f"Non-O predictions: {len(non_o)}")
    if non_o:
        print(f"First 20 non-O: {non_o[:20]}")


if __name__ == '__main__':
    main()
