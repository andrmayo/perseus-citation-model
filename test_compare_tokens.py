#!/usr/bin/env python3
"""Compare tokenization between dataset preprocessing and direct tokenization."""

import json
import torch
from perscit_model.extraction.inference import InferenceModel
from perscit_model.extraction.data_loader import ID2LABEL, create_extraction_dataset
from perscit_model.extraction.evaluate import strip_xml_tags


def main():
    print("Loading model...")
    model = InferenceModel('outputs/models/extraction')

    # Load validation examples
    with open('model_data/extraction/phase_2/val.jsonl') as f:
        val_examples = [json.loads(line) for line in f]

    # Load preprocessed dataset
    dataset = create_extraction_dataset('model_data/extraction/phase_2/val.jsonl', num_proc=1)

    # Find example with labels
    for idx in range(len(val_examples)):
        ds_example = dataset[idx]
        true_labels = ds_example['labels']
        non_o_true = [(i, ID2LABEL[l]) for i, l in enumerate(true_labels) if l not in (-100, 0)]
        if len(non_o_true) > 5:
            break

    print(f"\n=== Example {idx} ===")
    example = val_examples[idx]
    original_xml = example['window_text']
    stripped_text = strip_xml_tags(original_xml)

    print(f"\nOriginal XML:\n{original_xml[:500]}...")
    print(f"\nStripped text:\n{stripped_text[:500]}...")

    # Dataset preprocessing
    ds_input_ids = ds_example['input_ids']
    ds_attention_mask = ds_example['attention_mask']

    # Direct tokenization (no padding)
    direct_encoding = model.loader.tokenizer(
        stripped_text,
        truncation=True,
        max_length=512,
        padding=False,
        return_tensors="pt"
    )
    direct_input_ids = direct_encoding['input_ids'][0].tolist()
    direct_attention_mask = direct_encoding['attention_mask'][0].tolist()

    print(f"\n=== Token Comparison ===")
    print(f"Dataset input_ids length: {len(ds_input_ids)}")
    print(f"Direct input_ids length: {len(direct_input_ids)}")
    print(f"Dataset attention_mask sum: {sum(ds_attention_mask)}")
    print(f"Direct attention_mask sum: {sum(direct_attention_mask)}")

    # Check if token sequences are similar
    min_len = min(len(ds_input_ids), len(direct_input_ids))
    same_tokens = sum(1 for a, b in zip(ds_input_ids[:min_len], direct_input_ids[:min_len]) if a == b)
    print(f"Same tokens in first {min_len}: {same_tokens} ({100*same_tokens/min_len:.1f}%)")

    # Decode both to text
    ds_decoded = model.loader.tokenizer.decode(ds_input_ids, skip_special_tokens=True)
    direct_decoded = model.loader.tokenizer.decode(direct_input_ids, skip_special_tokens=True)

    print(f"\nDataset decoded (first 300):\n{ds_decoded[:300]}...")
    print(f"\nDirect decoded (first 300):\n{direct_decoded[:300]}...")

    # Show first few different tokens
    print(f"\n=== First 20 token IDs ===")
    print(f"Dataset: {ds_input_ids[:20]}")
    print(f"Direct:  {direct_input_ids[:20]}")

    # Decode first 20 tokens to see what they are
    ds_tokens = [model.loader.tokenizer.decode([t]) for t in ds_input_ids[:20]]
    direct_tokens = [model.loader.tokenizer.decode([t]) for t in direct_input_ids[:20]]
    print(f"\nDataset tokens: {ds_tokens}")
    print(f"Direct tokens:  {direct_tokens}")


if __name__ == '__main__':
    main()
