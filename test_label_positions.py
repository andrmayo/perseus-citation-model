#!/usr/bin/env python3
"""Check where citation labels appear in the sequence."""

import json
from perscit_model.extraction.inference import InferenceModel
from perscit_model.extraction.data_loader import ID2LABEL, create_extraction_dataset


def main():
    print("Loading model...")
    model = InferenceModel('outputs/models/extraction')

    # Load preprocessed dataset
    dataset = create_extraction_dataset('model_data/extraction/phase_2/val.jsonl', num_proc=1)

    # Find example with labels
    for idx in range(len(dataset)):
        ds_example = dataset[idx]
        true_labels = ds_example['labels']
        non_o_indices = [i for i, l in enumerate(true_labels) if l not in (-100, 0)]
        if len(non_o_indices) > 5:
            break

    print(f"\n=== Example {idx} ===")
    print(f"Total tokens in dataset: {len(ds_example['input_ids'])}")
    print(f"Number of non-O labels: {len(non_o_indices)}")
    print(f"Position of first non-O label: {non_o_indices[0]}")
    print(f"Position of last non-O label: {non_o_indices[-1]}")
    print(f"\nAll non-O label positions: {non_o_indices}")

    # Decode the tokens around the labels
    print(f"\n=== Text around citations ===")
    for i in range(0, len(non_o_indices), 10):
        pos = non_o_indices[i]
        context_start = max(0, pos - 5)
        context_end = min(len(ds_example['input_ids']), pos + 10)
        context_tokens = ds_example['input_ids'][context_start:context_end]
        context_labels = [ID2LABEL.get(l, 'SKIP') for l in true_labels[context_start:context_end]]
        context_text = model.loader.tokenizer.decode(context_tokens)
        print(f"Position {pos}: {context_text}")
        print(f"  Labels: {context_labels}")


if __name__ == '__main__':
    main()
