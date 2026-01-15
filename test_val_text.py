#!/usr/bin/env python3
"""Test inference on actual validation text (stripped of tags)."""

import json
import torch
from perscit_model.extraction.inference import InferenceModel
from perscit_model.extraction.data_loader import ID2LABEL, create_extraction_dataset
from perscit_model.extraction.evaluate import strip_xml_tags


def main():
    print("Loading model...")
    model = InferenceModel('outputs/models/extraction')
    model.model.eval()

    # Load validation examples
    print("Loading validation data...")
    with open('model_data/extraction/phase_2/val.jsonl') as f:
        val_examples = [json.loads(line) for line in f]

    # Also load the preprocessed dataset for comparison
    dataset = create_extraction_dataset('model_data/extraction/phase_2/val.jsonl', num_proc=1)

    print("\n=== Testing on validation text (stripped of tags) ===")

    for idx in range(min(10, len(val_examples))):
        example = val_examples[idx]
        original_xml = example['window_text']

        # Strip citation tags (like inference would do)
        stripped_text = strip_xml_tags(original_xml)

        # Get ground truth from dataset
        ds_example = dataset[idx]
        true_labels = ds_example['labels']
        non_o_true = [(i, ID2LABEL[l]) for i, l in enumerate(true_labels) if l not in (-100, 0)]

        if len(non_o_true) == 0:
            continue

        print(f"\n=== Example {idx} ===")
        print(f"Original XML (first 200 chars): {original_xml[:200]}...")
        print(f"Stripped text (first 200 chars): {stripped_text[:200]}...")
        print(f"True non-O labels in dataset: {len(non_o_true)}")

        # Method 1: Use process_text (with my fix)
        encoding, labels = model.process_text(stripped_text)  # Full text
        non_o_pred = [(i, l) for i, l in enumerate(labels) if l != 'O']
        print(f"Predictions via process_text: {len(non_o_pred)} non-O")

        # Method 2: Direct tokenization matching dataset
        # Tokenize without padding to see what happens
        inputs = model.loader.tokenizer(
            stripped_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False  # No padding!
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # With all-1s mask (like training)
        inputs["attention_mask"] = torch.ones_like(inputs["attention_mask"])

        with torch.no_grad():
            outputs = model.model(**inputs)

        if hasattr(model.model, 'decode'):
            preds = model.model.decode(outputs.logits, inputs["attention_mask"].byte())[0]
        else:
            preds = outputs.logits.argmax(dim=-1)[0].tolist()

        labels2 = [ID2LABEL[p] for p in preds]
        non_o_pred2 = [(i, l) for i, l in enumerate(labels2) if l != 'O']
        print(f"Predictions via direct tokenize (no padding): {len(non_o_pred2)} non-O")

        # Method 3: Use dataset preprocessing (ground truth method)
        input_ids = torch.tensor([ds_example['input_ids']]).to(model.device)
        attention_mask = torch.tensor([ds_example['attention_mask']]).to(model.device)

        with torch.no_grad():
            outputs = model.model(input_ids=input_ids, attention_mask=attention_mask)

        if hasattr(model.model, 'decode'):
            preds3 = model.model.decode(outputs.logits, attention_mask.byte())[0]
        else:
            preds3 = outputs.logits.argmax(dim=-1)[0].tolist()

        labels3 = [ID2LABEL[p] for p in preds3]
        non_o_pred3 = [(i, l) for i, l in enumerate(labels3) if l != 'O']
        print(f"Predictions via dataset preprocessing: {len(non_o_pred3)} non-O")

        if idx >= 2:  # Show first 3 examples with entities
            break


if __name__ == '__main__':
    main()
