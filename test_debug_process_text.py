#!/usr/bin/env python3
"""Debug process_text to understand the prediction flow."""

import json
import torch
from perscit_model.extraction.inference import InferenceModel
from perscit_model.extraction.data_loader import ID2LABEL
from perscit_model.extraction.evaluate import strip_xml_tags


def main():
    print("Loading model...")
    model = InferenceModel('outputs/models/extraction')
    model.model.eval()

    # Get validation example
    with open('model_data/extraction/phase_2/val.jsonl') as f:
        val_examples = [json.loads(line) for line in f]
    example = val_examples[6]
    stripped_text = strip_xml_tags(example['window_text'])

    print(f"Full stripped text: {len(stripped_text)} chars")

    # Test 1: Full text via process_text
    print("\n=== Test 1: Full text via process_text ===")
    encoding, labels = model.process_text(stripped_text)
    non_o = [(i, l) for i, l in enumerate(labels) if l != 'O']
    print(f"Labels returned: {len(labels)}")
    print(f"Non-O predictions: {len(non_o)}")

    # Debug: check what tokenize_text produces
    print("\n=== Debug: tokenize_text output ===")
    inputs = model.loader.tokenize_text(stripped_text)
    print(f"input_ids shape: {inputs['input_ids'].shape}")
    print(f"attention_mask shape: {inputs['attention_mask'].shape}")
    print(f"attention_mask sum: {inputs['attention_mask'].sum().item()}")

    # Test 2: Manual process matching dataset (pad to 512, all-1s mask)
    print("\n=== Test 2: Manual process (pad to 512, all-1s mask) ===")
    encoding_no_pad = model.loader.tokenizer(
        stripped_text,
        truncation=True,
        max_length=512,
        padding=False,  # Don't pad
        return_tensors="pt"
    )
    input_ids = encoding_no_pad['input_ids']
    print(f"Unpadded input_ids shape: {input_ids.shape}")

    # Pad to 512 manually
    if input_ids.shape[1] < 512:
        pad_len = 512 - input_ids.shape[1]
        input_ids_padded = torch.nn.functional.pad(
            input_ids, (0, pad_len), value=model.loader.tokenizer.pad_token_id
        )
    else:
        input_ids_padded = input_ids

    # All-1s attention mask
    attention_mask = torch.ones_like(input_ids_padded)

    inputs_device = {
        'input_ids': input_ids_padded.to(model.device),
        'attention_mask': attention_mask.to(model.device)
    }

    with torch.no_grad():
        outputs = model.model(**inputs_device)
        preds = model.model.decode(outputs.logits, attention_mask.to(model.device).byte())[0]

    # Only take predictions for original length
    original_len = encoding_no_pad['input_ids'].shape[1]
    preds_trimmed = preds[:original_len]
    labels_manual = [ID2LABEL[p] for p in preds_trimmed]
    non_o_manual = [(i, l) for i, l in enumerate(labels_manual) if l != 'O']
    print(f"Labels returned: {len(labels_manual)}")
    print(f"Non-O predictions: {len(non_o_manual)}")
    if non_o_manual:
        print(f"First few: {non_o_manual[:10]}")


if __name__ == '__main__':
    main()
