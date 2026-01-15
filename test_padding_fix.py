#!/usr/bin/env python3
"""Test if padding to match training length restores predictions."""

import torch
import torch.nn.functional as F
from perscit_model.extraction.inference import InferenceModel
from perscit_model.extraction.data_loader import ID2LABEL, create_extraction_dataset
from perscit_model.extraction.evaluate import strip_xml_tags
import json


def main():
    print("Loading model...")
    model = InferenceModel('outputs/models/extraction')
    model.model.eval()

    # Load dataset for reference
    dataset = create_extraction_dataset('model_data/extraction/phase_2/val.jsonl', num_proc=1)
    ds_example = dataset[6]
    print(f"Dataset example 6: {len(ds_example['input_ids'])} tokens")

    # Get the raw text
    with open('model_data/extraction/phase_2/val.jsonl') as f:
        val_examples = [json.loads(line) for line in f]
    example = val_examples[6]
    stripped_text = strip_xml_tags(example['window_text'])

    # Tokenize stripped text
    encoding = model.loader.tokenizer(
        stripped_text,
        truncation=True,
        max_length=512,
        padding=False,
        return_tensors="pt"
    )
    input_ids = encoding['input_ids']
    print(f"Direct tokenization: {input_ids.shape[1]} tokens")

    # Test 1: No padding (current behavior)
    print("\n=== Test 1: No padding (421 tokens) ===")
    attention_mask = torch.ones_like(input_ids)
    inputs = {
        'input_ids': input_ids.to(model.device),
        'attention_mask': attention_mask.to(model.device)
    }
    with torch.no_grad():
        outputs = model.model(**inputs)
        preds = model.model.decode(outputs.logits, inputs['attention_mask'].byte())[0]
    non_o = [(i, ID2LABEL[p]) for i, p in enumerate(preds) if p != 0]
    print(f"Non-O predictions: {len(non_o)}")

    # Test 2: Pad to 507 tokens (matching dataset length)
    print("\n=== Test 2: Pad to 507 tokens (PAD token) ===")
    pad_token_id = model.loader.tokenizer.pad_token_id
    target_len = 507
    current_len = input_ids.shape[1]
    pad_len = target_len - current_len

    if pad_len > 0:
        input_ids_padded = F.pad(input_ids, (0, pad_len), value=pad_token_id)
        attention_mask_padded = torch.ones((1, target_len))  # All 1s like training
    else:
        input_ids_padded = input_ids
        attention_mask_padded = torch.ones_like(input_ids)

    inputs = {
        'input_ids': input_ids_padded.to(model.device),
        'attention_mask': attention_mask_padded.to(model.device)
    }
    with torch.no_grad():
        outputs = model.model(**inputs)
        preds = model.model.decode(outputs.logits, inputs['attention_mask'].byte())[0]
    non_o = [(i, ID2LABEL[p]) for i, p in enumerate(preds[:current_len]) if p != 0]
    print(f"Non-O predictions (first {current_len}): {len(non_o)}")

    # Test 3: Pad to 512 tokens (full max length)
    print("\n=== Test 3: Pad to 512 tokens ===")
    target_len = 512
    pad_len = target_len - current_len
    input_ids_padded = F.pad(input_ids, (0, pad_len), value=pad_token_id)
    attention_mask_padded = torch.ones((1, target_len))

    inputs = {
        'input_ids': input_ids_padded.to(model.device),
        'attention_mask': attention_mask_padded.to(model.device)
    }
    with torch.no_grad():
        outputs = model.model(**inputs)
        preds = model.model.decode(outputs.logits, inputs['attention_mask'].byte())[0]
    non_o = [(i, ID2LABEL[p]) for i, p in enumerate(preds[:current_len]) if p != 0]
    print(f"Non-O predictions (first {current_len}): {len(non_o)}")

    # Test 4: Use ACTUAL tokens from dataset to extend (most similar to training)
    print("\n=== Test 4: Extend with dataset tokens 421-507 ===")
    # Take the first 421 from direct tokenization, then append tokens 421-507 from dataset
    ds_tokens_extension = ds_example['input_ids'][421:507]
    extended_ids = input_ids[0].tolist() + ds_tokens_extension
    input_ids_extended = torch.tensor([extended_ids])
    attention_mask_extended = torch.ones((1, len(extended_ids)))

    inputs = {
        'input_ids': input_ids_extended.to(model.device),
        'attention_mask': attention_mask_extended.to(model.device)
    }
    with torch.no_grad():
        outputs = model.model(**inputs)
        preds = model.model.decode(outputs.logits, inputs['attention_mask'].byte())[0]
    non_o = [(i, ID2LABEL[p]) for i, p in enumerate(preds[:current_len]) if p != 0]
    print(f"Non-O predictions (first {current_len}): {len(non_o)}")


if __name__ == '__main__':
    main()
