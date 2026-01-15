#!/usr/bin/env python3
"""Test model on subset of dataset tokens."""

import torch
from perscit_model.extraction.inference import InferenceModel
from perscit_model.extraction.data_loader import ID2LABEL, create_extraction_dataset


def main():
    print("Loading model...")
    model = InferenceModel('outputs/models/extraction')
    model.model.eval()

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
    print(f"Dataset tokens: {len(ds_example['input_ids'])}")
    print(f"Citations at positions: {non_o_indices[0]} to {non_o_indices[-1]}")

    # Test with full 507 tokens
    print("\n=== Full 507 tokens (should work) ===")
    input_ids = torch.tensor([ds_example['input_ids']]).to(model.device)
    attention_mask = torch.tensor([ds_example['attention_mask']]).to(model.device)

    with torch.no_grad():
        outputs = model.model(input_ids=input_ids, attention_mask=attention_mask)
        preds = model.model.decode(outputs.logits, attention_mask.byte())[0]

    non_o_pred = [(i, ID2LABEL[p]) for i, p in enumerate(preds) if p != 0]
    print(f"Non-O predictions: {len(non_o_pred)}")

    # Test with first 421 tokens (matching direct tokenization length)
    print("\n=== First 421 tokens (subset) ===")
    input_ids_421 = torch.tensor([ds_example['input_ids'][:421]]).to(model.device)
    attention_mask_421 = torch.ones_like(input_ids_421)

    with torch.no_grad():
        outputs = model.model(input_ids=input_ids_421, attention_mask=attention_mask_421)
        preds = model.model.decode(outputs.logits, attention_mask_421.byte())[0]

    non_o_pred_421 = [(i, ID2LABEL[p]) for i, p in enumerate(preds) if p != 0]
    print(f"Non-O predictions: {len(non_o_pred_421)}")
    print(f"Predictions at citation positions (315-419): {[(i, l) for i, l in non_o_pred_421 if 315 <= i <= 419][:10]}")

    # Compare exact differences between dataset tokens and direct tokens
    print("\n=== Finding the 2 different tokens ===")
    from perscit_model.extraction.evaluate import strip_xml_tags
    import json
    with open('model_data/extraction/phase_2/val.jsonl') as f:
        val_examples = [json.loads(line) for line in f]
    example = val_examples[idx]
    stripped_text = strip_xml_tags(example['window_text'])

    direct_encoding = model.loader.tokenizer(
        stripped_text,
        truncation=True,
        max_length=512,
        padding=False,
        return_tensors="pt"
    )
    direct_input_ids = direct_encoding['input_ids'][0].tolist()

    ds_ids = ds_example['input_ids'][:421]
    for i, (ds, direct) in enumerate(zip(ds_ids, direct_input_ids)):
        if ds != direct:
            ds_token = model.loader.tokenizer.decode([ds])
            direct_token = model.loader.tokenizer.decode([direct])
            print(f"Position {i}: dataset='{ds_token}' ({ds}), direct='{direct_token}' ({direct})")


if __name__ == '__main__':
    main()
