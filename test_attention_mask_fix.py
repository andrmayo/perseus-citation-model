#!/usr/bin/env python3
"""Test that the attention mask fix allows inference to work properly."""

import torch
from perscit_model.extraction.inference import InferenceModel
from perscit_model.extraction.data_loader import ID2LABEL, create_extraction_dataset


def main():
    print("Loading model...")
    model = InferenceModel('outputs/models/extraction')
    model.model.eval()

    print("\n=== Test 1: Using InferenceModel.process_text (should now work) ===")
    # Test text with known citation patterns
    test_text = "According to Herodotus 1.1, the history begins here. Thucydides 2.5 disagrees."

    encoding, labels = model.process_text(test_text)
    non_o = [(i, l) for i, l in enumerate(labels) if l != 'O']
    print(f"Test text: {test_text}")
    print(f"Non-O predictions: {len(non_o)}")
    print(f"Predictions: {non_o[:20]}")

    print("\n=== Test 2: Direct tokenizer call with manual all-1s mask ===")
    inputs = model.loader.tokenizer(
        test_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Force all-1s attention mask (like the fix does internally)
    inputs["attention_mask"] = torch.ones_like(inputs["attention_mask"])

    with torch.no_grad():
        outputs = model.model(**inputs)

    if hasattr(model.model, "decode"):
        preds = model.model.decode(outputs.logits, inputs["attention_mask"].byte())[0]
    else:
        preds = outputs.logits.argmax(dim=-1)[0].tolist()

    labels2 = [ID2LABEL[p] for p in preds]
    non_o2 = [(i, l) for i, l in enumerate(labels2) if l != 'O']
    print(f"Non-O predictions with all-1s mask: {len(non_o2)}")
    print(f"Predictions: {non_o2[:20]}")

    print("\n=== Test 3: Compare with dataset preprocessing (ground truth) ===")
    # Load a validation example to compare
    dataset = create_extraction_dataset('model_data/extraction/phase_2/val.jsonl', num_proc=1)

    # Find an example with non-O labels
    for idx in range(min(50, len(dataset))):
        example = dataset[idx]
        true_labels = example['labels']
        non_o_true = [(i, ID2LABEL[l]) for i, l in enumerate(true_labels) if l not in (-100, 0)]

        if len(non_o_true) > 5:
            print(f"\nValidation example {idx}:")
            input_ids = torch.tensor([example['input_ids']]).to(model.device)
            attention_mask = torch.tensor([example['attention_mask']]).to(model.device)

            print(f"True non-O labels: {len(non_o_true)}")
            print(f"attention_mask sum: {attention_mask.sum().item()} / {attention_mask.shape[1]}")

            with torch.no_grad():
                outputs = model.model(input_ids=input_ids, attention_mask=attention_mask)

            if hasattr(model.model, 'decode'):
                preds = model.model.decode(outputs.logits, attention_mask.byte())[0]
            else:
                preds = outputs.logits.argmax(dim=-1)[0].tolist()

            pred_labels = [ID2LABEL[p] for p in preds]
            non_o_pred = [(i, l) for i, l in enumerate(pred_labels) if l != 'O']
            print(f"Predicted non-O labels: {len(non_o_pred)}")

            # Check overlap
            true_set = set(non_o_true)
            pred_set = set(non_o_pred)
            overlap = len(true_set & pred_set)
            print(f"Overlap: {overlap}/{len(true_set)}")
            break

    print("\n=== Test 4: Logit statistics ===")
    # Check logit distribution with the fix
    inputs = model.loader.tokenize_text(test_text)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    inputs["attention_mask"] = torch.ones_like(inputs["attention_mask"])

    with torch.no_grad():
        outputs = model.model(**inputs)
        logits = outputs.logits[0]

    print("Logit statistics per class:")
    for i, label in ID2LABEL.items():
        print(f"  {label}: mean={logits[:, i].mean():.3f}, max={logits[:, i].max():.3f}")


if __name__ == '__main__':
    main()
