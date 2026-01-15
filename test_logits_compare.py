#!/usr/bin/env python3
"""Compare logits between full and truncated sequences."""

import torch
from perscit_model.extraction.inference import InferenceModel
from perscit_model.extraction.data_loader import ID2LABEL, create_extraction_dataset


def main():
    print("Loading model...")
    model = InferenceModel('outputs/models/extraction')
    model.model.eval()

    dataset = create_extraction_dataset('model_data/extraction/phase_2/val.jsonl', num_proc=1)

    # Find example with labels
    for idx in range(len(dataset)):
        ds_example = dataset[idx]
        true_labels = ds_example['labels']
        non_o_indices = [i for i, l in enumerate(true_labels) if l not in (-100, 0)]
        if len(non_o_indices) > 5:
            break

    print(f"Example {idx}: citations at positions {non_o_indices[0]}-{non_o_indices[-1]}")

    # Full 507 tokens
    input_ids_full = torch.tensor([ds_example['input_ids']]).to(model.device)
    attention_mask_full = torch.tensor([ds_example['attention_mask']]).to(model.device)

    # First 421 tokens
    input_ids_421 = torch.tensor([ds_example['input_ids'][:421]]).to(model.device)
    attention_mask_421 = torch.ones_like(input_ids_421)

    with torch.no_grad():
        outputs_full = model.model(input_ids=input_ids_full, attention_mask=attention_mask_full)
        outputs_421 = model.model(input_ids=input_ids_421, attention_mask=attention_mask_421)

    logits_full = outputs_full.logits[0]
    logits_421 = outputs_421.logits[0]

    print(f"\n=== Logits comparison for position 315 (first citation) ===")
    print(f"Full sequence logits: {logits_full[315].tolist()}")
    print(f"421-token logits:     {logits_421[315].tolist()}")

    print(f"\n=== Logit statistics at citation positions (315-335) ===")
    print("Full sequence:")
    for i, label in ID2LABEL.items():
        vals = logits_full[315:336, i]
        print(f"  {label}: mean={vals.mean():.3f}, max={vals.max():.3f}")

    print("\n421 tokens:")
    for i, label in ID2LABEL.items():
        vals = logits_421[315:336, i]
        print(f"  {label}: mean={vals.mean():.3f}, max={vals.max():.3f}")

    print(f"\n=== Argmax predictions at citation positions ===")
    argmax_full = logits_full[315:336].argmax(dim=-1)
    argmax_421 = logits_421[315:336].argmax(dim=-1)
    print(f"Full: {[ID2LABEL[p.item()] for p in argmax_full]}")
    print(f"421:  {[ID2LABEL[p.item()] for p in argmax_421]}")

    print(f"\n=== CRF decode comparison ===")
    preds_full = model.model.decode(outputs_full.logits, attention_mask_full.byte())[0]
    preds_421 = model.model.decode(outputs_421.logits, attention_mask_421.byte())[0]

    print(f"Full CRF decode at 315-335: {[ID2LABEL[preds_full[i]] for i in range(315, 336)]}")
    print(f"421 CRF decode at 315-335:  {[ID2LABEL[preds_421[i]] for i in range(315, 336)]}")


if __name__ == '__main__':
    main()
