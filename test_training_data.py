#!/usr/bin/env python3
import torch
from perscit_model.extraction.model import load_model_from_checkpoint
from perscit_model.extraction.data_loader import ID2LABEL, create_extraction_dataset

def main():
    model = load_model_from_checkpoint('outputs/models/extraction')
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Load data exactly as training does
    dataset = create_extraction_dataset('model_data/extraction/phase_2/val.jsonl', num_proc=1)

    # Find examples with entity labels
    for idx in range(min(100, len(dataset))):
        example = dataset[idx]
        true_labels = example['labels']
        non_o_true = [(i, ID2LABEL[l]) for i, l in enumerate(true_labels) if l not in (-100, 0)]

        if len(non_o_true) > 0:
            print(f"\n=== Example {idx} ===")
            input_ids = torch.tensor([example['input_ids']]).to(device)
            attention_mask = torch.tensor([example['attention_mask']]).to(device)

            print(f"Non-O true labels ({len(non_o_true)} total): {non_o_true[:20]}")

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                if hasattr(model, 'decode'):
                    preds = model.decode(outputs.logits, attention_mask.byte())[0]
                else:
                    preds = outputs.logits.argmax(dim=-1)[0].tolist()

            pred_labels = [ID2LABEL[p] for p in preds]
            non_o_pred = [(i, l) for i, l in enumerate(pred_labels) if l != 'O']
            print(f"Non-O predictions ({len(non_o_pred)} total): {non_o_pred[:20]}")

            if idx >= 5:  # Show first 5 examples with entities
                break

if __name__ == '__main__':
    main()
