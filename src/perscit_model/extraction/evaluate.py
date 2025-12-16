"""Evaluation functions for citation extraction model."""

import json
import logging
import re
from pathlib import Path
from typing import cast

import torch
import transformers
from seqeval.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from perscit_model.extraction.data_loader import ID2LABEL, ExtractionDataLoader
from perscit_model.extraction.inference import InferenceModel

logger = logging.getLogger(__name__)

# Compile regex patterns once for better performance
_CIT_OPEN_PATTERN = re.compile(r"<cit[^>]*>")
_CIT_CLOSE_PATTERN = re.compile(r"</cit>")
_BIBL_OPEN_PATTERN = re.compile(r"<bibl[^>]*>")
_BIBL_CLOSE_PATTERN = re.compile(r"</bibl>")
_QUOTE_OPEN_PATTERN = re.compile(r"<quote[^>]*>")
_QUOTE_CLOSE_PATTERN = re.compile(r"</quote>")

# Patterns to extract citation elements with their content
_BIBL_EXTRACT_PATTERN = re.compile(r"<bibl[^>]*>(.*?)</bibl>", re.DOTALL)
_QUOTE_EXTRACT_PATTERN = re.compile(r"<quote[^>]*>(.*?)</quote>", re.DOTALL)


def extract_citations(text: str) -> dict[str, list[str]]:
    """
    Extract all <bibl> and <quote> elements from XML text.

    Args:
        text: XML text with citation tags

    Returns:
        Dict with 'bibl' and 'quote' keys containing lists of extracted text
    """
    bibls = _BIBL_EXTRACT_PATTERN.findall(text)
    quotes = _QUOTE_EXTRACT_PATTERN.findall(text)

    return {
        "bibl": bibls,
        "quote": quotes,
    }


def strip_xml_tags(text: str) -> str:
    """
    Remove <bibl>, <quote>, and <cit> tags from XML text.

    Keeps other XML tags (like <foreign>, <title>) intact.

    Args:
        text: XML text with citation tags

    Returns:
        Text with citation tags removed
    """
    # Use precompiled patterns for better performance
    text = _CIT_OPEN_PATTERN.sub("", text)
    text = _CIT_CLOSE_PATTERN.sub("", text)
    text = _BIBL_OPEN_PATTERN.sub("", text)
    text = _BIBL_CLOSE_PATTERN.sub("", text)
    text = _QUOTE_OPEN_PATTERN.sub("", text)
    text = _QUOTE_CLOSE_PATTERN.sub("", text)

    return text


def evaluate_model(
    model_path: Path | str | None = None,
    test_path: Path | str = "model_data/extraction/test.jsonl",
    output_dir: Path | str = "outputs/extraction/test",
    batch_size: int | None = None,
    last_trained: bool = False,
) -> dict:
    """
    Evaluate extraction model on test set.

    This function:
    1. Loads test data with ground truth labels
    2. Strips citation tags from input text
    3. Runs inference to predict tags
    4. Computes metrics (accuracy, precision, recall, F1)
    5. Saves predictions to output file

    Args:
        model_path: Path to model checkpoint (if None, uses last trained)
        test_path: Path to test.jsonl with ground truth
        output_dir: Directory to save evaluation results
        batch_size: Batch size for inference
        last_trained: If True, load most recent model from outputs/extraction

    Returns:
        Dict of evaluation metrics
    """

    # set batch_size depending on whether GPU or CPU is available
    if batch_size is None:
        if torch.cuda.is_available():
            batch_size = 128
        else:
            batch_size = 32

    # Setup paths
    test_path = Path(test_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading test data from {test_path}")
    logger.info(f"Output directory: {output_dir}")

    # Load test data
    test_examples = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            test_examples.append(json.loads(line))

    logger.info(f"Loaded {len(test_examples)} test examples")

    # Load model
    logger.info(f"Loading model (last_trained={last_trained})")
    model = InferenceModel(model_path=model_path, last_trained=last_trained)

    # Load data loader once (reuse tokenizer for all examples)
    loader = ExtractionDataLoader()

    # Create dataset using the SAME method as training - this ensures identical tokenization
    logger.info("Creating test dataset (same as training pipeline)...")
    from perscit_model.extraction.data_loader import create_extraction_dataset

    test_dataset = create_extraction_dataset(test_path, num_proc=1)
    logger.info(f"Test dataset size: {len(test_dataset)}")

    # Extract ground truth data from dataset
    all_tokenized_inputs = test_dataset["input_ids"]
    all_ground_truth_labels = test_dataset["labels"]
    all_attention_masks = test_dataset["attention_mask"]

    max_length = 512  # Model's maximum sequence length

    # Process all examples
    all_predictions = []
    all_true_labels = []
    all_results = []

    logger.info("Running inference on test set...")

    # Process in batches - dataset is tokenized (same as training)
    for i in tqdm(range(0, len(test_examples), batch_size)):
        batch_examples = test_examples[i : i + batch_size]
        batch_input_ids = all_tokenized_inputs[i : i + batch_size]
        batch_attention_masks = all_attention_masks[i : i + batch_size]

        # Pad batch to same length (like DataCollatorForTokenClassification does)
        max_len = max(len(ids) for ids in batch_input_ids)
        padded_input_ids = []
        padded_attention_masks = []

        for ids, mask in zip(batch_input_ids, batch_attention_masks):
            padding_length = max_len - len(ids)
            padded_ids = ids + [loader.tokenizer.pad_token_id] * padding_length
            padded_mask = mask + [0] * padding_length
            padded_input_ids.append(padded_ids)
            padded_attention_masks.append(padded_mask)

        # Create batch inputs
        batch_inputs = {
            "input_ids": torch.tensor(padded_input_ids, device=model.device),
            "attention_mask": torch.tensor(padded_attention_masks, device=model.device),
        }

        # Run model inference
        with torch.no_grad():
            outputs = model.model(**batch_inputs)

        # Get predictions
        predictions = outputs.logits.argmax(dim=-1).cpu().tolist()

        # Process each example in batch
        for j, (example, pred_ids, input_ids) in enumerate(
            zip(batch_examples, predictions, batch_input_ids)
        ):
            original_text = example["xml_context"]
            example_idx = i + j
            true_labels = all_ground_truth_labels[example_idx]
            attention_mask = all_attention_masks[
                example_idx
            ]  # Original mask from dataset

            # Get actual sequence length (excluding padding)
            seq_length = sum(attention_mask)

            # Convert prediction IDs to labels (only for non-padded tokens)
            pred_labels = [ID2LABEL[p] for p in pred_ids[:seq_length]]

            # Filter out special tokens (label = -100) from ground truth
            # Both predictions and ground truth are now aligned to the same tokens
            # Only process non-padded tokens
            filtered_true = []
            filtered_pred = []

            for true_label, pred_label in zip(true_labels[:seq_length], pred_labels):
                if true_label != -100:  # Skip special tokens
                    # Convert int labels to strings using ID2LABEL
                    filtered_true.append(ID2LABEL[true_label])
                    filtered_pred.append(pred_label)  # pred_label is already a string

            all_true_labels.append(filtered_true)
            all_predictions.append(filtered_pred)

            # Decode tokens to get stripped text for output (only non-padded tokens)
            stripped_text = loader.tokenizer.decode(
                input_ids[:seq_length], skip_special_tokens=True
            )

            # For predicted XML, we need offset mapping
            # Re-tokenize just this one example to get offsets
            tokens_for_xml = loader.tokenizer(
                stripped_text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                return_offsets_mapping=True,
            )

            # Insert predicted tags into stripped text
            predicted_xml = model.insert_tags_into_xml(
                stripped_text,
                tokens_for_xml,
                pred_labels,
            )

            # Extract citations from both original and predicted XML
            original_citations = extract_citations(original_text)
            predicted_citations = extract_citations(predicted_xml)

            # Store result (without labels - they can be inferred from XML)
            all_results.append(
                {
                    "filename": example["filename"],
                    "original_xml": original_text,
                    "stripped_text": stripped_text,
                    "predicted_xml": predicted_xml,
                    "original_citations": original_citations,
                    "predicted_citations": predicted_citations,
                }
            )

    # Compute metrics using seqeval
    logger.info("Computing metrics...")
    metrics = {
        "accuracy": accuracy_score(all_true_labels, all_predictions),
        "precision": precision_score(all_true_labels, all_predictions),
        "recall": recall_score(all_true_labels, all_predictions),
        "f1": f1_score(all_true_labels, all_predictions),
        "num_examples": len(test_examples),
    }

    # Get detailed classification report (ensure it returns a string)
    report = cast(str, classification_report(all_true_labels, all_predictions))
    logger.info("\nClassification Report:\n" + report)

    # Save metrics
    metrics_path = output_dir / "test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    # Save classification report
    report_path = output_dir / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Saved classification report to {report_path}")

    # Save predictions with reconstructed XML
    predictions_path = output_dir / "test_predictions.jsonl"
    with open(predictions_path, "w", encoding="utf-8") as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(all_results)} predictions to {predictions_path}")

    return metrics
