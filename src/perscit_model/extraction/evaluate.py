"""Evaluation functions for citation extraction model."""

import json
import logging
from pathlib import Path
from typing import cast

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


def strip_xml_tags(text: str) -> str:
    """
    Remove <bibl>, <quote>, and <cit> tags from XML text.

    Keeps other XML tags (like <foreign>, <title>) intact.

    Args:
        text: XML text with citation tags

    Returns:
        Text with citation tags removed
    """
    import re

    # Remove <cit> tags (can contain other tags)
    text = re.sub(r"<cit[^>]*>", "", text)
    text = re.sub(r"</cit>", "", text)

    # Remove <bibl> tags
    text = re.sub(r"<bibl[^>]*>", "", text)
    text = re.sub(r"</bibl>", "", text)

    # Remove <quote> tags
    text = re.sub(r"<quote[^>]*>", "", text)
    text = re.sub(r"</quote>", "", text)

    return text


def evaluate_model(
    model_path: Path | str | None = None,
    test_path: Path | str = "model_data/extraction/test.jsonl",
    output_dir: Path | str = "outputs/extraction/test",
    batch_size: int = 32,
    last_trained: bool = True,
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

    # Process all examples
    all_predictions = []
    all_true_labels = []
    all_results = []

    logger.info("Running inference on test set...")

    # Process in batches using process_batch method
    for i in tqdm(range(0, len(test_examples), batch_size)):
        batch_examples = test_examples[i : i + batch_size]

        # Extract original XML texts and strip tags
        xml_texts = [ex["xml_context"] for ex in batch_examples]
        stripped_texts = [strip_xml_tags(text) for text in xml_texts]

        # Run batch inference - returns list of (encoding, labels) tuples
        batch_results = model.process_batch(
            stripped_texts, batch_size=len(stripped_texts)
        )

        # For each example in batch, get ground truth and compare
        for j, (example, (inputs, pred_labels)) in enumerate(
            zip(batch_examples, batch_results)
        ):
            # Get ground truth labels by re-processing the original XML

            original_text = example["xml_context"]

            # Parse XML to special tokens, then tokenize
            parsed_text = ExtractionDataLoader.parse_xml_to_bio(original_text)
            ground_truth_inputs = cast(
                transformers.BatchEncoding, loader.tokenize_text(parsed_text)
            )
            ground_truth_labels_ids = loader.generate_bio_labels(
                ground_truth_inputs["input_ids"][0].tolist()
            )

            # Strip special tokens from ground truth to match inference
            true_labels_clean, _ = loader.strip_special_tokens_and_align_labels(
                ground_truth_inputs["input_ids"][0].tolist(), ground_truth_labels_ids
            )
            true_labels = [
                ID2LABEL.get(label_id, "O") for label_id in true_labels_clean
            ]

            # Filter out special tokens (label = -100) from both
            filtered_true = []
            filtered_pred = []

            for true_label, pred_label in zip(true_labels, pred_labels):
                if true_label != -100:  # Skip special tokens
                    filtered_true.append(true_label)
                    filtered_pred.append(pred_label)

            all_true_labels.append(filtered_true)
            all_predictions.append(filtered_pred)

            # Insert predicted tags into stripped text
            predicted_xml = model.insert_tags_into_xml(
                stripped_texts[j],
                inputs,
                [pred_labels],  # Wrap in list for single text
            )

            # Store result
            all_results.append(
                {
                    "filename": example["filename"],
                    "original_xml": original_text,
                    "stripped_text": stripped_texts[j],
                    "predicted_xml": predicted_xml,
                    "true_labels": filtered_true,
                    "pred_labels": filtered_pred,
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
