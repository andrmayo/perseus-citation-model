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

    # Precompute ground truth labels and stripped texts (do expensive ops once upfront)
    logger.info("Preprocessing test examples (computing labels and stripping tags)...")
    all_ground_truth_labels = []
    all_stripped_texts = []
    max_length = 512  # Model's maximum sequence length

    for example in tqdm(test_examples, desc="Preprocessing"):
        # Use the SAME approach as training: convert XML tags to special tokens,
        # generate labels, then strip special tokens
        xml_with_special_tokens = ExtractionDataLoader.parse_xml_to_bio(
            example["xml_context"]
        )

        # Tokenize WITH special tokens (same as training does)
        tokens_with_special = cast(
            transformers.BatchEncoding,
            loader.tokenizer(
                xml_with_special_tokens,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ),
        )

        # Generate BIO labels from special token positions (same as training)
        input_ids_with_special = tokens_with_special["input_ids"][0].tolist()
        labels_with_special = loader.generate_bio_labels(input_ids_with_special)

        # Strip special tokens and align labels (same as training)
        clean_input_ids, aligned_labels = loader.strip_special_tokens_and_align_labels(
            input_ids_with_special, labels_with_special
        )

        # Decode the clean input to get stripped text
        stripped_text = loader.tokenizer.decode(clean_input_ids, skip_special_tokens=True)

        all_stripped_texts.append(stripped_text)
        all_ground_truth_labels.append(aligned_labels)

    # Process all examples
    all_predictions = []
    all_true_labels = []
    all_results = []

    logger.info("Running inference on test set...")

    # Process in batches using process_batch method
    for i in tqdm(range(0, len(test_examples), batch_size)):
        batch_examples = test_examples[i : i + batch_size]

        # Get precomputed stripped texts for this batch
        batch_stripped_texts = all_stripped_texts[i : i + batch_size]

        # Run batch inference - returns list of (encoding, labels) tuples
        batch_results = model.process_batch(batch_stripped_texts)

        # For each example in batch, compare predictions with ground truth
        for j, (example, (inputs, pred_labels)) in enumerate(
            zip(batch_examples, batch_results)
        ):
            original_text = example["xml_context"]
            example_idx = i + j
            true_labels = all_ground_truth_labels[example_idx]

            # Filter out special tokens (label = -100) from both
            # Convert integer label IDs to strings for seqeval
            filtered_true = []
            filtered_pred = []

            for true_label, pred_label in zip(true_labels, pred_labels):
                if true_label != -100:  # Skip special tokens
                    # Convert int labels to strings using ID2LABEL
                    filtered_true.append(ID2LABEL[true_label])
                    filtered_pred.append(pred_label)  # pred_label is already a string

            all_true_labels.append(filtered_true)
            all_predictions.append(filtered_pred)

            # Insert predicted tags into stripped text
            predicted_xml = model.insert_tags_into_xml(
                batch_stripped_texts[j],
                inputs,
                [pred_labels],  # Wrap in list for single text
            )

            # Extract citations from both original and predicted XML
            original_citations = extract_citations(original_text)
            predicted_citations = extract_citations(predicted_xml)

            # Store result (without labels - they can be inferred from XML)
            all_results.append(
                {
                    "filename": example["filename"],
                    "original_xml": original_text,
                    "stripped_text": batch_stripped_texts[j],
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
