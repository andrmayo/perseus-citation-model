"""Training functions for citation extraction model (BIO tagging)."""

import json
import logging
import random
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import cast

import torch
from datasets import DatasetDict
from seqeval.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from perscit_model.extraction.data_loader import (
    ID2LABEL,
    ExtractionDataLoader,
    create_extraction_dataset,
)
from perscit_model.extraction.model import create_model
from perscit_model.shared.data_loader import DEFAULT_CONFIG
from perscit_model.shared.training_utils import TrainingConfig

logger = logging.getLogger(__name__)


def _count_tags_in_text(text: str) -> tuple[int, int]:
    """Count <bibl> and <quote> opening tags in text.

    Returns:
        Tuple of (bibl_count, quote_count)
    """
    bibl_count = len(re.findall(r"<bibl[^>]*>", text))
    quote_count = len(re.findall(r"<quote[^>]*>", text))
    return bibl_count, quote_count


def split_data(
    input_file: Path | str,
    output_dir: Path | str,
    train_ratio: float | None = None,
    val_ratio: float | None = None,
    test_ratio: float | None = None,
    config_path: Path | str = DEFAULT_CONFIG,
    seed: int | None = None,
    force_redo: bool = False,
    label_tolerance: float = 0.10,
) -> tuple[Path, Path, Path]:
    """
    Split JSONL data into train/val/test sets with stratification by label distribution.

    Args:
        input_file: Path to input JSONL file
        output_dir: Directory to save train.jsonl, val.jsonl, test.jsonl
        train_ratio: Proportion for training (reads from config if None)
        val_ratio: Proportion for validation (reads from config if None)
        test_ratio: Proportion for test (reads from config if None)
        config_path: Path to training config YAML (for reading split ratios and seed)
        seed: Random seed for reproducibility (reads from config if None)
        force_redo: If True, always redo the split even if existing split matches
        label_tolerance: Max allowed deviation of QUOTE ratio from overall ratio (default 10%)

    Returns:
        Tuple of (train_path, val_path, test_path)
    """

    # Load config for defaults
    config = TrainingConfig.from_yaml(config_path)

    # Load ratios and seed from config if not provided
    train_ratio = train_ratio if train_ratio is not None else config.train_ratio
    val_ratio = val_ratio if val_ratio is not None else config.val_ratio
    test_ratio = test_ratio if test_ratio is not None else config.test_ratio
    seed = seed if seed is not None else config.seed

    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        )

    input_file = Path(input_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if split info file exists
    split_info_path = output_dir / "split_info.json"
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    test_path = output_dir / "test.jsonl"

    # Define current split configuration
    current_split_config = {
        "split_strategy": "by_filename",  # Split by file to prevent data leakage
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
        "input_file": str(input_file.absolute()),
    }

    # Check if we need to redo the split
    needs_resplit = force_redo

    if split_info_path.exists() and not force_redo:
        # Load existing split info
        with open(split_info_path) as f:
            existing_split_config = json.load(f)

        # Compare configurations
        if existing_split_config != current_split_config:
            logger.info("Split configuration has changed, re-splitting data...")
            logger.info(f"Old config: {existing_split_config}")
            logger.info(f"New config: {current_split_config}")
            needs_resplit = True
        else:
            # Config matches, check if split files exist
            if train_path.exists() and val_path.exists() and test_path.exists():
                logger.info("Using existing split (configuration matches)")
                return (train_path, val_path, test_path)
            else:
                logger.info("Split files missing, re-splitting...")
                needs_resplit = True
    elif not split_info_path.exists():
        logger.info("No existing split info found, creating new split...")
        needs_resplit = True

    # Delete existing split files if re-splitting
    if needs_resplit:
        for path in [train_path, val_path, test_path]:
            if path.exists():
                path.unlink()
                logger.info(f"Deleted existing file: {path}")

    # Load all data and group by filename to prevent data leakage
    logger.info(f"Loading data from {input_file}")
    from collections import defaultdict

    examples_by_file: dict[str, list[dict]] = defaultdict(list)
    tags_by_file: dict[str, tuple[int, int]] = {}  # filename -> (bibl_count, quote_count)

    with open(input_file) as f:
        for line in f:
            example = json.loads(line)
            filename = example.get("filename", "")
            examples_by_file[filename].append(example)

            # Count tags in this example (check both window_text and xml_context fields)
            text = example.get("window_text", "") or example.get("xml_context", "")
            bibl, quote = _count_tags_in_text(text)
            prev_bibl, prev_quote = tags_by_file.get(filename, (0, 0))
            tags_by_file[filename] = (prev_bibl + bibl, prev_quote + quote)

    total_examples = sum(len(examples) for examples in examples_by_file.values())
    total_bibl = sum(t[0] for t in tags_by_file.values())
    total_quote = sum(t[1] for t in tags_by_file.values())
    total_tags = total_bibl + total_quote
    overall_quote_ratio = total_quote / total_tags if total_tags > 0 else 0.0

    logger.info(
        f"Loaded {total_examples} examples from {len(examples_by_file)} unique files"
    )
    if total_tags > 0:
        logger.info(
            f"Overall tag distribution: BIBL={total_bibl} ({100*total_bibl/total_tags:.1f}%), "
            f"QUOTE={total_quote} ({100*total_quote/total_tags:.1f}%)"
        )
    else:
        logger.info("No BIBL/QUOTE tags found in data, skipping label stratification")

    # Split by files (not individual examples) to prevent overlapping contexts
    # Try multiple shuffles to find a split that's close to target ratios AND label distribution
    random.seed(seed)
    filenames = list(examples_by_file.keys())

    max_attempts = 100
    tolerance = 0.02  # Allow 2% deviation from target example ratios
    best_split = None
    best_score = float("inf")  # Combined score of example + label deviation

    for attempt in range(max_attempts):
        random.shuffle(filenames)

        # Calculate split indices for files
        n_train_files = int(len(filenames) * train_ratio)
        n_val_files = int(len(filenames) * val_ratio)

        # Split filenames into groups
        train_files = filenames[:n_train_files]
        val_files = filenames[n_train_files : n_train_files + n_val_files]
        test_files = filenames[n_train_files + n_val_files :]

        # Count examples in each split (fast - just sum lengths)
        train_count = sum(len(examples_by_file[f]) for f in train_files)
        val_count = sum(len(examples_by_file[f]) for f in val_files)
        test_count = sum(len(examples_by_file[f]) for f in test_files)

        # Calculate actual example ratios
        actual_train_ratio = train_count / total_examples
        actual_val_ratio = val_count / total_examples
        actual_test_ratio = test_count / total_examples

        # Calculate example ratio deviation
        max_example_deviation = max(
            abs(actual_train_ratio - train_ratio),
            abs(actual_val_ratio - val_ratio),
            abs(actual_test_ratio - test_ratio),
        )

        # Calculate QUOTE ratio for each split
        def get_quote_ratio(file_list: list[str]) -> float:
            bibl = sum(tags_by_file.get(f, (0, 0))[0] for f in file_list)
            quote = sum(tags_by_file.get(f, (0, 0))[1] for f in file_list)
            total = bibl + quote
            return quote / total if total > 0 else 0.0

        train_quote_ratio = get_quote_ratio(train_files)
        val_quote_ratio = get_quote_ratio(val_files)
        test_quote_ratio = get_quote_ratio(test_files)

        # Calculate label deviation from overall ratio
        max_label_deviation = max(
            abs(train_quote_ratio - overall_quote_ratio),
            abs(val_quote_ratio - overall_quote_ratio),
            abs(test_quote_ratio - overall_quote_ratio),
        )

        # Combined score: prioritize example ratio but also consider label balance
        combined_score = max_example_deviation + max_label_deviation

        # Keep track of best split
        if combined_score < best_score:
            best_score = combined_score
            best_split = (
                train_files,
                val_files,
                test_files,
                train_count,
                val_count,
                test_count,
                max_example_deviation,
                max_label_deviation,
                train_quote_ratio,
                val_quote_ratio,
                test_quote_ratio,
            )

        # If within tolerance, use this split
        # Skip label check if no tags found (total_tags == 0)
        label_ok = total_tags == 0 or max_label_deviation <= label_tolerance
        if max_example_deviation <= tolerance and label_ok:
            if total_tags > 0:
                logger.info(
                    f"Found acceptable split after {attempt + 1} attempts "
                    f"(example deviation: {max_example_deviation:.1%}, "
                    f"label deviation: {max_label_deviation:.1%})"
                )
            else:
                logger.info(
                    f"Found acceptable split after {attempt + 1} attempts "
                    f"(example deviation: {max_example_deviation:.1%})"
                )
            break
    else:
        # Use best split found
        if best_split is None:
            raise ValueError("No valid split found")
        (
            train_files,
            val_files,
            test_files,
            train_count,
            val_count,
            test_count,
            best_example_dev,
            best_label_dev,
            train_quote_ratio,
            val_quote_ratio,
            test_quote_ratio,
        ) = best_split
        if total_tags > 0:
            logger.warning(
                f"Could not find split within tolerances after {max_attempts} attempts. "
                f"Using best split (example dev: {best_example_dev:.1%}, label dev: {best_label_dev:.1%})"
            )
        else:
            logger.warning(
                f"Could not find split within tolerances after {max_attempts} attempts. "
                f"Using best split (example dev: {best_example_dev:.1%})"
            )

    # Log the actual distribution
    logger.info(
        f"Split ratios - Train: {train_count}/{total_examples} ({train_count / total_examples:.1%}), "
        f"Val: {val_count}/{total_examples} ({val_count / total_examples:.1%}), "
        f"Test: {test_count}/{total_examples} ({test_count / total_examples:.1%})"
    )
    if total_tags > 0:
        logger.info(
            f"QUOTE ratios - Train: {train_quote_ratio:.1%}, Val: {val_quote_ratio:.1%}, "
            f"Test: {test_quote_ratio:.1%} (target: {overall_quote_ratio:.1%})"
        )

    # Collect all examples for each split (now we know it's balanced)
    train_data = [ex for f in train_files for ex in examples_by_file[f]]
    val_data = [ex for f in val_files for ex in examples_by_file[f]]
    test_data = [ex for f in test_files for ex in examples_by_file[f]]

    # Save splits
    paths = []
    for split_name, split_data in [
        ("train", train_data),
        ("val", val_data),
        ("test", test_data),
    ]:
        output_path = output_dir / f"{split_name}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for example in split_data:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        logger.info(f"{split_name}: {len(split_data)} examples -> {output_path}")
        paths.append(output_path)

    # Save split configuration info
    with open(split_info_path, "w") as f:
        json.dump(current_split_config, f, indent=2)
    logger.info(f"Split configuration saved to {split_info_path}")

    return tuple(paths)


def create_datasets(
    train_path: Path | str,
    val_path: Path | str | None = None,
    test_path: Path | str | None = None,
    config_path: Path | str | None = None,
    tag_retention_prob: float | None = None,
) -> DatasetDict:
    """
    Create train/val/test datasets from JSONL files.

    Returns:
        DatasetDict with train/validation/test splits
    """
    logger.info("Creating datasets...")

    datasets = {}

    logger.info(f"Loading training data from {train_path}")
    datasets["train"] = create_extraction_dataset(
        train_path, config_path, tag_retention_prob=tag_retention_prob
    )
    logger.info(f"Training examples: {len(datasets['train'])}")

    # Create validation dataset
    if val_path:
        if not isinstance(val_path, Path):
            val_path = Path(val_path)
        if not val_path.exists():
            logger.warning(
                "No valid val_path provided to create_datasets, not creating validation dataset"
            )
        else:
            logger.info(f"Loading validation data from {val_path}")
            datasets["validation"] = create_extraction_dataset(
                val_path, config_path, tag_retention_prob=tag_retention_prob
            )
            logger.info(f"Validation examples: {len(datasets['validation'])}")
    else:
        logger.warning(
            "No valid val_path provided to create_datasets, not creating validation dataset"
        )

    if test_path:
        if not isinstance(test_path, Path):
            test_path = Path(test_path)
        if not test_path.exists():
            logger.warning(
                "No valid test_path provided to create_datasets, not creating test dataset"
            )
        else:
            logger.info(f"Loading test data from {test_path}")
            datasets["test"] = create_extraction_dataset(
                test_path, config_path, tag_retention_prob=tag_retention_prob
            )
            logger.info(f"Test examples: {len(datasets['test'])}")

    return DatasetDict(datasets)


def make_compute_metrics(model=None):
    """
    Create a compute_metrics function, optionally using CRF Viterbi decoding.

    When a CRF model is provided, predictions are decoded using Viterbi algorithm
    instead of argmax, ensuring training metrics match inference behavior.

    Args:
        model: Optional model with CRF layer. If provided and has decode() method,
            uses Viterbi decoding. Otherwise falls back to argmax.

    Returns:
        compute_metrics function for HuggingFace Trainer
    """

    def compute_metrics(eval_pred):
        """
        Compute evaluation metrics for BIO tagging.

        Uses seqeval for BIO tag evaluation (entity-level F1).

        Args:
            eval_pred: Tuple of (predictions/logits, labels)

        Returns:
            Dict of metrics
        """
        logits, labels = eval_pred

        # Use CRF Viterbi decoding if model has CRF layer, otherwise argmax
        if model is not None and hasattr(model, "decode"):
            # CRF decode expects torch tensors on same device as model
            device = next(model.parameters()).device
            logits_tensor = torch.tensor(logits, device=device)
            # Create mask from labels (non -100 positions are valid)
            # But CRF needs attention_mask style (1 for real tokens, 0 for padding)
            # Since labels has -100 for padding AND special tokens, we need to be careful
            # Use all positions as valid for CRF, then filter by labels later
            mask = torch.ones(logits_tensor.shape[:2], dtype=torch.bool, device=device)
            predictions = model.decode(logits_tensor, mask)
        else:
            # Fallback to argmax
            predictions = logits.argmax(axis=-1)

        # Convert IDs to label strings, removing -100 (special tokens)
        true_labels = []
        true_predictions = []

        for pred_seq, label_seq in zip(predictions, labels):
            true_label = []
            true_pred = []
            for pred, lab in zip(pred_seq, label_seq):
                if lab != -100:  # Ignore special tokens
                    true_label.append(ID2LABEL[lab])
                    true_pred.append(ID2LABEL[pred])
            true_labels.append(true_label)
            true_predictions.append(true_pred)

        # Compute metrics
        results = {
            "accuracy": accuracy_score(true_labels, true_predictions),
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }

        # Log detailed classification report
        logger.info(
            "\n" + cast(str, classification_report(true_labels, true_predictions))
        )

        return results

    return compute_metrics


def train(
    train_path: Path | str,
    val_path: Path | str | None = None,
    test_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    config_path: Path | str = DEFAULT_CONFIG,
    resume_from_checkpoint: Path | str | None = None,
    auto_resume: bool = False,
    pretrained_model_path: Path | str | None = None,
    learning_rate: float | None = None,
    num_epochs: int | None = None,
    batch_size: int | None = None,
    warmup_steps: int | None = None,
    weight_decay: float | None = None,
    early_stopping_patience: int | None = None,
    seed: int | None = None,
    tag_retention_prob: float | None = None,
    max_steps: int | None = None,
) -> Trainer:
    """
    Train citation extraction model.

    Args:
        train_path: path to training JSONL file
        val_path: path to validation JSONL file, if applicable
        test_path: path to testing JSONL file, if applicable
        output_dir: directory to save model checkpoints (reads from config if None)
        config_path: path to YAML config file (default: configs/extraction/baseline.yaml)
        resume_from_checkpoint: optional checkpoint to resume from (includes optimizer state, epoch counter, NOT for curriculum learning)
        auto_resume: automatically resume from last checkpoint, defaults to false
        pretrained_model_path: optional pretrained model to initialize weights from (use this for curriculum learning)
        learning_rate: learning rate for AdamW optimizer
        num_epochs: number of training epoch
        batch_size: training batch size per device
        warmup_steps: number of warmup steps for learning rate scheduler
        weight_decay: weight decay for regularization
        early_stopping_patience: patience for early stopping in epochs
        seed: random seed for reproducible experiments (reads from config if None)
        max_steps: maximum number of training steps (overrides num_epochs if set)

    Returns:
        Trained Trainer object
    """
    # Load config first to get defaults
    training_config = TrainingConfig.from_yaml(config_path)

    # Use config seed if not provided
    if seed is None:
        seed = training_config.seed

    random.seed(seed)
    torch.manual_seed(seed)

    datasets = create_datasets(
        train_path,
        val_path,
        test_path,
        config_path,
        tag_retention_prob,
    )

    logger.info("Creating model...")
    loader = ExtractionDataLoader(config_path=config_path)
    model = create_model(
        loader.tokenizer,
        config_path=config_path,
        pretrained_model_path=pretrained_model_path,
        use_crf=training_config.use_crf,
    )

    logger.info(f"Model: {model.config.model_type}")
    logger.info(f"Model uses CRF: {training_config.use_crf}")
    logger.info(f"Vocabulary size: {loader.tokenizer.vocab_size}")
    logger.info(f"Number of parameters: {model.num_parameters()}")

    config = training_config.get_training_args()
    cli_overrides = {
        "output_dir": str(output_dir) if isinstance(output_dir, Path) else output_dir,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "num_train_epochs": num_epochs,
        "max_steps": max_steps,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "seed": seed,
        # Conditional overrides
        "eval_strategy": "epoch" if val_path else "no",
        "load_best_model_at_end": True if val_path else False,
        "metric_for_best_model": "eval_f1" if val_path else None,
    }
    # Only update if value is not None
    for key, value in cli_overrides.items():
        if value is not None:
            config[key] = value

    training_args = TrainingArguments(**config)

    # for handling padding
    data_collator = DataCollatorForTokenClassification(
        tokenizer=loader.tokenizer, padding=True
    )

    # Determine early stopping patience
    patience = (
        early_stopping_patience
        if early_stopping_patience is not None
        else training_config.early_stopping_patience
    )

    # Create compute_metrics with model for CRF Viterbi decoding (if CRF is enabled)
    compute_metrics_fn = make_compute_metrics(model if training_config.use_crf else None)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets.get("validation"),
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)]
        if val_path
        else [],
    )

    # resume from checkpoint if provided
    if resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
    elif auto_resume:
        # try to find last checkpoint in output_dir
        resume_from_checkpoint = get_last_checkpoint(training_args.output_dir)
        if resume_from_checkpoint:
            logger.info(
                f"Found checkpoint at {resume_from_checkpoint}, resuming training..."
            )

    # Train
    logger.info("Starting training...")
    train_results = (
        trainer.train(resume_from_checkpoint=str(resume_from_checkpoint))
        if resume_from_checkpoint
        else trainer.train()
    )

    # Save final model to timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if training_args.output_dir is None:
        raise ValueError
    final_model_dir = Path(training_args.output_dir) / f"final-model-{timestamp}"
    logger.info(f"Saving final model to {final_model_dir}")
    trainer.save_model(str(final_model_dir))

    # Save training metrics in the final model directory
    metrics = train_results.metrics
    trainer.log_metrics("train", metrics)
    # Save metrics to final model dir as well
    import json

    with open(final_model_dir / "train_results.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Save state to final model dir
    trainer.save_state()
    # Also copy state to final model dir
    state_file = Path(training_args.output_dir) / "trainer_state.json"
    if state_file.exists():
        shutil.copy(state_file, final_model_dir / "trainer_state.json")

    # Eval on validation set
    val_metrics = None
    if val_path:
        logger.info("Evaluating on validation set...")
        val_metrics = trainer.evaluate()
        trainer.log_metrics("eval", val_metrics)
        # Save to final model dir
        with open(final_model_dir / "eval_results.json", "w") as f:
            json.dump(val_metrics, f, indent=4)

    # Eval on test set
    test_metrics = None
    if test_path:
        logger.info("Evaluating on test set ...")
        # Remove early stopping callback to avoid spurious warning during test evaluation
        trainer.callback_handler.callbacks = [
            cb
            for cb in trainer.callback_handler.callbacks
            if not isinstance(cb, EarlyStoppingCallback)
        ]
        test_metrics = trainer.evaluate(datasets["test"], metric_key_prefix="test")  # type: ignore[arg-type]
        trainer.log_metrics("test", test_metrics)
        # Save to final model dir
        with open(final_model_dir / "test_results.json", "w") as f:
            json.dump(test_metrics, f, indent=4)

    # Save all results together
    all_results = {**metrics}
    if val_path and val_metrics is not None:
        all_results.update(val_metrics)
    if test_path and test_metrics is not None:
        all_results.update(test_metrics)
    with open(final_model_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=4)

    logger.info(f"Training complete. Final model saved to {final_model_dir}")
    return trainer


def train_pipeline(
    data_dir: Path | str | None = None,
    config_path: Path | str = DEFAULT_CONFIG,
    src_path: Path | str = Path(__file__).parent.parent.parent.parent
    / "cit_data/xml_files/window_data.jsonl",
    **kwargs,
) -> Trainer:
    """
    High-level pipeline for training citation extraction model.

    This function handles data discovery, logging setup, and calls train().

    Args:
        data_dir: Directory containing train.jsonl, val.jsonl, test.jsonl
            If None, looks in model_data/ directory in project root
        config_path: Path to training config YAML
        src_path: Defaults to Path to file with 512-token windows from XML files in jsonl format
        **kwargs: additional arguments passed to train(), most relevant is probably resume_from_checkpoint, which expect path to weight for checkpoint

    Returns:
        Trained Trainer object

    Note: src_path needs to be provided as an explicit argument for curriculum learning.
    """

    if data_dir is None:
        data_dir = (
            Path(__file__).parent.parent.parent.parent / "model_data" / "extraction"
        )
    else:
        data_dir = Path(data_dir)

    # Split data (creates train/val/test files)
    train_path, val_path, test_path = split_data(
        src_path,
        output_dir=data_dir,
        config_path=config_path,
    )

    return train(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        config_path=config_path,
        **kwargs,
    )
