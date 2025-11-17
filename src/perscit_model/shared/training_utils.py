import inspect
import json
import yaml
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional, Union

from transformers import (
    EarlyStoppingCallback,
    TrainingArguments,
    TrainerCallback,
)


@dataclass
class TrainingConfig:
    """Training configuration that maps to TrainingArguments."""

    # Fields that map directly to TrainingArguments
    output_dir: str = "outputs"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    eval_strategy: str = "epoch"
    eval_steps: Optional[int] = None
    save_strategy: str = "epoch"
    save_steps: Optional[int] = None
    save_total_limit: int = 3
    logging_strategy: str = "steps"
    logging_steps: int = 100
    report_to: str = "none"
    fp16: bool = True
    dataloader_num_workers: int = 4
    gradient_accumulation_steps: int = 1
    seed: int = 42
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    remove_unused_columns: bool = True
    disable_tqdm: bool = False

    # Custom fields NOT in TrainingArguments
    model_name: str = "microsoft/deberta-v3-base"
    max_length: int = 512
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0001

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TrainingConfig":
        """Load config from YAML file."""
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, path: Union[str, Path]):
        """Save config to YAML file."""
        with open(path, "w") as f:
            yaml.dump(asdict(self), f)

    def get_training_args(self) -> dict:
        """Extract only fields that TrainingArguments accepts."""
        # Get TrainingArguments signature
        training_args_params = set(
            inspect.signature(TrainingArguments.__init__).parameters.keys()
        )
        config_dict = asdict(self)

        # Filter to only TrainingArguments parameters
        return {k: v for k, v in config_dict.items() if k in training_args_params}


def get_training_arguments(config: TrainingConfig) -> TrainingArguments:
    """
    Create TrainingArguments from config.

    Args:
        config: Training configuration

    Returns:
        TrainingArguments instance
    """
    return TrainingArguments(**config.get_training_args())


def get_early_stopping_callback(config: TrainingConfig) -> EarlyStoppingCallback:
    """
    Create early stopping callback from config.

    Args:
        config: Training configuration

    Returns:
        EarlyStoppingCallback instance
    """
    return EarlyStoppingCallback(
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_threshold=config.early_stopping_threshold,
    )


class LoggingCallback(TrainerCallback):
    """Custom callback for enhanced logging."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log training progress."""
        if logs is not None:
            # Filter out None values
            logs = {k: v for k, v in logs.items() if v is not None}

            # Format and print key metrics
            if "loss" in logs:
                print(f"Step {state.global_step}: loss={logs['loss']:.4f}")
            if "eval_loss" in logs:
                print(f"Eval loss: {logs['eval_loss']:.4f}")


class CheckpointCallback(TrainerCallback):
    """Callback to save additional checkpoint info."""

    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_save(self, args, state, control, **kwargs):
        """Save additional metadata when checkpoint is saved."""
        metadata = {
            "global_step": state.global_step,
            "epoch": state.epoch,
            "best_metric": state.best_metric,
            "best_model_checkpoint": state.best_model_checkpoint,
        }

        metadata_file = self.save_dir / f"checkpoint_{state.global_step}_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)


def get_default_callbacks(
    config: TrainingConfig,
    save_dir: Optional[Path] = None,
) -> list[TrainerCallback]:
    """
    Get default callbacks for training.

    Args:
        config: Training configuration
        save_dir: Directory to save checkpoint metadata

    Returns:
        List of callbacks
    """
    callbacks = [
        get_early_stopping_callback(config),
        LoggingCallback(),
    ]

    if save_dir:
        callbacks.append(CheckpointCallback(save_dir))

    return callbacks


def create_optimizer_and_scheduler(
    model,
    training_args: TrainingArguments,
    num_training_steps: int,
):
    """
    Create optimizer and learning rate scheduler.

    Args:
        model: The model to optimize
        training_args: Training arguments
        num_training_steps: Total number of training steps

    Returns:
        Tuple of (optimizer, scheduler)
    """
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    # Separate parameters for weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=training_args.learning_rate,
        eps=1e-8,
    )

    # Calculate warmup steps
    num_warmup_steps = int(num_training_steps * training_args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    return optimizer, scheduler


def compute_metrics_wrapper(
    metric_fn: Callable,
    label_list: Optional[list[str]] = None,
) -> Callable:
    """
    Create a compute_metrics function for Trainer.

    Args:
        metric_fn: Function that takes (predictions, labels) and returns dict of metrics
        label_list: Optional list of label names for decoding

    Returns:
        Function compatible with Trainer's compute_metrics parameter
    """

    def compute_metrics(eval_pred):
        """Compute metrics from EvalPrediction."""
        predictions, labels = eval_pred

        # predictions might be logits, so get argmax
        if len(predictions.shape) > 1:
            predictions = predictions.argmax(axis=-1)

        return metric_fn(predictions, labels, label_list=label_list)

    return compute_metrics
