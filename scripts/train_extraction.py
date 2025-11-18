import logging

from perscit_model.extraction.train import train_pipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

train_pipeline()

# train_pipeline should run as intendended without arguments;
# but see below for overriding defaults, esp. as regards hyperparameters
# and resume_from_checkpoint

# def train(
#    train_path: Path | str,
#    val_path: Path | str | None = None,
#    test_path: Path | str | None = None,
#    output_dir: Path | str | None = None,
#    config_path: Path | str = DEFAULT_CONFIG,
#    resume_from_checkpoint: Path | str | None = None,
#    auto_resume: bool = False,
#    learning_rate: float | None = None,
#    num_epochs: int | None = None,
#    batch_size: int | None = None,
#    warmup_steps: int | None = None,
#    weight_decay: float | None = None,
#    early_stopping_patience: int | None = None,
#    seed: int | None = None,
# ) -> Trainer:
#    """
#    Train citation extraction model.
#
#    Args:
#        train_path: path to training JSONL file
#        val_path: path to validation JSONL file, if applicable
#        test_path: path to testing JSONL file, if applicable
#        output_dir: directory to save model checkpoints (reads from config if None)
#        config_path: path to YAML config file (default: configs/extraction/baseline.yaml)
#        resume_from_checkpoint: optional checkpoint to resume from
#        auto_resume: automatically resume from last checkpoint, defaults to false
#        learning_rate: learning rate for AdamW optimizer
#        num_epochs: number of training epoch
#        batch_size: training batch size per device
#        warmup_steps: number of warmup steps for learning rate scheduler
#        weight_decay: weight decay for regularization
#        early_stopping_patience: patience for early stopping in epochs
#        seed: random seed for reproducible experiments (reads from config if None)
#
#    Returns:
#        Trained Trainer object
#    """
#    # Load config first to get defaults
#    training_config = TrainingConfig.from_yaml(config_path)
#
#
# def train_pipeline(
#    data_dir: Path | str | None = None,
#    config_path: Path | str = DEFAULT_CONFIG,
#    **kwargs,
# ) -> Trainer:
#    """
#    High-level pipeline for training citation extraction model.
#
#    This function handles data discovery, logging setup, and calls train().
#
#    Args:
#        data_dir: Directory containing train.jsonl, val.jsonl, test.jsonl
#            If None, looks in model_data/ directory in project root
#        config_path: Path to training config YAML
#        **kwargs: additional arguments passed to train(), most relevant is probably resume_from_checkpoint, which expect path to weight for checkpoint
#
#    Returns:
#        Trained Trainer object
#    """
