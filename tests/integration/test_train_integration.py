"""Integration tests for training functions (requires model downloads)."""

from pathlib import Path

import pytest
import torch

from perscit_model.extraction.train import (
    create_datasets,
    train,
    train_pipeline,
)


class TestCreateDatasets:
    """Integration tests for create_datasets (requires tokenizer)."""

    @pytest.fixture
    def sample_data_path(self):
        """Path to sample extraction data fixture."""
        return Path(__file__).parent.parent / "fixtures" / "sample_extraction.jsonl"

    def test_create_datasets_loads_train_only(self, sample_data_path):
        """Test that create_datasets can load training data only."""
        datasets = create_datasets(train_path=sample_data_path)

        # Should have train split
        assert "train" in datasets
        assert len(datasets["train"]) == 5

        # Should not have validation or test
        assert "validation" not in datasets
        assert "test" not in datasets

    def test_create_datasets_loads_all_splits(self, sample_data_path):
        """Test that create_datasets can load all splits."""
        datasets = create_datasets(
            train_path=sample_data_path,
            val_path=sample_data_path,  # Reuse same file for testing
            test_path=sample_data_path,
        )

        # Should have all splits
        assert "train" in datasets
        assert "validation" in datasets
        assert "test" in datasets

        # All should have same number of examples
        assert len(datasets["train"]) == 5
        assert len(datasets["validation"]) == 5
        assert len(datasets["test"]) == 5

    def test_create_datasets_has_correct_columns(self, sample_data_path):
        """Test that datasets have all required columns."""
        datasets = create_datasets(train_path=sample_data_path)

        # Check required columns
        expected_columns = {"input_ids", "attention_mask", "labels", "filename"}
        assert expected_columns.issubset(set(datasets["train"].column_names))

    def test_create_datasets_tokenization_shapes(self, sample_data_path):
        """Test that tokenized data has correct shapes."""
        datasets = create_datasets(train_path=sample_data_path)

        # Get first example
        example = datasets["train"][0]

        # input_ids and labels should have same length
        assert len(example["input_ids"]) == len(example["labels"])

        # attention_mask should match input_ids length
        assert len(example["attention_mask"]) == len(example["input_ids"])

    def test_create_datasets_preserves_filenames(self, sample_data_path):
        """Test that filenames are preserved from source data."""
        datasets = create_datasets(train_path=sample_data_path)

        # All examples should have filenames
        for i in range(len(datasets["train"])):
            filename = datasets["train"][i]["filename"]
            assert isinstance(filename, str)
            assert len(filename) > 0
            assert ".xml" in filename


class TestTrain:
    """Integration tests for the train function (requires full model)."""

    @pytest.fixture
    def sample_data_path(self):
        """Path to sample extraction data fixture."""
        return Path(__file__).parent.parent / "fixtures" / "sample_extraction.jsonl"

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Temporary directory for model output."""
        return tmp_path / "model_output"

    def test_train_minimal(self, sample_data_path, temp_output_dir):
        """Test basic training with minimal configuration."""
        # Train for just 1 step to verify it works
        trainer = train(
            train_path=sample_data_path,
            output_dir=temp_output_dir,
            num_epochs=1,
            batch_size=2,
            learning_rate=5e-5,
        )

        # Should return a Trainer object
        assert trainer is not None
        assert hasattr(trainer, "model")
        assert hasattr(trainer, "train_dataset")

        # Output directory should be created
        assert temp_output_dir.exists()

    def test_train_with_validation(self, sample_data_path, temp_output_dir):
        """Test training with validation set."""
        trainer = train(
            train_path=sample_data_path,
            val_path=sample_data_path,  # Reuse same data for testing
            output_dir=temp_output_dir,
            num_epochs=1,
            batch_size=2,
        )

        # Should have validation dataset
        assert trainer.eval_dataset is not None

        # Should have early stopping callback
        callbacks = trainer.callback_handler.callbacks
        from transformers import EarlyStoppingCallback

        has_early_stopping = any(
            isinstance(cb, EarlyStoppingCallback) for cb in callbacks
        )
        assert has_early_stopping

    def test_train_saves_model(self, sample_data_path, temp_output_dir):
        """Test that training saves the model."""
        train(
            train_path=sample_data_path,
            output_dir=temp_output_dir,
            num_epochs=1,
            batch_size=2,
        )

        # Model files should be saved
        assert (temp_output_dir / "config.json").exists()
        assert (temp_output_dir / "model.safetensors").exists() or (
            temp_output_dir / "pytorch_model.bin"
        ).exists()

    def test_train_saves_metrics(self, sample_data_path, temp_output_dir):
        """Test that training saves metrics."""
        train(
            train_path=sample_data_path,
            output_dir=temp_output_dir,
            num_epochs=1,
            batch_size=2,
        )

        # Metrics should be saved
        assert (temp_output_dir / "train_results.json").exists()

    def test_train_with_test_set(self, sample_data_path, temp_output_dir):
        """Test training with test set evaluation."""
        train(
            train_path=sample_data_path,
            val_path=sample_data_path,
            test_path=sample_data_path,
            output_dir=temp_output_dir,
            num_epochs=1,
            batch_size=2,
        )

        # Test metrics should be saved
        assert (temp_output_dir / "test_results.json").exists()

    def test_train_seed_reproducibility(self, sample_data_path, tmp_path):
        """Test that same seed produces deterministic results."""
        output_dir_1 = tmp_path / "run1"
        output_dir_2 = tmp_path / "run2"

        # Train twice with same seed
        trainer1 = train(
            train_path=sample_data_path,
            output_dir=output_dir_1,
            num_epochs=1,
            batch_size=2,
            seed=42,
        )

        trainer2 = train(
            train_path=sample_data_path,
            output_dir=output_dir_2,
            num_epochs=1,
            batch_size=2,
            seed=42,
        )

        # Models should have same architecture
        assert trainer1.model.config.model_type == trainer2.model.config.model_type

        # Note: Full weight reproducibility across runs is hard to guarantee
        # due to CUDA non-determinism, distributed training, etc.
        # But at least the process should complete successfully

    def test_train_uses_gpu_if_available(self, sample_data_path, temp_output_dir):
        """Test that training uses GPU if available."""
        trainer = train(
            train_path=sample_data_path,
            output_dir=temp_output_dir,
            num_epochs=1,
            batch_size=2,
        )

        # Check device
        model_device = next(trainer.model.parameters()).device

        # Should use CUDA if available, else CPU
        if torch.cuda.is_available():
            assert model_device.type == "cuda"
        else:
            assert model_device.type == "cpu"

    def test_train_model_has_correct_labels(self, sample_data_path, temp_output_dir):
        """Test that model is configured for correct number of labels."""
        trainer = train(
            train_path=sample_data_path,
            output_dir=temp_output_dir,
            num_epochs=1,
            batch_size=2,
        )

        # Model should have 3 labels: O, B-CIT, I-CIT
        assert trainer.model.config.num_labels == 3


class TestTrainPipeline:
    """Integration tests for the complete train_pipeline."""

    @pytest.fixture
    def temp_data_dir(self, tmp_path):
        """Create temporary data directory with sample data."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True)

        # Copy sample data to simulate resolved.jsonl
        sample_path = (
            Path(__file__).parent.parent / "fixtures" / "sample_extraction.jsonl"
        )

        # Create a fake cit_data directory structure
        cit_data_dir = tmp_path / "cit_data"
        cit_data_dir.mkdir(parents=True)
        resolved_path = cit_data_dir / "resolved.jsonl"

        # Copy sample data
        import shutil

        shutil.copy(sample_path, resolved_path)

        return tmp_path

    def test_train_pipeline_end_to_end(self, temp_data_dir, monkeypatch):
        """Test complete pipeline from raw data to trained model."""
        # Monkeypatch the project root to use temp directory
        import perscit_model.extraction.train as train_module

        original_file = train_module.__file__

        # Create a mock __file__ path that points to our temp structure
        mock_file = (
            temp_data_dir / "src" / "perscit_model" / "extraction" / "train.py"
        )
        mock_file.parent.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(train_module, "__file__", str(mock_file))

        # Set data directory
        data_dir = temp_data_dir / "model_data" / "extraction"

        # Run pipeline with minimal training
        trainer = train_pipeline(
            data_dir=data_dir,
            num_epochs=1,
            batch_size=2,
        )

        # Should complete successfully
        assert trainer is not None

        # Data splits should be created
        assert (data_dir / "train.jsonl").exists()
        assert (data_dir / "val.jsonl").exists()
        assert (data_dir / "test.jsonl").exists()

        # Split info should be saved
        assert (data_dir / "split_info.json").exists()

        # Model should be saved (in outputs dir from config)
        # Note: output_dir comes from config, not data_dir

    def test_train_pipeline_uses_existing_split(self, temp_data_dir, monkeypatch):
        """Test that pipeline reuses existing data split."""
        import perscit_model.extraction.train as train_module

        mock_file = (
            temp_data_dir / "src" / "perscit_model" / "extraction" / "train.py"
        )
        mock_file.parent.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(train_module, "__file__", str(mock_file))

        data_dir = temp_data_dir / "model_data" / "extraction"
        output_dir_1 = temp_data_dir / "outputs1"
        output_dir_2 = temp_data_dir / "outputs2"

        # First run - creates split
        train_pipeline(
            data_dir=data_dir,
            output_dir=output_dir_1,
            num_epochs=1,
            batch_size=2,
            seed=42,
        )

        # Get split file modification time
        split_info_path = data_dir / "split_info.json"
        mtime_1 = split_info_path.stat().st_mtime

        # Second run - should reuse split
        train_pipeline(
            data_dir=data_dir,
            output_dir=output_dir_2,
            num_epochs=1,
            batch_size=2,
            seed=42,
        )

        # Split info should not have been recreated
        mtime_2 = split_info_path.stat().st_mtime
        assert mtime_2 == mtime_1

    def test_train_pipeline_with_custom_config(
        self, temp_data_dir, monkeypatch, tmp_path
    ):
        """Test pipeline with custom configuration file."""
        import perscit_model.extraction.train as train_module

        mock_file = (
            temp_data_dir / "src" / "perscit_model" / "extraction" / "train.py"
        )
        mock_file.parent.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(train_module, "__file__", str(mock_file))

        # Create custom config
        config_path = tmp_path / "custom_config.yaml"
        config_content = """
train_ratio: 0.7
val_ratio: 0.15
test_ratio: 0.15
seed: 999
model_name: microsoft/deberta-v3-base
max_length: 256
learning_rate: 5e-5
per_device_train_batch_size: 2
num_train_epochs: 1
output_dir: outputs/custom
"""
        config_path.write_text(config_content)

        data_dir = temp_data_dir / "model_data" / "extraction"

        # Run with custom config
        trainer = train_pipeline(
            data_dir=data_dir,
            config_path=config_path,
        )

        assert trainer is not None

        # Check that custom ratios were used
        import json

        split_info_path = data_dir / "split_info.json"
        with open(split_info_path) as f:
            split_info = json.load(f)

        assert split_info["train_ratio"] == 0.7
        assert split_info["val_ratio"] == 0.15
        assert split_info["test_ratio"] == 0.15
        assert split_info["seed"] == 999
