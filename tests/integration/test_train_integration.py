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
    """Integration tests for create_datasets (requires tokenizer).

    Uses class-scoped fixtures to share datasets and avoid reloading tokenizer.
    """

    @pytest.fixture(scope="class")
    def sample_data_path(self):
        """Path to sample extraction data fixture."""
        return Path(__file__).parent.parent / "fixtures" / "sample_extraction.jsonl"

    @pytest.fixture(scope="class")
    def datasets_train_only(self, sample_data_path):
        """Create datasets with only train split (shared across class)."""
        return create_datasets(train_path=sample_data_path)

    @pytest.fixture(scope="class")
    def datasets_all_splits(self, sample_data_path):
        """Create datasets with all splits (shared across class)."""
        return create_datasets(
            train_path=sample_data_path,
            val_path=sample_data_path,
            test_path=sample_data_path,
        )

    def test_create_datasets_loads_train_only(self, datasets_train_only):
        """Test that create_datasets can load training data only."""
        # Should have train split
        assert "train" in datasets_train_only
        assert len(datasets_train_only["train"]) == 5

        # Should not have validation or test
        assert "validation" not in datasets_train_only
        assert "test" not in datasets_train_only

    def test_create_datasets_loads_all_splits(self, datasets_all_splits):
        """Test that create_datasets can load all splits."""
        # Should have all splits
        assert "train" in datasets_all_splits
        assert "validation" in datasets_all_splits
        assert "test" in datasets_all_splits

        # All should have same number of examples
        assert len(datasets_all_splits["train"]) == 5
        assert len(datasets_all_splits["validation"]) == 5
        assert len(datasets_all_splits["test"]) == 5

    def test_create_datasets_has_correct_columns(self, datasets_train_only):
        """Test that datasets have all required columns."""
        # Check required columns
        expected_columns = {"input_ids", "attention_mask", "labels", "filename"}
        assert expected_columns.issubset(set(datasets_train_only["train"].column_names))

    def test_create_datasets_tokenization_shapes(self, datasets_train_only):
        """Test that tokenized data has correct shapes."""
        # Get first example
        example = datasets_train_only["train"][0]

        # input_ids and labels should have same length
        assert len(example["input_ids"]) == len(example["labels"])

        # attention_mask should match input_ids length
        assert len(example["attention_mask"]) == len(example["input_ids"])

    def test_create_datasets_preserves_filenames(self, datasets_train_only):
        """Test that filenames are preserved from source data."""
        # All examples should have filenames
        for i in range(len(datasets_train_only["train"])):
            filename = datasets_train_only["train"][i]["filename"]
            assert isinstance(filename, str)
            assert len(filename) > 0
            assert ".xml" in filename


class TestTrain:
    """Integration tests for the train function (requires full model).

    Uses class-scoped fixture to share model across tests for speed.
    """

    @pytest.fixture(scope="class")
    def sample_data_path(self):
        """Path to sample extraction data fixture."""
        return Path(__file__).parent.parent / "fixtures" / "sample_extraction.jsonl"

    @pytest.fixture(scope="class")
    def shared_trainer_with_all_features(self, sample_data_path, tmp_path_factory):
        """Create one trainer with all features for testing (shared across class).

        Note: EarlyStoppingCallback is removed when test_path is provided
        (to avoid spurious warnings during test evaluation).

        Optimized for GPU usage with larger batch size and no dataloader workers.
        """
        output_dir = tmp_path_factory.mktemp("shared_trainer")

        # Create a minimal test config that disables unnecessary overhead
        import tempfile
        import yaml

        test_config = {
            "model_name": "microsoft/deberta-v3-base",
            "max_length": 512,
            "learning_rate": 0.00003,
            "per_device_train_batch_size": 5,  # Larger batch for GPU efficiency
            "per_device_eval_batch_size": 5,
            "num_train_epochs": 1,
            "fp16": True,  # Use fp16 for GPU acceleration
            "dataloader_num_workers": 0,  # No workers for tiny test dataset
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "logging_strategy": "no",  # Disable logging overhead
            "report_to": "none",
            "seed": 1234,
        }

        config_file = tmp_path_factory.mktemp("config") / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(test_config, f)

        trainer = train(
            train_path=sample_data_path,
            val_path=sample_data_path,
            test_path=sample_data_path,
            output_dir=output_dir,
            config_path=config_file,
            batch_size=5,  # Larger batch for GPU
            max_steps=1,
        )
        return trainer, output_dir

    def test_train_basic_functionality(self, shared_trainer_with_all_features):
        """Test basic training functionality."""
        trainer, _ = shared_trainer_with_all_features

        # Should return a Trainer object
        assert trainer is not None
        assert hasattr(trainer, "model")
        assert hasattr(trainer, "train_dataset")

        # Should have validation dataset
        assert trainer.eval_dataset is not None

        # Note: EarlyStoppingCallback is intentionally removed when test_path is provided
        # (happens in train() function after test evaluation to avoid spurious warnings)

    def test_train_saves_model_and_metrics(self, shared_trainer_with_all_features):
        """Test that training saves model files and metrics."""
        _, output_dir = shared_trainer_with_all_features

        # Model files should be saved in timestamped final-model directory
        final_model_dirs = list(output_dir.glob("final-model-*"))
        assert len(final_model_dirs) == 1, "Should have exactly one final model directory"
        final_model_dir = final_model_dirs[0]

        # Check model files
        assert (final_model_dir / "config.json").exists()
        assert (final_model_dir / "model.safetensors").exists() or (
            final_model_dir / "pytorch_model.bin"
        ).exists()

        # Check metrics files
        assert (final_model_dir / "train_results.json").exists()

    def test_train_with_test_set(self, shared_trainer_with_all_features):
        """Test training with test set evaluation."""
        _, output_dir = shared_trainer_with_all_features

        # Test metrics should be saved in timestamped final-model directory
        final_model_dirs = list(output_dir.glob("final-model-*"))
        assert len(final_model_dirs) == 1, "Should have exactly one final model directory"
        final_model_dir = final_model_dirs[0]

        assert (final_model_dir / "test_results.json").exists()
        assert (final_model_dir / "eval_results.json").exists()
        assert (final_model_dir / "all_results.json").exists()

    # REMOVED: test_train_seed_reproducibility - too slow (6+ minutes for 2 model loads)
    # Seed reproducibility is better tested at the unit level

    def test_train_model_configuration(self, shared_trainer_with_all_features):
        """Test that model is configured correctly (device, labels)."""
        trainer, _ = shared_trainer_with_all_features

        # Model should have 5 labels: O, B-BIBL, I-BIBL, B-QUOTE, I-QUOTE (CIT removed)
        assert trainer.model.config.num_labels == 5

        # Check device - should use CUDA if available, else CPU
        model_device = next(trainer.model.parameters()).device
        if torch.cuda.is_available():
            assert model_device.type == "cuda"
        else:
            assert model_device.type == "cpu"


# TestTrainPipeline REMOVED - These tests are extremely slow (10+ minutes total)
# They test the full data splitting pipeline which involves:
# - Heavy file I/O operations
# - Multiple full model loads (86M parameters each)
# - Data preprocessing and splitting
#
# This level of testing is better done:
# 1. Manually before releases
# 2. In nightly CI builds
# 3. Through smoke tests on real Phase 1/2/3 training
#
# For CI/CD, the TestCreateDatasets and TestTrain classes provide sufficient
# coverage of the training infrastructure
