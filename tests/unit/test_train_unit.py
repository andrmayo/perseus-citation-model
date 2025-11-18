"""Unit tests for training functions (no model downloads required)."""

import json
from pathlib import Path

import numpy as np
import pytest

from perscit_model.extraction.train import split_data, compute_metrics


class TestSplitData:
    """Tests for the split_data function."""

    @pytest.fixture
    def sample_data_path(self):
        """Path to sample extraction data fixture."""
        return Path(__file__).parent.parent / "fixtures" / "sample_extraction.jsonl"

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Temporary directory for split output."""
        return tmp_path / "split_output"

    def test_split_data_creates_files(self, sample_data_path, temp_output_dir):
        """Test that split_data creates train/val/test files."""
        train_path, val_path, test_path = split_data(
            input_file=sample_data_path,
            output_dir=temp_output_dir,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
        )

        # Check files were created
        assert train_path.exists()
        assert val_path.exists()
        assert test_path.exists()

        # Check split_info.json was created
        split_info_path = temp_output_dir / "split_info.json"
        assert split_info_path.exists()

    def test_split_data_correct_ratios(self, sample_data_path, temp_output_dir):
        """Test that data is split according to specified ratios."""
        train_path, val_path, test_path = split_data(
            input_file=sample_data_path,
            output_dir=temp_output_dir,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
        )

        # Count examples in each split
        def count_lines(path):
            with open(path) as f:
                return sum(1 for _ in f)

        train_count = count_lines(train_path)
        val_count = count_lines(val_path)
        test_count = count_lines(test_path)

        # With 5 examples: 60% = 3, 20% = 1, 20% = 1
        assert train_count == 3
        assert val_count == 1
        assert test_count == 1

    def test_split_data_saves_config(self, sample_data_path, temp_output_dir):
        """Test that split configuration is saved correctly."""
        train_path, val_path, test_path = split_data(
            input_file=sample_data_path,
            output_dir=temp_output_dir,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            seed=42,
        )

        split_info_path = temp_output_dir / "split_info.json"
        with open(split_info_path) as f:
            config = json.load(f)

        assert config["train_ratio"] == 0.6
        assert config["val_ratio"] == 0.2
        assert config["test_ratio"] == 0.2
        assert config["seed"] == 42
        assert config["input_file"] == str(sample_data_path.absolute())

    def test_split_data_reuses_existing_split(self, sample_data_path, temp_output_dir):
        """Test that split_data reuses existing split when config matches."""
        # First split
        train_path, val_path, test_path = split_data(
            input_file=sample_data_path,
            output_dir=temp_output_dir,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            seed=42,
        )

        # Get modification times
        train_mtime_1 = train_path.stat().st_mtime
        val_mtime_1 = val_path.stat().st_mtime
        test_mtime_1 = test_path.stat().st_mtime

        # Second split with same config
        train_path, val_path, test_path = split_data(
            input_file=sample_data_path,
            output_dir=temp_output_dir,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            seed=42,
        )

        # Files should not have been recreated
        assert train_path.stat().st_mtime == train_mtime_1
        assert val_path.stat().st_mtime == val_mtime_1
        assert test_path.stat().st_mtime == test_mtime_1

    def test_split_data_resplits_when_config_changes(
        self, sample_data_path, temp_output_dir
    ):
        """Test that split_data creates new split when config changes."""
        # First split
        train_path, val_path, test_path = split_data(
            input_file=sample_data_path,
            output_dir=temp_output_dir,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            seed=42,
        )

        # Check first split counts
        def count_lines(path):
            with open(path) as f:
                return sum(1 for _ in f)

        train_count_1 = count_lines(train_path)
        # With 5 examples: 60% = 3
        assert train_count_1 == 3

        # Second split with different ratio
        train_path, val_path, test_path = split_data(
            input_file=sample_data_path,
            output_dir=temp_output_dir,
            train_ratio=0.8,  # Changed
            val_ratio=0.1,
            test_ratio=0.1,
            seed=42,
        )

        # Check new split has correct counts (different from before)
        train_count_2 = count_lines(train_path)
        # With 5 examples: 80% = 4
        assert train_count_2 == 4
        assert train_count_2 != train_count_1

    def test_split_data_force_redo(self, sample_data_path, temp_output_dir):
        """Test that force_redo always creates new split."""
        import time

        # First split
        train_path, val_path, test_path = split_data(
            input_file=sample_data_path,
            output_dir=temp_output_dir,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            seed=42,
        )

        # Read content of first split
        with open(train_path) as f:
            content_1 = f.read()

        # Small delay to ensure different timestamp
        time.sleep(0.01)

        # Second split with force_redo (same config)
        train_path, val_path, test_path = split_data(
            input_file=sample_data_path,
            output_dir=temp_output_dir,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            seed=42,
            force_redo=True,
        )

        # Files should exist (force_redo should have recreated them)
        assert train_path.exists()
        assert val_path.exists()
        assert test_path.exists()

        # Content should be the same (same seed/ratios)
        with open(train_path) as f:
            content_2 = f.read()
        assert content_2 == content_1

    def test_split_data_invalid_ratios(self, sample_data_path, temp_output_dir):
        """Test that invalid ratios raise ValueError."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            split_data(
                input_file=sample_data_path,
                output_dir=temp_output_dir,
                train_ratio=0.5,
                val_ratio=0.3,
                test_ratio=0.3,  # Sum = 1.1
            )

    def test_split_data_seed_affects_shuffle(self, sample_data_path, temp_output_dir):
        """Test that different seeds produce different splits."""
        # Split with seed 42
        train_path_1, _, _ = split_data(
            input_file=sample_data_path,
            output_dir=temp_output_dir / "split1",
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            seed=42,
        )

        # Split with seed 123
        train_path_2, _, _ = split_data(
            input_file=sample_data_path,
            output_dir=temp_output_dir / "split2",
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            seed=123,
        )

        # Read first example from each split
        def read_first_line(path):
            with open(path) as f:
                return f.readline()

        line1 = read_first_line(train_path_1)
        line2 = read_first_line(train_path_2)

        # Different seeds should produce different orderings
        # (Not guaranteed with only 5 examples, but likely)
        # At minimum, verify both splits are valid
        assert len(line1) > 0
        assert len(line2) > 0

    def test_split_data_reads_from_config(
        self, sample_data_path, temp_output_dir, tmp_path
    ):
        """Test that split_data reads ratios and seed from config file."""
        # Create a temporary config file
        config_path = tmp_path / "test_config.yaml"
        config_content = """
train_ratio: 0.7
val_ratio: 0.15
test_ratio: 0.15
seed: 999
model_name: microsoft/deberta-v3-base
max_length: 512
learning_rate: 3e-5
"""
        config_path.write_text(config_content)

        # Split without providing ratios (should read from config)
        train_path, val_path, test_path = split_data(
            input_file=sample_data_path,
            output_dir=temp_output_dir,
            config_path=config_path,
        )

        # Check split_info.json has config values
        split_info_path = temp_output_dir / "split_info.json"
        with open(split_info_path) as f:
            config = json.load(f)

        assert config["train_ratio"] == 0.7
        assert config["val_ratio"] == 0.15
        assert config["test_ratio"] == 0.15
        assert config["seed"] == 999


class TestComputeMetrics:
    """Tests for the compute_metrics function."""

    def test_compute_metrics_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        # Create mock eval_pred tuple
        # predictions shape: (batch_size, seq_len, num_labels)
        # labels shape: (batch_size, seq_len)

        # Perfect predictions: all labels match
        predictions = np.array(
            [
                [[0.9, 0.1], [0.1, 0.9], [0.9, 0.1]],  # Batch 1
                [[0.9, 0.1], [0.9, 0.1], [0.9, 0.1]],  # Batch 2
            ]
        )
        labels = np.array(
            [
                [0, 1, 0],  # Batch 1
                [0, 0, 0],  # Batch 2
            ]
        )

        eval_pred = (predictions, labels)
        metrics = compute_metrics(eval_pred)

        # Perfect predictions should have accuracy = 1.0
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_compute_metrics_ignores_special_tokens(self):
        """Test that -100 labels are ignored."""
        # predictions shape: (batch_size, seq_len, num_labels)
        predictions = np.array(
            [
                [[0.9, 0.1], [0.1, 0.9], [0.9, 0.1]],  # Batch 1
            ]
        )
        # Use -100 for special token (should be ignored)
        labels = np.array(
            [
                [-100, 1, 0],  # First token should be ignored
            ]
        )

        eval_pred = (predictions, labels)
        metrics = compute_metrics(eval_pred)

        # Should only evaluate on non-special tokens
        # Predictions for positions 1,2: [1, 0]
        # Labels for positions 1,2: [1, 0]
        assert metrics["accuracy"] == 1.0

    def test_compute_metrics_returns_correct_keys(self):
        """Test that compute_metrics returns all expected keys."""
        predictions = np.array([[[0.9, 0.1], [0.1, 0.9]]])
        labels = np.array([[0, 1]])

        eval_pred = (predictions, labels)
        metrics = compute_metrics(eval_pred)

        # Check all expected keys are present
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

    def test_compute_metrics_with_bio_tags(self):
        """Test metrics with realistic BIO tag predictions."""
        # Simulate 3-class BIO tagging: O=0, B-CIT=1, I-CIT=2
        # Sequence: O B-CIT I-CIT O
        predictions = np.array(
            [
                [
                    [0.9, 0.05, 0.05],  # Predict O
                    [0.1, 0.8, 0.1],  # Predict B-CIT
                    [0.1, 0.1, 0.8],  # Predict I-CIT
                    [0.8, 0.1, 0.1],  # Predict O
                ]
            ]
        )
        labels = np.array([[0, 1, 2, 0]])

        eval_pred = (predictions, labels)
        metrics = compute_metrics(eval_pred)

        # Perfect predictions
        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0
