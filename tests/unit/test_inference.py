"""Unit tests for inference module."""

import pytest
import torch
import transformers
from unittest.mock import Mock, MagicMock, patch, PropertyMock

from perscit_model.extraction.inference import InferenceModel


@pytest.fixture
def mock_model():
    """Create a mock model."""
    model = Mock(spec=transformers.AutoModelForTokenClassification)
    model.config = Mock()
    model.config.max_position_embeddings = 512
    model.to = Mock(return_value=model)
    model.eval = Mock()
    return model


@pytest.fixture
def mock_loader():
    """Create a mock data loader."""
    loader = Mock()
    tokenizer = Mock()
    tokenizer.cls_token_id = 101
    tokenizer.sep_token_id = 102
    tokenizer.pad_token_id = 0
    loader.tokenizer = tokenizer
    return loader


@pytest.fixture
def mock_inference_model(mock_model, mock_loader):
    """Create an InferenceModel with mocked dependencies."""
    with patch('perscit_model.extraction.inference.InferenceModel.load_model', return_value=mock_model):
        with patch('perscit_model.extraction.inference.ExtractionDataLoader', return_value=mock_loader):
            with patch('torch.cuda.is_available', return_value=False):
                model = InferenceModel()
    return model


class TestInferenceModelInit:
    """Tests for InferenceModel initialization."""

    def test_init_sets_device_to_cuda_when_available(self, mock_model, mock_loader):
        """Test that model is moved to CUDA when available."""
        with patch('perscit_model.extraction.inference.InferenceModel.load_model', return_value=mock_model):
            with patch('perscit_model.extraction.inference.ExtractionDataLoader', return_value=mock_loader):
                with patch('torch.cuda.is_available', return_value=True):
                    model = InferenceModel()

        assert model.device == "cuda"
        mock_model.to.assert_called_once_with("cuda")
        mock_model.eval.assert_called_once()

    def test_init_sets_device_to_cpu_when_cuda_unavailable(self, mock_model, mock_loader):
        """Test that model stays on CPU when CUDA unavailable."""
        with patch('perscit_model.extraction.inference.InferenceModel.load_model', return_value=mock_model):
            with patch('perscit_model.extraction.inference.ExtractionDataLoader', return_value=mock_loader):
                with patch('torch.cuda.is_available', return_value=False):
                    model = InferenceModel()

        assert model.device == "cpu"
        mock_model.to.assert_called_once_with("cpu")

    def test_init_respects_explicit_device(self, mock_model, mock_loader):
        """Test that explicit device parameter is respected."""
        with patch('perscit_model.extraction.inference.InferenceModel.load_model', return_value=mock_model):
            with patch('perscit_model.extraction.inference.ExtractionDataLoader', return_value=mock_loader):
                model = InferenceModel(device="cuda:1")

        assert model.device == "cuda:1"
        mock_model.to.assert_called_once_with("cuda:1")


class TestProcessBatch:
    """Tests for process_batch method."""

    def test_process_batch_handles_single_text(self, mock_inference_model):
        """Test processing a single text."""
        texts = ["This is a test."]

        # Mock tokenizer output
        mock_inference_model.loader.tokenizer.return_value = {
            "input_ids": torch.tensor([[101, 2023, 2003, 1037, 3231, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]]),
            "offset_mapping": torch.tensor([[(0, 0), (0, 4), (5, 7), (8, 9), (10, 14), (14, 15)]]),
        }

        # Mock model output
        mock_logits = torch.tensor([[[0.1, 0.9, 0.0]] * 6])
        mock_output = Mock()
        mock_output.logits = mock_logits
        mock_inference_model.model.return_value = mock_output

        results = mock_inference_model.process_batch(texts)

        assert len(results) == 1
        encoding, labels = results[0]
        assert isinstance(encoding, transformers.BatchEncoding)
        assert isinstance(labels, list)
        assert len(labels) == 6

    def test_process_batch_handles_multiple_texts(self, mock_inference_model):
        """Test processing multiple texts."""
        texts = ["First text.", "Second text."]

        mock_inference_model.loader.tokenizer.return_value = {
            "input_ids": torch.tensor([[101, 2023, 102, 0], [101, 2019, 3231, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]),
            "offset_mapping": torch.tensor([
                [(0, 0), (0, 5), (5, 6), (0, 0)],
                [(0, 0), (0, 6), (7, 11), (11, 12)]
            ]),
        }

        mock_logits = torch.tensor([
            [[0.1, 0.9, 0.0]] * 4,
            [[0.1, 0.9, 0.0]] * 4,
        ])
        mock_output = Mock()
        mock_output.logits = mock_logits
        mock_inference_model.model.return_value = mock_output

        results = mock_inference_model.process_batch(texts)

        assert len(results) == 2

    def test_process_batch_raises_on_too_long_text(self, mock_inference_model):
        """Test that process_batch raises error when text is too long."""
        texts = ["x" * 10000]  # Very long text

        # Mock tokenizer to return sequence longer than max_length
        mock_inference_model.loader.tokenizer.return_value = {
            "input_ids": torch.tensor([[1] * 600]),  # 600 > 512
            "attention_mask": torch.tensor([[1] * 600]),
            "offset_mapping": torch.tensor([[(0, 1)] * 600]),
        }

        with pytest.raises(ValueError, match="Input text too long"):
            mock_inference_model.process_batch(texts)

    def test_process_batch_moves_inputs_to_device(self, mock_inference_model):
        """Test that inputs are moved to the correct device."""
        texts = ["Test text."]
        mock_inference_model.device = "cpu"  # Use CPU to avoid CUDA initialization

        # Create real tensors so they can be subscripted
        input_ids = torch.tensor([[101, 2023, 3793, 102, 0]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 0]])

        mock_inference_model.loader.tokenizer.return_value = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": torch.tensor([[(0, 0), (0, 4), (5, 9), (9, 10), (0, 0)]]),
        }

        mock_logits = torch.tensor([[[0.1, 0.9, 0.0]] * 5])
        mock_output = Mock()
        mock_output.logits = mock_logits

        # Track calls to the model while preserving config
        calls_to_model = []
        original_config = mock_inference_model.model.config
        def mock_model_call(**kwargs):
            calls_to_model.append(kwargs)
            return mock_output

        # Add config attribute to the function
        mock_model_call.config = original_config
        mock_inference_model.model = mock_model_call

        mock_inference_model.process_batch(texts)

        # Verify model was called with inputs
        assert len(calls_to_model) == 1
        assert "input_ids" in calls_to_model[0]
        assert "attention_mask" in calls_to_model[0]


class TestProcessText:
    """Tests for process_text method."""

    def test_process_text_returns_encoding_and_labels(self, mock_inference_model):
        """Test that process_text returns proper encoding and labels."""
        text = "Test citation text."

        mock_encoding = {
            "input_ids": torch.tensor([[101, 2023, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "offset_mapping": torch.tensor([[(0, 0), (0, 4), (4, 5)]]),
        }
        mock_inference_model.loader.tokenize_text.return_value = mock_encoding

        mock_logits = torch.tensor([[[0.1, 0.9, 0.0], [0.8, 0.1, 0.1], [0.1, 0.9, 0.0]]])
        mock_output = Mock()
        mock_output.logits = mock_logits
        mock_inference_model.model.return_value = mock_output

        encoding, labels = mock_inference_model.process_text(text)

        assert isinstance(labels, list)
        assert len(labels) == 3
        mock_inference_model.loader.tokenize_text.assert_called_once()


class TestInsertTagsIntoXml:
    """Tests for insert_tags_into_xml method."""

    def test_insert_tags_single_text(self, mock_inference_model):
        """Test inserting tags into a single text."""
        text = "This is a citation."
        encoding = {
            "input_ids": torch.tensor([[101, 2023, 2003, 1037, 102]]),
            "offset_mapping": torch.tensor([[(0, 0), (0, 4), (5, 7), (8, 9), (9, 10)]]),
        }
        labels = ["O", "B-BIBL", "I-BIBL", "I-BIBL", "O"]

        # Mock _insert_tags to return tagged text
        with patch.object(mock_inference_model, '_insert_tags', return_value="<bibl>This is a</bibl> citation."):
            result = mock_inference_model.insert_tags_into_xml(text, encoding, labels)

        assert isinstance(result, str)
        assert "<bibl>" in result
        assert "</bibl>" in result

    def test_insert_tags_raises_without_offset_mapping(self, mock_inference_model):
        """Test that insert_tags raises error without offset_mapping."""
        text = "Test text."
        encoding = {"input_ids": torch.tensor([[101, 2023, 102]])}
        labels = ["O", "O", "O"]

        with pytest.raises(KeyError, match="offset_mappping"):
            mock_inference_model.insert_tags_into_xml(text, encoding, labels)


class TestInsertTags:
    """Tests for _insert_tags method."""

    def test_insert_tags_bibl(self, mock_inference_model):
        """Test inserting BIBL tags."""
        xml = "This is a citation."
        tokens = torch.tensor([101, 2023, 2003, 1037, 5992, 102])
        offset_mapping = torch.tensor([(0, 0), (0, 4), (5, 7), (8, 9), (10, 18), (18, 19)])
        labels = ["O", "O", "O", "B-BIBL", "I-BIBL", "O"]

        result = mock_inference_model._insert_tags(xml, tokens, offset_mapping, labels)

        assert "<bibl>a citation</bibl>" in result

    def test_insert_tags_quote(self, mock_inference_model):
        """Test inserting QUOTE tags."""
        xml = "He said hello there."
        tokens = torch.tensor([101, 2002, 2056, 7592, 2045, 102])
        offset_mapping = torch.tensor([(0, 0), (0, 2), (3, 7), (8, 13), (14, 19), (19, 20)])
        labels = ["O", "O", "O", "B-QUOTE", "I-QUOTE", "O"]

        result = mock_inference_model._insert_tags(xml, tokens, offset_mapping, labels)

        assert "<quote>hello there</quote>" in result

    def test_insert_tags_multiple_citations(self, mock_inference_model):
        """Test inserting multiple citation tags."""
        xml = "First citation and second citation."
        tokens = torch.tensor([101, 2034, 5992, 1998, 2117, 5992, 102])
        offset_mapping = torch.tensor([
            (0, 0), (0, 5), (6, 14), (15, 18), (19, 25), (26, 34), (34, 35)
        ])
        labels = ["O", "B-BIBL", "I-BIBL", "O", "B-QUOTE", "I-QUOTE", "O"]

        result = mock_inference_model._insert_tags(xml, tokens, offset_mapping, labels)

        assert "<bibl>First citation</bibl>" in result
        assert "<quote>second citation</quote>" in result

    def test_insert_tags_no_citations(self, mock_inference_model):
        """Test text with no citations remains unchanged."""
        xml = "Just plain text."
        tokens = torch.tensor([101, 2074, 5810, 3793, 102])
        offset_mapping = torch.tensor([(0, 0), (0, 4), (5, 10), (11, 15), (15, 16)])
        labels = ["O", "O", "O", "O", "O"]

        result = mock_inference_model._insert_tags(xml, tokens, offset_mapping, labels)

        assert result == xml

    def test_insert_tags_filters_special_tokens(self, mock_inference_model):
        """Test that special tokens (CLS, SEP, PAD) are filtered out."""
        xml = "Test text."
        # Include special tokens: CLS=101, SEP=102, PAD=0
        tokens = torch.tensor([101, 2023, 3793, 102, 0, 0])
        offset_mapping = torch.tensor([(0, 0), (0, 4), (5, 9), (9, 10), (0, 0), (0, 0)])
        labels = ["O", "B-BIBL", "I-BIBL", "O", "O", "O"]

        result = mock_inference_model._insert_tags(xml, tokens, offset_mapping, labels)

        # Should only tag actual text, not special tokens
        assert "<bibl>Test text</bibl>" in result


class TestLoadModel:
    """Tests for load_model static method."""

    def test_load_model_with_last_trained_true(self):
        """Test loading the most recent trained model."""
        mock_path = Mock()
        mock_path.glob.return_value = [
            Mock(stat=Mock(return_value=Mock(st_mtime=100))),
            Mock(stat=Mock(return_value=Mock(st_mtime=200))),  # Most recent
        ]

        with patch('perscit_model.extraction.inference.MODEL_TRAIN_DIR', mock_path):
            with patch('perscit_model.extraction.inference.load_model_from_checkpoint') as mock_load:
                InferenceModel.load_model(last_trained=True)

        # Should load the most recent model
        mock_load.assert_called_once()

    def test_load_model_raises_when_no_models_found(self):
        """Test that load_model raises error when no models exist."""
        mock_path = Mock()
        mock_path.glob.return_value = []

        with patch('perscit_model.extraction.inference.MODEL_TRAIN_DIR', mock_path):
            with pytest.raises(FileNotFoundError, match="No final models"):
                InferenceModel.load_model(last_trained=True)
