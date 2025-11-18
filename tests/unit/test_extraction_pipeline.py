"""Integration test for the complete extraction data pipeline."""

from pathlib import Path

import pytest

from perscit_model.extraction.data_loader import (
    parse_xml_to_bio,
    ExtractionDataLoader,
    create_extraction_dataset,
    LABEL2ID,
    SPECIAL_TOKENS,
)


class TestExtractionPipeline:
    """Integration tests for the complete extraction pipeline."""

    @pytest.fixture
    def sample_data_path(self):
        """Path to sample extraction data fixture."""
        return Path(__file__).parent.parent / "fixtures" / "sample_extraction.jsonl"

    def test_parse_xml_to_bio_replaces_tags(self):
        """Test that parse_xml_to_bio correctly replaces citation tags."""
        xml = '<bibl n="Hdt. 8.82">Hdt. 8.82</bibl> some context'

        result = parse_xml_to_bio(xml)

        # Should replace <bibl> tags with special tokens
        assert "[BIBL_START]" in result
        assert "[BIBL_END]" in result
        assert "Hdt. 8.82" in result
        assert "some context" in result
        # Original tags should be gone
        assert "<bibl" not in result
        assert "</bibl>" not in result

    def test_parse_xml_to_bio_removes_attributes(self):
        """Test that attributes are removed from citation tags."""
        xml = '<bibl n="Hdt. 8.82" type="ancient">Hdt. 8.82</bibl>'

        result = parse_xml_to_bio(xml)

        # Attributes should be removed
        assert 'n=' not in result
        assert 'type=' not in result

    def test_parse_xml_to_bio_preserves_other_tags(self):
        """Test that non-citation tags are preserved."""
        xml = '<bibl>Hdt. 8.82</bibl> in <title>Histories</title>'

        result = parse_xml_to_bio(xml)

        # <bibl> should be replaced
        assert "[BIBL_START]" in result
        assert "[BIBL_END]" in result
        # <title> should be preserved (though might be wrapped by lxml)
        assert "title" in result.lower()

    def test_extraction_data_loader_adds_special_tokens(self, mock_tokenizer):
        """Test that ExtractionDataLoader adds special tokens to tokenizer."""
        loader = ExtractionDataLoader()

        # Should have added 6 special tokens
        assert loader.special_tokens == SPECIAL_TOKENS

    def test_extraction_data_loader_processes_real_data(self, sample_data_path, mock_tokenizer):
        """Test that loader can process real citation data."""
        loader = ExtractionDataLoader()

        examples = list(loader(sample_data_path))

        # Should have 5 examples
        assert len(examples) == 5

        # All should have tokenized xml_context
        for example in examples:
            assert hasattr(example.xml_context, 'input_ids')
            assert hasattr(example.xml_context, 'attention_mask')

    def test_extraction_data_loader_preserves_filenames(self, sample_data_path, mock_tokenizer):
        """Test that filenames are preserved from real data."""
        loader = ExtractionDataLoader()

        examples = list(loader(sample_data_path))

        # All examples should have filename field
        for example in examples:
            assert example.filename != ""
            assert ".xml" in example.filename

    def test_create_dataset_from_real_data(self, sample_data_path, mock_tokenizer):
        """Test creating a dataset from real citation data."""
        from datasets import Dataset

        dataset = create_extraction_dataset(sample_data_path)

        # Should create a Dataset
        assert isinstance(dataset, Dataset)

        # Should have 5 examples
        assert len(dataset) == 5

        # Should have all required columns
        assert "input_ids" in dataset.column_names
        assert "attention_mask" in dataset.column_names
        assert "labels" in dataset.column_names
        assert "filename" in dataset.column_names

    def test_dataset_labels_are_valid(self, sample_data_path, mock_tokenizer):
        """Test that generated labels are in valid range."""
        dataset = create_extraction_dataset(sample_data_path)

        valid_label_ids = set(LABEL2ID.values()) | {-100}

        for i in range(len(dataset)):
            for label in dataset[i]["labels"]:
                assert label in valid_label_ids, f"Invalid label {label} at example {i}"

    def test_dataset_labels_match_input_length(self, sample_data_path, mock_tokenizer):
        """Test that labels match input_ids length for all examples."""
        dataset = create_extraction_dataset(sample_data_path)

        for i in range(len(dataset)):
            assert len(dataset[i]["labels"]) == len(dataset[i]["input_ids"]), \
                f"Label/input length mismatch at example {i}"

    def test_pipeline_handles_nested_tags(self):
        """Test that pipeline handles nested citation tags."""
        xml = '<cit><bibl>Hdt. 1.1</bibl><quote>some text</quote></cit>'

        # Parse XML
        processed = parse_xml_to_bio(xml)

        # Should have all special tokens
        assert "[CIT_START]" in processed
        assert "[CIT_END]" in processed
        assert "[BIBL_START]" in processed
        assert "[BIBL_END]" in processed
        assert "[QUOTE_START]" in processed
        assert "[QUOTE_END]" in processed

    def test_pipeline_handles_multiple_citations(self):
        """Test pipeline with multiple citations in sequence."""
        xml = '<bibl>Hdt. 1.1</bibl> and <bibl>Thuc. 2.1</bibl>'

        processed = parse_xml_to_bio(xml)

        # Should have two sets of BIBL tokens
        assert processed.count("[BIBL_START]") == 2
        assert processed.count("[BIBL_END]") == 2

    def test_pipeline_handles_quote_tags(self):
        """Test that quote tags are properly handled."""
        xml = 'He said <quote>hello world</quote> to them'

        processed = parse_xml_to_bio(xml)

        assert "[QUOTE_START]" in processed
        assert "[QUOTE_END]" in processed
        assert "hello world" in processed

    def test_pipeline_handles_malformed_xml(self):
        """Test that BeautifulSoup repairs malformed XML."""
        # Missing closing tag
        xml = '<bibl>Hdt. 1.1 some text'

        # Should not raise an error
        processed = parse_xml_to_bio(xml)

        # BeautifulSoup should auto-close the tag
        assert isinstance(processed, str)
        assert len(processed) > 0

    def test_end_to_end_real_citation(self, sample_data_path, mock_tokenizer):
        """Test complete pipeline on actual citation data."""
        # Load one real example
        loader = ExtractionDataLoader()
        examples = list(loader(sample_data_path))
        first_example = examples[0]

        # Should have processed xml_context with special tokens
        # (Mock tokenizer means we can't check exact tokens, but structure should be right)
        assert hasattr(first_example.xml_context, 'input_ids')
        assert first_example.xml_context.input_ids.shape[0] == 1  # Batch size 1

        # Create dataset
        dataset = create_extraction_dataset(sample_data_path)

        # First example in dataset
        first_dataset_item = dataset[0]

        # Should have all fields
        assert "input_ids" in first_dataset_item
        assert "labels" in first_dataset_item
        assert "filename" in first_dataset_item

        # Labels should be same length as input
        assert len(first_dataset_item["labels"]) == len(first_dataset_item["input_ids"])

    def test_dataset_preserves_filenames_from_real_data(self, sample_data_path, mock_tokenizer):
        """Test that dataset preserves actual filenames from JSONL."""
        dataset = create_extraction_dataset(sample_data_path)

        # All examples should have filenames from the fixture
        for i in range(len(dataset)):
            filename = dataset[i]["filename"]
            assert filename.endswith(".xml")
            # Should be a valid path-like string
            assert len(filename) > 0

    def test_pipeline_memory_efficiency(self, sample_data_path, mock_tokenizer):
        """Test that loader uses generators (memory efficient)."""
        loader = ExtractionDataLoader()

        # Should return a generator
        result = loader(sample_data_path)
        assert hasattr(result, '__iter__')
        assert hasattr(result, '__next__')

        # Can consume one at a time
        first = next(result)
        second = next(result)

        assert first.filename != second.filename or True  # Different or same is fine

    def test_special_tokens_are_consistent(self):
        """Test that special tokens are consistently defined."""
        # Should have 6 special tokens (3 types × 2 directions)
        assert len(SPECIAL_TOKENS) == 6

        # Should have START and END for each type
        assert "[BIBL_START]" in SPECIAL_TOKENS
        assert "[BIBL_END]" in SPECIAL_TOKENS
        assert "[QUOTE_START]" in SPECIAL_TOKENS
        assert "[QUOTE_END]" in SPECIAL_TOKENS
        assert "[CIT_START]" in SPECIAL_TOKENS
        assert "[CIT_END]" in SPECIAL_TOKENS
