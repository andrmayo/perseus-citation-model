"""Unit tests for ExtractionDataLoader."""

import json
import tempfile
from pathlib import Path

import pytest

from perscit_model.extraction.data_loader import ExtractionDataLoader


class TestExtractionDataLoader:
    """Test suite for ExtractionDataLoader."""

    @pytest.fixture
    def temp_jsonl_file(self):
        """Create a temporary JSONL file with extraction task data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            test_data = [
                {
                    "xml_context": "<bibl>Hdt. 8.82</bibl> some context",
                    "filename": "file1.xml",
                    "bibl": "Hdt. 8.82",
                    "urn": "urn:cts:greekLit:tlg0016.tlg001:8.82"
                },
                {
                    "xml_context": "<quote>τᾶς πολυχρύσου</quote> <bibl>Soph. OT 151</bibl>",
                    "filename": "file2.xml",
                    "bibl": "Soph. OT 151",
                    "urn": "urn:cts:greekLit:tlg0011.tlg004:151"
                },
                {
                    "xml_context": "plain text <cit>citation</cit> more text",
                    "filename": "file3.xml",
                    "bibl": "Eur. Med. 1",
                    "urn": "urn:cts:greekLit:tlg0006.tlg012:1"
                },
            ]
            for item in test_data:
                f.write(json.dumps(item) + '\n')
            temp_path = f.name

        yield temp_path

        # Cleanup
        Path(temp_path).unlink()

    def test_initialization(self, mock_tokenizer):
        """Test that ExtractionDataLoader initializes correctly."""
        loader = ExtractionDataLoader()

        assert loader.model_name == "microsoft/deberta-v3-base"
        assert loader.max_length == 512
        assert loader._tokenizer is None

    def test_inherits_from_shared_loader(self, mock_tokenizer):
        """Test that ExtractionDataLoader inherits SharedDataLoader methods."""
        loader = ExtractionDataLoader()

        # Should have inherited methods
        assert hasattr(loader, 'load_jsonl')
        assert hasattr(loader, 'tokenize_text')
        assert hasattr(loader, 'tokenizer')

    def test_call_returns_generator(self, temp_jsonl_file, mock_tokenizer):
        """Test that __call__ returns a generator."""
        loader = ExtractionDataLoader()
        result = loader(temp_jsonl_file)

        # Check it's a generator
        assert hasattr(result, '__iter__')
        assert hasattr(result, '__next__')

    def test_call_yields_extraction_data(self, temp_jsonl_file, mock_tokenizer):
        """Test that __call__ yields dicts with extraction data."""
        loader = ExtractionDataLoader()
        data = list(loader(temp_jsonl_file))

        # Should yield 3 items
        assert len(data) == 3

        # All should be dicts
        for item in data:
            assert isinstance(item, dict)
            assert "xml_context" in item
            assert "filename" in item

    def test_extraction_data_has_correct_fields(self, temp_jsonl_file, mock_tokenizer):
        """Test that dicts have xml_context and filename fields."""
        loader = ExtractionDataLoader()
        data = list(loader(temp_jsonl_file))

        first_item = data[0]

        # Should have xml_context as raw string (not tokenized)
        assert isinstance(first_item["xml_context"], str)
        assert "bibl" in first_item["xml_context"].lower()

        # Should have filename as string
        assert isinstance(first_item["filename"], str)
        assert first_item["filename"] == "file1.xml"

    def test_only_tokenizes_xml_context(self, temp_jsonl_file, mock_tokenizer):
        """Test that xml_context is raw text, not tokenized."""
        loader = ExtractionDataLoader()
        data = list(loader(temp_jsonl_file))

        first_item = data[0]

        # xml_context should be raw string (not tokenized in loader anymore)
        assert isinstance(first_item["xml_context"], str)

        # filename should be string
        assert isinstance(first_item["filename"], str)

    def test_handles_missing_filename(self, mock_tokenizer):
        """Test that loader handles missing filename field."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            # Data without filename
            test_data = {"xml_context": "test context", "bibl": "test"}
            f.write(json.dumps(test_data) + '\n')
            temp_path = f.name

        try:
            loader = ExtractionDataLoader()
            data = list(loader(temp_path))

            # Should default to empty string
            assert data[0]["filename"] == ""
        finally:
            Path(temp_path).unlink()

    def test_tokenized_context_correct_length(self, temp_jsonl_file, mock_tokenizer):
        """Test that loader returns raw strings (tokenization happens later)."""
        loader = ExtractionDataLoader(max_length=128)
        data = list(loader(temp_jsonl_file))

        # Should return raw strings, not tokenized
        for item in data:
            assert isinstance(item["xml_context"], str)

    def test_preserves_order(self, temp_jsonl_file, mock_tokenizer):
        """Test that data is yielded in file order."""
        loader = ExtractionDataLoader()
        data = list(loader(temp_jsonl_file))

        assert data[0]["filename"] == "file1.xml"
        assert data[1]["filename"] == "file2.xml"
        assert data[2]["filename"] == "file3.xml"

    def test_memory_efficient_streaming(self, temp_jsonl_file, mock_tokenizer):
        """Test that loader processes one item at a time."""
        loader = ExtractionDataLoader()
        gen = loader(temp_jsonl_file)

        # Get first item without exhausting generator
        first_item = next(gen)
        assert isinstance(first_item, dict)
        assert first_item["filename"] == "file1.xml"

        # Should still have more items
        second_item = next(gen)
        assert second_item["filename"] == "file2.xml"

    def test_handles_greek_text(self, temp_jsonl_file, mock_tokenizer):
        """Test that loader handles Greek text correctly."""
        loader = ExtractionDataLoader()
        data = list(loader(temp_jsonl_file))

        # Second item has Greek text
        greek_item = data[1]
        assert isinstance(greek_item["xml_context"], str)
        # Should contain Greek characters
        assert "τᾶς" in greek_item["xml_context"] or "quote" in greek_item["xml_context"]

    def test_custom_config_path(self, mock_tokenizer):
        """Test loader with custom config path."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
model_name: bert-base-uncased
max_length: 256
""")
            config_path = f.name

        try:
            loader = ExtractionDataLoader(config_path=config_path)
            assert loader.model_name == "bert-base-uncased"
            assert loader.max_length == 256
        finally:
            Path(config_path).unlink()


class TestParseXmlToBio:
    """Test suite for parse_xml_to_bio method."""

    def test_default_converts_all_tags(self):
        """Test default behavior (tag_retention_prob=0.0) converts all tags."""
        xml = '<bibl n="1">Hdt. 8.82</bibl> some text <quote type="verse">content</quote>'
        result = ExtractionDataLoader.parse_xml_to_bio(xml, tag_retention_prob=0.0)

        # All tags should be converted to special tokens
        assert "[BIBL_START]" in result
        assert "[BIBL_END]" in result
        assert "[QUOTE_START]" in result
        assert "[QUOTE_END]" in result

        # No original tags with attributes should remain
        assert '<bibl' not in result
        assert '<quote' not in result
        assert 'n="1"' not in result
        assert 'type="verse"' not in result

        # Content should be preserved
        assert "Hdt. 8.82" in result
        assert "content" in result
        assert "some text" in result

    def test_full_retention_keeps_all_tags(self):
        """Test full retention (tag_retention_prob=1.0) keeps all tags with attributes."""
        xml = '<bibl n="1">Hdt. 8.82</bibl> text <quote type="verse">content</quote>'
        result = ExtractionDataLoader.parse_xml_to_bio(xml, tag_retention_prob=1.0)

        # Original tags with attributes should be preserved
        assert '<bibl n="1">' in result
        assert '</bibl>' in result
        assert '<quote type="verse">' in result
        assert '</quote>' in result

        # No special tokens should be present
        assert "[BIBL_START]" not in result
        assert "[BIBL_END]" not in result
        assert "[QUOTE_START]" not in result
        assert "[QUOTE_END]" not in result

        # Content should be preserved
        assert "Hdt. 8.82" in result
        assert "content" in result

    def test_partial_retention_deterministic(self):
        """Test that paired tags get same decision (both kept or both converted)."""
        # Use a fixed seed for deterministic testing
        import random
        random.seed(42)

        xml = '<bibl n="1">text1</bibl> <quote>text2</quote> <cit>text3</cit>'
        result = ExtractionDataLoader.parse_xml_to_bio(xml, tag_retention_prob=0.5)

        # Check that opening and closing tags are consistent
        # If <bibl n="1"> is present, </bibl> should also be present
        has_bibl_opening = '<bibl n="1">' in result
        has_bibl_closing = '</bibl>' in result
        has_bibl_start = '[BIBL_START]' in result
        has_bibl_end = '[BIBL_END]' in result

        # Either both original tags or both special tokens
        assert has_bibl_opening == has_bibl_closing
        assert has_bibl_start == has_bibl_end

        # Exactly one of the two options
        assert (has_bibl_opening and not has_bibl_start) or (not has_bibl_opening and has_bibl_start)

    def test_preserves_content_between_tags(self):
        """Test that all content between tags is preserved."""
        xml = 'before <bibl>inside1</bibl> middle <quote>inside2</quote> after'
        result = ExtractionDataLoader.parse_xml_to_bio(xml, tag_retention_prob=0.0)

        assert "before" in result
        assert "inside1" in result
        assert "middle" in result
        assert "inside2" in result
        assert "after" in result

    def test_handles_orphaned_opening_tag(self):
        """Test handling of orphaned opening tag (from excerpting)."""
        # Simulates text excerpted in middle of citation
        xml = 'some text <bibl n="ref">content here...'

        # With retention=0.0, should convert orphaned tag
        result = ExtractionDataLoader.parse_xml_to_bio(xml, tag_retention_prob=0.0)
        assert "[BIBL_START]" in result
        assert '<bibl' not in result

        # With retention=1.0, should keep orphaned tag with attributes
        result = ExtractionDataLoader.parse_xml_to_bio(xml, tag_retention_prob=1.0)
        assert '<bibl n="ref">' in result
        assert "[BIBL_START]" not in result

    def test_handles_orphaned_closing_tag(self):
        """Test handling of orphaned closing tag (from excerpting)."""
        # Simulates text starting in middle of citation
        xml = '...continuing text</bibl> more content'

        # With retention=0.0, should convert orphaned tag
        result = ExtractionDataLoader.parse_xml_to_bio(xml, tag_retention_prob=0.0)
        assert "[BIBL_END]" in result
        assert '</bibl>' not in result.replace('[BIBL_END]', '')

        # With retention=1.0, should keep orphaned tag
        result = ExtractionDataLoader.parse_xml_to_bio(xml, tag_retention_prob=1.0)
        assert '</bibl>' in result
        assert "[BIBL_END]" not in result

    def test_handles_nested_tags(self):
        """Test handling of nested citation structures."""
        xml = '<cit><bibl n="1">Hdt. 1.1</bibl><quote>text</quote></cit>'

        # With retention=0.0: CIT stripped, bibl/quote converted
        result = ExtractionDataLoader.parse_xml_to_bio(xml, tag_retention_prob=0.0)
        # CIT tags are completely stripped
        assert "[CIT_START]" not in result
        assert "[CIT_END]" not in result
        assert '<cit>' not in result
        # BIBL and QUOTE converted to special tokens
        assert "[BIBL_START]" in result
        assert "[BIBL_END]" in result
        assert "[QUOTE_START]" in result
        assert "[QUOTE_END]" in result

        # With retention=1.0: CIT stripped, bibl/quote kept with attributes
        result = ExtractionDataLoader.parse_xml_to_bio(xml, tag_retention_prob=1.0)
        # CIT tags still stripped (we never want them)
        assert '<cit>' not in result
        assert '</cit>' not in result
        # BIBL and QUOTE kept with attributes
        assert '<bibl n="1">' in result
        assert '</bibl>' in result
        assert '<quote>' in result
        assert '</quote>' in result

    def test_preserves_other_tags(self):
        """Test that non-citation tags are preserved unchanged."""
        xml = '<p>Text with <bibl>citation</bibl> and <title>title</title></p>'
        result = ExtractionDataLoader.parse_xml_to_bio(xml, tag_retention_prob=0.0)

        # Citation tags converted
        assert "[BIBL_START]" in result
        assert "[BIBL_END]" in result

        # Other tags preserved
        assert '<p>' in result
        assert '</p>' in result
        assert '<title>' in result
        assert '</title>' in result

    def test_handles_tags_without_attributes(self):
        """Test tags without attributes work correctly."""
        xml = '<bibl>simple</bibl> <quote>text</quote>'

        result = ExtractionDataLoader.parse_xml_to_bio(xml, tag_retention_prob=0.0)
        assert "[BIBL_START]" in result and "[BIBL_END]" in result

        result = ExtractionDataLoader.parse_xml_to_bio(xml, tag_retention_prob=1.0)
        assert '<bibl>' in result and '</bibl>' in result

    def test_handles_multiple_attributes(self):
        """Test tags with multiple attributes."""
        xml = '<bibl n="1" type="ancient" xml:lang="grc">Hdt.</bibl>'

        # With retention=1.0, all attributes preserved
        result = ExtractionDataLoader.parse_xml_to_bio(xml, tag_retention_prob=1.0)
        assert 'n="1"' in result
        assert 'type="ancient"' in result
        assert 'xml:lang="grc"' in result

        # With retention=0.0, attributes stripped
        result = ExtractionDataLoader.parse_xml_to_bio(xml, tag_retention_prob=0.0)
        assert 'n="1"' not in result
        assert 'type="ancient"' not in result

    def test_handles_empty_tags(self):
        """Test empty citation tags."""
        xml = 'text <quote></quote> more'

        result = ExtractionDataLoader.parse_xml_to_bio(xml, tag_retention_prob=0.0)
        assert "[QUOTE_START]" in result
        assert "[QUOTE_END]" in result

        result = ExtractionDataLoader.parse_xml_to_bio(xml, tag_retention_prob=1.0)
        assert '<quote>' in result
        assert '</quote>' in result

    def test_strips_cit_tags_completely(self):
        """Test that <cit> tags are completely stripped from training data."""
        xml = '<cit><bibl>Hdt. 1.1</bibl><quote>text</quote></cit>'

        # Phase 1/2: CIT stripped, bibl/quote converted
        result = ExtractionDataLoader.parse_xml_to_bio(xml, tag_retention_prob=0.0)
        assert '<cit>' not in result
        assert '</cit>' not in result
        assert '[CIT_START]' not in result
        assert '[CIT_END]' not in result
        assert '[BIBL_START]' in result
        assert '[QUOTE_START]' in result

        # Phase 3: CIT still stripped, bibl/quote kept
        result = ExtractionDataLoader.parse_xml_to_bio(xml, tag_retention_prob=1.0)
        assert '<cit>' not in result
        assert '</cit>' not in result
        assert '<bibl>' in result
        assert '<quote>' in result

    def test_strips_cit_with_attributes(self):
        """Test that <cit> tags with attributes are stripped."""
        xml = '<cit n="1" type="example"><bibl>text</bibl></cit>'

        result = ExtractionDataLoader.parse_xml_to_bio(xml, tag_retention_prob=0.0)
        assert '<cit' not in result
        assert 'n="1"' not in result
        assert 'type="example"' not in result
        assert '[BIBL_START]' in result

    def test_only_five_labels(self):
        """Test that we only have 5 BIO labels (no CIT)."""
        from perscit_model.extraction.data_loader import BIO_LABELS, LABEL2ID, ID2LABEL

        assert len(BIO_LABELS) == 5
        assert 'B-CIT' not in BIO_LABELS
        assert 'I-CIT' not in BIO_LABELS
        assert 'O' in BIO_LABELS
        assert 'B-BIBL' in BIO_LABELS
        assert 'I-BIBL' in BIO_LABELS
        assert 'B-QUOTE' in BIO_LABELS
        assert 'I-QUOTE' in BIO_LABELS

        # Verify mappings are consistent
        assert len(LABEL2ID) == 5
        assert len(ID2LABEL) == 5

    def test_no_cit_special_tokens(self):
        """Test that CIT special tokens don't exist."""
        from perscit_model.extraction.data_loader import SPECIAL_TOKENS

        assert '[CIT_START]' not in SPECIAL_TOKENS
        assert '[CIT_END]' not in SPECIAL_TOKENS
        assert '[BIBL_START]' in SPECIAL_TOKENS
        assert '[BIBL_END]' in SPECIAL_TOKENS
        assert '[QUOTE_START]' in SPECIAL_TOKENS
        assert '[QUOTE_END]' in SPECIAL_TOKENS
        assert len(SPECIAL_TOKENS) == 4
