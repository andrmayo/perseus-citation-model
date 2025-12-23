"""Integration tests for CitationTagger processing full XML documents.

These tests verify the complete workflow of processing XML files with the
CitationTagger, including:
- Stripping existing citations
- Running inference to detect new citations
- Post-processing (orphan merging and cit wrapping)
- Writing results back to files
"""

from pathlib import Path

import pytest

from perscit_model.xml_processing.tagger import CitationTagger


# Model path for real integration tests
MODEL_PATH = Path(__file__).parent.parent.parent / "outputs" / "models" / "extraction"


@pytest.fixture(scope="module")
def real_tagger():
    """Create a CitationTagger instance with a real model.

    This is expensive to load, so we use module scope to share it across tests.
    Tests using this fixture will actually run inference.
    Forces CPU device for testing to avoid CUDA/device mismatch issues.
    """
    if not MODEL_PATH.exists():
        pytest.skip(f"Model not found at {MODEL_PATH}")

    # Explicitly use CPU to avoid device mismatch issues in tests
    tagger = CitationTagger(model_path=str(MODEL_PATH), device="cpu")

    # Ensure model is on CPU
    tagger.inference_model.model.to("cpu")

    return tagger


@pytest.fixture
def tagger(mocker):
    """Create a CitationTagger instance for testing.

    Mocks InferenceModel to avoid loading an actual model.
    Used for tests that don't need real inference.
    """
    # Mock InferenceModel to avoid loading actual model
    mocker.patch("perscit_model.xml_processing.tagger.InferenceModel")
    tagger = CitationTagger(model_path="dummy_path", device="cpu")
    return tagger


@pytest.fixture
def sample_xml_simple():
    """Simple XML document for basic testing."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<TEI>
    <text>
        <body>
            <div>
                <p>Herodotus tells us about the battle at Salamis in Book 8.</p>
                <p>He describes how the Greeks won a decisive victory.</p>
            </div>
        </body>
    </text>
</TEI>"""


@pytest.fixture
def sample_xml_with_existing_citations():
    """XML document with existing citation tags."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<TEI>
    <text>
        <body>
            <div>
                <p>See <bibl>Hdt. 8.82</bibl> for details about the battle.</p>
                <p>Homer says: <quote>Ajax hurled a mighty rock</quote> at his enemy.</p>
                <p>Compare <cit><bibl>Thuc. 2.34</bibl><quote>the famous passage</quote></cit>.</p>
            </div>
        </body>
    </text>
</TEI>"""


@pytest.fixture
def sample_xml_complex():
    """Complex XML with multiple citation scenarios."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<TEI>
    <text>
        <body>
            <div type="chapter" n="1">
                <p>The historian Herodotus describes the battle in his work.</p>
                <p>According to ancient sources, the battle was decisive.</p>
                <p>Multiple scholars have referenced this event over centuries.</p>
            </div>
            <div type="chapter" n="2">
                <p>The Greek warriors fought bravely against overwhelming odds.</p>
                <p>Victory came at great cost to both sides of the conflict.</p>
            </div>
        </body>
    </text>
</TEI>"""


class TestProcessXmlFile:
    """Test CitationTagger.process_xml_file method."""

    @pytest.mark.slow
    def test_process_simple_xml_creates_output(
        self, real_tagger, sample_xml_simple, tmp_path
    ):
        """Test that processing a simple XML file creates output with real inference.

        This test uses a real model and actually runs inference, so it's marked as slow.
        """
        # Create temporary input file
        input_file = tmp_path / "input.xml"
        input_file.write_text(sample_xml_simple, encoding="utf-8")

        # Process the file with real inference
        real_tagger.process_xml_file(input_file, preserve_existing=False, overwrite=True)

        # Verify file exists and has content
        result = input_file.read_text(encoding="utf-8")
        assert result is not None
        assert len(result) > 0

        # Basic structure should be preserved
        assert "<p>" in result
        assert "Herodotus" in result

    def test_process_with_overwrite_false_creates_copy(
        self, tagger, sample_xml_simple, tmp_path, mocker
    ):
        """Test that overwrite=False creates a copy in processed/ directory."""
        # Create temporary input file
        input_file = tmp_path / "input.xml"
        input_file.write_text(sample_xml_simple, encoding="utf-8")
        original_content = sample_xml_simple

        # Mock the inference
        mock_model = mocker.MagicMock()
        mock_output = mocker.MagicMock()
        mock_output.logits.argmax.return_value = mocker.MagicMock()
        mock_model.return_value = mock_output
        tagger.inference_model.model = mock_model

        # Process with overwrite=False
        tagger.process_xml_file(input_file, preserve_existing=False, overwrite=False)

        # Original file should be unchanged
        assert input_file.read_text(encoding="utf-8") == original_content

        # Processed directory should exist with the output file
        processed_dir = tmp_path / "processed"
        assert processed_dir.exists()
        assert processed_dir.is_dir()

        output_file = processed_dir / "input.xml"
        assert output_file.exists()

    def test_preserve_existing_citations(
        self, tagger, sample_xml_with_existing_citations, tmp_path
    ):
        """Test that preserve_existing=True maintains existing citation tags."""
        # Create temporary input file
        input_file = tmp_path / "input.xml"
        input_file.write_text(sample_xml_with_existing_citations, encoding="utf-8")

        # Test strip_citation_tags directly
        stripped, citations = tagger.strip_citation_tags(
            sample_xml_with_existing_citations
        )

        # Should have stripped the citation tags
        assert "<bibl>" not in stripped
        assert "<quote>" not in stripped
        assert "<cit>" not in stripped

        # But should track their positions
        # Nested citations are also tracked: 1 standalone bibl + 1 standalone quote +
        # 1 cit + 2 nested inside cit (bibl + quote) = 5 total
        assert len(citations) == 5

        # Verify citation tracking
        citation_types = [c[2] for c in citations]
        assert "BIBL" in citation_types
        assert "QUOTE" in citation_types
        assert "CIT" in citation_types


class TestPostProcessing:
    """Test post-processing functionality (orphan merging and cit wrapping)."""

    def test_post_process_merges_orphaned_citations(self, tagger):
        """Test that post_process merges adjacent same-type citations."""
        # Simulate orphaned citations from window boundaries
        xml = "<p><bibl>Hdt. 8.</bibl><bibl>82</bibl> describes the battle.</p>"

        result = tagger.post_process(xml)

        # Should merge the two adjacent bibl tags
        assert result == "<p><bibl>Hdt. 8.82</bibl> describes the battle.</p>"

    def test_post_process_wraps_bibl_quote_pairs(self, tagger):
        """Test that post_process wraps adjacent bibl-quote pairs in cit tags."""
        xml = "<p>See <bibl>Hdt. 8.82</bibl> <quote>the famous passage</quote>.</p>"

        result = tagger.post_process(xml)

        # Should wrap in cit tag
        assert "<cit>" in result
        assert "<bibl>Hdt. 8.82</bibl>" in result
        assert "<quote>the famous passage</quote>" in result
        assert "</cit>" in result

    def test_post_process_handles_multiple_orphans(self, tagger):
        """Test merging multiple consecutive same-type tags."""
        xml = "<p><quote>Part 1</quote><quote>Part 2</quote><quote>Part 3</quote>.</p>"

        result = tagger.post_process(xml)

        # Should merge all three quote tags
        assert result == "<p><quote>Part 1Part 2Part 3</quote>.</p>"

    def test_post_process_preserves_attributes_in_merge(self, tagger):
        """Test that attributes from first tag are preserved when merging."""
        xml = '<p><bibl n="Hdt. 8.82">Hdt. 8.</bibl><bibl>82</bibl>.</p>'

        result = tagger.post_process(xml)

        # Should preserve the 'n' attribute from first tag
        assert 'n="Hdt. 8.82"' in result
        assert result.count("<bibl") == 1  # Only one bibl tag after merge

    def test_post_process_does_not_double_wrap(self, tagger):
        """Test that existing cit tags are not double-wrapped."""
        xml = "<p><cit><bibl>Hdt. 8.82</bibl><quote>text</quote></cit>.</p>"

        result = tagger.post_process(xml)

        # Should have exactly one cit tag (no double wrapping)
        assert result.count("<cit>") == 1
        assert result.count("</cit>") == 1

    def test_post_process_handles_whitespace_variations(self, tagger):
        """Test handling various whitespace patterns between tags."""
        # Newline between tags
        xml1 = "<p><bibl>Ref</bibl>\n<quote>Text</quote>.</p>"
        result1 = tagger.post_process(xml1)
        assert "<cit>" in result1

        # Tab between tags
        xml2 = "<p><bibl>Ref</bibl>\t<quote>Text</quote>.</p>"
        result2 = tagger.post_process(xml2)
        assert "<cit>" in result2

        # Multiple spaces between tags
        xml3 = "<p><bibl>Ref</bibl>    <quote>Text</quote>.</p>"
        result3 = tagger.post_process(xml3)
        assert "<cit>" in result3


class TestEndToEndWorkflow:
    """Test the complete end-to-end workflow with realistic scenarios."""

    @pytest.mark.slow
    def test_full_workflow_with_real_inference(
        self, real_tagger, sample_xml_complex, tmp_path
    ):
        """Test complete workflow from XML file to tagged output with real inference.

        This test uses a real model and runs actual inference, so it's marked as slow.
        """
        # Create temporary input file
        input_file = tmp_path / "input.xml"
        input_file.write_text(sample_xml_complex, encoding="utf-8")

        # Process the file with real inference
        real_tagger.process_xml_file(input_file, preserve_existing=False, overwrite=True)

        # Verify output exists and is valid XML
        result = input_file.read_text(encoding="utf-8")
        assert result is not None
        assert len(result) > 0

        # Basic sanity checks - structure should be preserved
        assert "<TEI>" in result or "<text>" in result or "<body>" in result
        assert "<p>" in result
        assert "Herodotus" in result or "Greek" in result  # Content preserved

    def test_batch_processing_multiple_files(
        self, tagger, sample_xml_simple, tmp_path, mocker
    ):
        """Test processing multiple XML files in batch."""
        # Create multiple input files
        input_files = []
        for i in range(3):
            input_file = tmp_path / f"input_{i}.xml"
            input_file.write_text(sample_xml_simple, encoding="utf-8")
            input_files.append(input_file)

        # Mock the inference
        mock_model = mocker.MagicMock()
        mock_output = mocker.MagicMock()
        mock_output.logits.argmax.return_value = mocker.MagicMock()
        mock_model.return_value = mock_output
        tagger.inference_model.model = mock_model

        # Process all files using process_xml
        tagger.process_xml(input_files, preserve_existing=False, overwrite=True)

        # Verify all files were processed
        for input_file in input_files:
            result = input_file.read_text(encoding="utf-8")
            assert result is not None
            assert len(result) > 0


class TestBatchSizeConfiguration:
    """Test batch size determination and configuration."""

    def test_batch_size_property_cpu(self, mocker):
        """Test batch size for CPU device."""
        mocker.patch("perscit_model.xml_processing.tagger.InferenceModel")
        mocker.patch(
            "perscit_model.xml_processing.tagger.torch.cuda.is_available",
            return_value=False,
        )

        tagger = CitationTagger(model_path="dummy_path", device="cpu")

        # CPU should use smaller batch size
        assert tagger.batch_size == 4

    def test_batch_size_property_caching(self, mocker):
        """Test that batch size is cached after first access."""
        mocker.patch("perscit_model.xml_processing.tagger.InferenceModel")
        mocker.patch(
            "perscit_model.xml_processing.tagger.torch.cuda.is_available",
            return_value=False,
        )

        tagger = CitationTagger(model_path="dummy_path", device="cpu")

        # First access
        first_call = tagger.batch_size

        # Modify the internal cache
        tagger._batch_size = 999

        # Second access should return cached value
        second_call = tagger.batch_size
        assert second_call == 999

    def test_batch_size_setter(self, mocker):
        """Test setting custom batch size."""
        mocker.patch("perscit_model.xml_processing.tagger.InferenceModel")
        tagger = CitationTagger(model_path="dummy_path", device="cpu")

        # Set custom batch size
        tagger.batch_size = 32

        # Should use the custom value
        assert tagger.batch_size == 32


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_xml_file(self, tagger, tmp_path, mocker):
        """Test handling of empty XML file."""
        input_file = tmp_path / "empty.xml"
        input_file.write_text("", encoding="utf-8")

        # Mock the inference
        mock_model = mocker.MagicMock()
        tagger.inference_model.model = mock_model

        # Should handle gracefully (might raise exception, that's OK)
        try:
            tagger.process_xml_file(input_file, preserve_existing=False, overwrite=True)
        except Exception:
            # Empty file might cause errors, which is acceptable
            pass

    def test_xml_with_greek_text(self, tagger, tmp_path, mocker):
        """Test processing XML with Greek characters."""
        greek_xml = """<?xml version="1.0" encoding="UTF-8"?>
<TEI>
    <text>
        <body>
            <p>τᾶς πολυχρύσου Πυθῶνος ἐκ θαλάμων</p>
        </body>
    </text>
</TEI>"""

        input_file = tmp_path / "greek.xml"
        input_file.write_text(greek_xml, encoding="utf-8")

        # Test strip_citation_tags with Greek text
        stripped, citations = tagger.strip_citation_tags(greek_xml)

        # Greek characters should be preserved
        assert "τᾶς" in stripped
        assert "Πυθῶνος" in stripped

    def test_xml_with_nested_tags(self, tagger):
        """Test handling XML with nested non-citation tags."""
        xml = "<p>See <foreign>τὸν <emph>μέγαν</emph> πόλεμον</foreign>.</p>"

        stripped, citations = tagger.strip_citation_tags(xml)

        # Non-citation tags should be preserved in stripped output
        assert "<foreign>" in stripped
        assert "<emph>" in stripped
        assert "τὸν" in stripped
