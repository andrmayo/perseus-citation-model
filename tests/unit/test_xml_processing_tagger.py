"""Unit tests for xml_processing/tagger.py"""

import pytest
from perscit_model.xml_processing.tagger import CitationTagger


class TestStripCitationTags:
    """Test CitationTagger.strip_citation_tags method."""

    @pytest.fixture
    def tagger(self, mocker):
        """Create a CitationTagger instance for testing.

        Mocks InferenceModel to avoid loading an actual model.
        """
        # Mock InferenceModel to avoid loading actual model
        mocker.patch("perscit_model.xml_processing.tagger.InferenceModel")
        tagger = CitationTagger(model_path="dummy_path")
        return tagger

    def test_single_bibl_tag(self, tagger):
        """Test extracting a single bibl tag."""
        xml = "<p>See <bibl>Hdt. 8.82</bibl> for details.</p>"

        stripped, citations = tagger.strip_citation_tags(xml)

        assert stripped == "<p>See Hdt. 8.82 for details.</p>"
        assert len(citations) == 1
        assert citations[0] == (7, 16, "BIBL")  # "Hdt. 8.82" at positions 7-16

    def test_single_quote_tag(self, tagger):
        """Test extracting a single quote tag."""
        xml = "<p>Homer says: <quote>Ajax hurled a rock</quote>.</p>"

        stripped, citations = tagger.strip_citation_tags(xml)

        assert stripped == "<p>Homer says: Ajax hurled a rock.</p>"
        assert len(citations) == 1
        assert citations[0] == (15, 33, "QUOTE")  # "Ajax hurled a rock"

    def test_multiple_citations(self, tagger):
        """Test extracting multiple citation tags."""
        xml = "<p>See <bibl>Hdt. 8.82</bibl> and <quote>text</quote> here.</p>"

        stripped, citations = tagger.strip_citation_tags(xml)

        assert stripped == "<p>See Hdt. 8.82 and text here.</p>"
        assert len(citations) == 2
        assert citations[0] == (7, 16, "BIBL")
        assert citations[1] == (21, 25, "QUOTE")

    def test_cit_tag(self, tagger):
        """Test extracting cit tag wrapping bibl and quote."""
        xml = "<p><cit><bibl>Hdt. 8.82</bibl><quote>text</quote></cit></p>"

        stripped, citations = tagger.strip_citation_tags(xml)

        assert stripped == "<p>Hdt. 8.82text</p>"
        assert len(citations) == 3
        # CIT wraps everything
        assert citations[0] == (3, 16, "CIT")
        # BIBL comes first
        assert citations[1] == (3, 12, "BIBL")
        # QUOTE comes after
        assert citations[2] == (12, 16, "QUOTE")

    def test_empty_bibl_tag(self, tagger):
        """Test handling empty bibl tag."""
        xml = "<p>See <bibl></bibl> for details.</p>"

        stripped, citations = tagger.strip_citation_tags(xml)

        assert stripped == "<p>See  for details.</p>"
        assert len(citations) == 1
        assert citations[0] == (7, 7, "BIBL")  # Empty span at position 7

    def test_empty_quote_tag(self, tagger):
        """Test handling empty quote tag."""
        xml = "<p><quote></quote>Nothing quoted.</p>"

        stripped, citations = tagger.strip_citation_tags(xml)

        assert stripped == "<p>Nothing quoted.</p>"
        assert len(citations) == 1
        assert citations[0] == (3, 3, "QUOTE")

    def test_whitespace_in_content(self, tagger):
        """Test that whitespace within citation content is preserved."""
        xml = "<p><bibl>Hdt.\t8.82</bibl></p>"

        stripped, citations = tagger.strip_citation_tags(xml)

        assert stripped == "<p>Hdt.\t8.82</p>"
        assert len(citations) == 1
        assert citations[0] == (3, 12, "BIBL")

    def test_newlines_in_content(self, tagger):
        """Test that newlines within citation content are preserved."""
        xml = "<p><quote>Line 1\nLine 2</quote></p>"

        stripped, citations = tagger.strip_citation_tags(xml)

        assert stripped == "<p>Line 1\nLine 2</p>"
        assert len(citations) == 1
        assert citations[0] == (3, 16, "QUOTE")

    def test_mixed_whitespace(self, tagger):
        """Test mixed whitespace (spaces, tabs, newlines)."""
        xml = "<p>\n\t<bibl>Hdt. 8.82</bibl>\n\t<quote>text here</quote>\n</p>"

        stripped, citations = tagger.strip_citation_tags(xml)

        # Whitespace outside tags should be preserved
        assert "\n" in stripped
        assert len(citations) == 2

    def test_citation_with_attributes(self, tagger):
        """Test citations with attributes are handled correctly."""
        xml = '<p><bibl n="Hdt. 8.82">Hdt. 8.82</bibl></p>'

        stripped, citations = tagger.strip_citation_tags(xml)

        assert stripped == "<p>Hdt. 8.82</p>"
        assert len(citations) == 1
        assert citations[0] == (3, 12, "BIBL")

    def test_no_citations(self, tagger):
        """Test XML with no citation tags."""
        xml = "<p>Just regular text with no citations.</p>"

        stripped, citations = tagger.strip_citation_tags(xml)

        assert stripped == "<p>Just regular text with no citations.</p>"
        assert len(citations) == 0

    def test_adjacent_citations(self, tagger):
        """Test adjacent citation tags with no space between."""
        xml = "<p><bibl>Ref1</bibl><quote>Quote1</quote></p>"

        stripped, citations = tagger.strip_citation_tags(xml)

        assert stripped == "<p>Ref1Quote1</p>"
        assert len(citations) == 2
        assert citations[0] == (3, 7, "BIBL")
        assert citations[1] == (7, 13, "QUOTE")

    def test_citation_at_document_start(self, tagger):
        """Test citation at the very start of document."""
        xml = "<bibl>Hdt. 8.82</bibl> rest of text"

        stripped, citations = tagger.strip_citation_tags(xml)

        assert stripped == "Hdt. 8.82 rest of text"
        assert len(citations) == 1
        assert citations[0] == (0, 9, "BIBL")

    def test_citation_at_document_end(self, tagger):
        """Test citation at the very end of document."""
        xml = "Some text <quote>final quote</quote>"

        stripped, citations = tagger.strip_citation_tags(xml)

        assert stripped == "Some text final quote"
        assert len(citations) == 1
        assert citations[0] == (10, 21, "QUOTE")

    def test_case_insensitive_tags(self, tagger):
        """Test that tag names are case-insensitive."""
        xml = "<p><BIBL>Ref</BIBL> and <Quote>text</Quote></p>"

        _, citations = tagger.strip_citation_tags(xml)

        assert len(citations) == 2
        assert citations[0][2] == "BIBL"
        assert citations[1][2] == "QUOTE"

    def test_nested_cit_bibl_quote(self, tagger):
        """Test typical nested structure: cit wrapping bibl and quote."""
        xml = (
            "<p><cit><bibl>Hdt. 8.82</bibl>: <quote>Ajax hurled</quote></cit> text</p>"
        )

        stripped, citations = tagger.strip_citation_tags(xml)

        assert stripped == "<p>Hdt. 8.82: Ajax hurled text</p>"
        assert len(citations) == 3
        # CIT should wrap entire content
        # BIBL and QUOTE are separate
        cit_citation = [c for c in citations if c[2] == "CIT"][0]
        bibl_citation = [c for c in citations if c[2] == "BIBL"][0]
        quote_citation = [c for c in citations if c[2] == "QUOTE"][0]

        # Verify CIT wraps both
        assert cit_citation[0] <= bibl_citation[0]
        assert cit_citation[1] >= quote_citation[1]

    def test_multiline_quote(self, tagger):
        """Test quote spanning multiple lines."""
        xml = """<p>
        Homer says:
        <quote>
            Ajax hurled a rock
            toward the enemy
        </quote>
        in the battle.
        </p>"""

        stripped, citations = tagger.strip_citation_tags(xml)

        # Should preserve the newlines and whitespace in content
        assert "\n" in stripped
        assert len(citations) == 1
        assert citations[0][2] == "QUOTE"

    def test_special_characters_in_content(self, tagger):
        """Test citations containing special characters."""
        xml = "<p><bibl>Hdt. 8.82 & more</bibl></p>"

        _, citations = tagger.strip_citation_tags(xml)

        # & should be preserved (or entity-encoded by BeautifulSoup)
        assert len(citations) == 1
        assert citations[0][2] == "BIBL"

    def test_greek_text_in_citation(self, tagger):
        """Test citations containing Greek text."""
        xml = "<p><quote>τᾶς πολυχρύσου Πυθῶνος</quote></p>"

        stripped, citations = tagger.strip_citation_tags(xml)

        assert "τᾶς" in stripped
        assert len(citations) == 1
        assert citations[0][2] == "QUOTE"
