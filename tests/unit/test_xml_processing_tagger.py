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
        assert citations[0] == (7, 16, "BIBL", "")  # "Hdt. 8.82" at positions 7-16

    def test_single_quote_tag(self, tagger):
        """Test extracting a single quote tag."""
        xml = "<p>Homer says: <quote>Ajax hurled a rock</quote>.</p>"

        stripped, citations = tagger.strip_citation_tags(xml)

        assert stripped == "<p>Homer says: Ajax hurled a rock.</p>"
        assert len(citations) == 1
        assert citations[0] == (15, 33, "QUOTE", "")  # "Ajax hurled a rock"

    def test_multiple_citations(self, tagger):
        """Test extracting multiple citation tags."""
        xml = "<p>See <bibl>Hdt. 8.82</bibl> and <quote>text</quote> here.</p>"

        stripped, citations = tagger.strip_citation_tags(xml)

        assert stripped == "<p>See Hdt. 8.82 and text here.</p>"
        assert len(citations) == 2
        assert citations[0] == (7, 16, "BIBL", "")
        assert citations[1] == (21, 25, "QUOTE", "")

    def test_cit_tag(self, tagger):
        """Test extracting cit tag wrapping bibl and quote."""
        xml = "<p><cit><bibl>Hdt. 8.82</bibl><quote>text</quote></cit></p>"

        stripped, citations = tagger.strip_citation_tags(xml)

        assert stripped == "<p>Hdt. 8.82text</p>"
        assert len(citations) == 3
        # CIT wraps everything
        assert citations[0] == (3, 16, "CIT", "")
        # BIBL comes first
        assert citations[1] == (3, 12, "BIBL", "")
        # QUOTE comes after
        assert citations[2] == (12, 16, "QUOTE", "")

    def test_empty_bibl_tag(self, tagger):
        """Test handling empty bibl tag."""
        xml = "<p>See <bibl></bibl> for details.</p>"

        stripped, citations = tagger.strip_citation_tags(xml)

        assert stripped == "<p>See  for details.</p>"
        assert len(citations) == 1
        assert citations[0] == (7, 7, "BIBL", "")  # Empty span at position 7

    def test_empty_quote_tag(self, tagger):
        """Test handling empty quote tag."""
        xml = "<p><quote></quote>Nothing quoted.</p>"

        stripped, citations = tagger.strip_citation_tags(xml)

        assert stripped == "<p>Nothing quoted.</p>"
        assert len(citations) == 1
        assert citations[0] == (3, 3, "QUOTE", "")

    def test_whitespace_in_content(self, tagger):
        """Test that whitespace within citation content is preserved."""
        xml = "<p><bibl>Hdt.\t8.82</bibl></p>"

        stripped, citations = tagger.strip_citation_tags(xml)

        assert stripped == "<p>Hdt.\t8.82</p>"
        assert len(citations) == 1
        assert citations[0] == (3, 12, "BIBL", "")

    def test_newlines_in_content(self, tagger):
        """Test that newlines within citation content are preserved."""
        xml = "<p><quote>Line 1\nLine 2</quote></p>"

        stripped, citations = tagger.strip_citation_tags(xml)

        assert stripped == "<p>Line 1\nLine 2</p>"
        assert len(citations) == 1
        assert citations[0] == (3, 16, "QUOTE", "")

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
        assert citations[0] == (3, 12, "BIBL", ' n="Hdt. 8.82"')

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
        assert citations[0] == (3, 7, "BIBL", "")
        assert citations[1] == (7, 13, "QUOTE", "")

    def test_citation_at_document_start(self, tagger):
        """Test citation at the very start of document."""
        xml = "<bibl>Hdt. 8.82</bibl> rest of text"

        stripped, citations = tagger.strip_citation_tags(xml)

        assert stripped == "Hdt. 8.82 rest of text"
        assert len(citations) == 1
        assert citations[0] == (0, 9, "BIBL", "")

    def test_citation_at_document_end(self, tagger):
        """Test citation at the very end of document."""
        xml = "Some text <quote>final quote</quote>"

        stripped, citations = tagger.strip_citation_tags(xml)

        assert stripped == "Some text final quote"
        assert len(citations) == 1
        assert citations[0] == (10, 21, "QUOTE", "")

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


class TestWrapCitationsInCit:
    """Test CitationTagger.post_process method."""

    @pytest.fixture
    def tagger(self, mocker):
        """Create a CitationTagger instance for testing.

        Mocks InferenceModel to avoid loading an actual model.
        """
        # Mock InferenceModel to avoid loading actual model
        mocker.patch("perscit_model.xml_processing.tagger.InferenceModel")
        tagger = CitationTagger(model_path="dummy_path")
        return tagger

    def test_bibl_quote_with_space(self, tagger):
        """Test wrapping bibl followed by quote with space between."""
        xml = "See <bibl>Ref</bibl> <quote>Text</quote>."
        result = tagger.post_process(xml)

        assert result == "See <cit><bibl>Ref</bibl> <quote>Text</quote></cit>."

    def test_quote_bibl_with_space(self, tagger):
        """Test wrapping quote followed by bibl with space between."""
        xml = "See <quote>Text</quote> <bibl>Ref</bibl>."
        result = tagger.post_process(xml)

        assert result == "See <cit><quote>Text</quote> <bibl>Ref</bibl></cit>."

    def test_adjacent_no_space(self, tagger):
        """Test wrapping adjacent tags with no space."""
        xml = "See <bibl>Ref</bibl><quote>Text</quote>."
        result = tagger.post_process(xml)

        assert result == "See <cit><bibl>Ref</bibl><quote>Text</quote></cit>."

    def test_non_whitespace_between_not_wrapped(self, tagger):
        """Test that tags with non-whitespace text between are NOT wrapped."""
        xml = "See <bibl>Ref</bibl> text <quote>Text</quote>."
        result = tagger.post_process(xml)

        # Should remain unchanged
        assert result == xml

    def test_already_wrapped_not_double_wrapped(self, tagger):
        """Test that already wrapped citations are NOT double-wrapped."""
        xml = "See <cit><bibl>Ref</bibl><quote>Text</quote></cit>."
        result = tagger.post_process(xml)

        # Should remain unchanged
        assert result == xml

    def test_same_type_adjacent_are_merged(self, tagger):
        """Test that two bibls or two quotes adjacent ARE merged (orphan handling)."""
        xml1 = "See <bibl>Ref1</bibl><bibl>Ref2</bibl>."
        result1 = tagger.post_process(xml1)
        assert result1 == "See <bibl>Ref1Ref2</bibl>."

        xml2 = "See <quote>Q1</quote><quote>Q2</quote>."
        result2 = tagger.post_process(xml2)
        assert result2 == "See <quote>Q1Q2</quote>."

    def test_multiple_pairs(self, tagger):
        """Test wrapping multiple bibl-quote pairs."""
        xml = "See <bibl>R1</bibl><quote>Q1</quote> and <quote>Q2</quote> <bibl>R2</bibl>."
        result = tagger.post_process(xml)

        assert result == "See <cit><bibl>R1</bibl><quote>Q1</quote></cit> and <cit><quote>Q2</quote> <bibl>R2</bibl></cit>."

    def test_newline_between_tags(self, tagger):
        """Test wrapping tags with newline between them."""
        xml = "See <bibl>Ref</bibl>\n<quote>Text</quote>."
        result = tagger.post_process(xml)

        assert result == "See <cit><bibl>Ref</bibl>\n<quote>Text</quote></cit>."

    def test_multiple_spaces_between_tags(self, tagger):
        """Test wrapping tags with multiple spaces between them."""
        xml = "See <bibl>Ref</bibl>    <quote>Text</quote>."
        result = tagger.post_process(xml)

        # BeautifulSoup normalizes multiple spaces to single space
        assert "<cit>" in result
        assert "<bibl>Ref</bibl>" in result
        assert "<quote>Text</quote>" in result
        assert "</cit>" in result

    def test_tab_between_tags(self, tagger):
        """Test wrapping tags with tab between them."""
        xml = "See <bibl>Ref</bibl>\t<quote>Text</quote>."
        result = tagger.post_process(xml)

        assert "<cit><bibl>Ref</bibl>" in result
        assert "<quote>Text</quote></cit>" in result

    def test_mixed_whitespace_between_tags(self, tagger):
        """Test wrapping tags with mixed whitespace (spaces, tabs, newlines)."""
        xml = "See <bibl>Ref</bibl> \n\t  <quote>Text</quote>."
        result = tagger.post_process(xml)

        assert "<cit><bibl>Ref</bibl>" in result
        assert "<quote>Text</quote></cit>" in result

    def test_no_citations(self, tagger):
        """Test text with no citation tags remains unchanged."""
        xml = "Just plain text."
        result = tagger.post_process(xml)

        assert result == xml

    def test_single_bibl_only(self, tagger):
        """Test single bibl without quote is not wrapped."""
        xml = "See <bibl>Ref</bibl>."
        result = tagger.post_process(xml)

        assert result == xml

    def test_single_quote_only(self, tagger):
        """Test single quote without bibl is not wrapped."""
        xml = "See <quote>Text</quote>."
        result = tagger.post_process(xml)

        assert result == xml

    def test_with_attributes_preserved(self, tagger):
        """Test that citation tag attributes are preserved."""
        xml = 'See <bibl n="1">Ref</bibl> <quote type="paraphrase">Text</quote>.'
        result = tagger.post_process(xml)

        assert '<bibl n="1">Ref</bibl>' in result
        assert '<quote type="paraphrase">Text</quote>' in result
        assert '<cit>' in result
        assert '</cit>' in result

    def test_nested_existing_cit_with_other_tags(self, tagger):
        """Test that nested tags inside cit are handled correctly."""
        xml = "See <cit><bibl>Hdt. <foreign>8.82</foreign></bibl><quote>text</quote></cit>."
        result = tagger.post_process(xml)

        # Should not double-wrap
        assert result.count("<cit>") == 1
        assert result.count("</cit>") == 1
        assert "<foreign>8.82</foreign>" in result

    def test_bibl_quote_separated_by_other_tag(self, tagger):
        """Test bibl and quote separated by another tag are NOT wrapped."""
        xml = "See <bibl>Ref</bibl><foreign>text</foreign><quote>Text</quote>."
        result = tagger.post_process(xml)

        # Should not wrap because there's a tag between them
        assert "<cit>" not in result
