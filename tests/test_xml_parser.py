"""Unit tests for XML parser BIO tagging functionality."""

from perscit_model.extraction.data.xml_parser import parse_xml_to_bio


class TestParseXmlToBio:
    """Test suite for parse_xml_to_bio function."""

    def test_simple_bibl_tag(self):
        """Test basic bibl tag produces correct BIO labels."""
        xml = "<bibl>Hdt. 8.82</bibl>"
        tokens, labels = parse_xml_to_bio(xml)

        assert tokens == ["Hdt.", "8.82"]
        assert labels == ["B-BIBL", "I-BIBL"]

    def test_simple_quote_tag(self):
        """Test basic quote tag produces correct BIO labels."""
        xml = "<quote>τᾶς πολυχρύσου Πυθῶνος</quote>"
        tokens, labels = parse_xml_to_bio(xml)

        assert tokens == ["τᾶς", "πολυχρύσου", "Πυθῶνος"]
        assert labels == ["B-QUOTE", "I-QUOTE", "I-QUOTE"]

    def test_simple_cit_tag(self):
        """Test basic cit tag produces correct BIO labels."""
        xml = "<cit>some citation text</cit>"
        tokens, labels = parse_xml_to_bio(xml)

        assert tokens == ["some", "citation", "text"]
        assert labels == ["B-CIT", "I-CIT", "I-CIT"]

    def test_nested_tags_innermost_wins(self):
        """Test that nested tags use innermost parent for labeling."""
        xml = "<cit><bibl>Soph. OT 151</bibl></cit>"
        tokens, labels = parse_xml_to_bio(xml)

        # Should use BIBL tags (innermost), not CIT
        assert tokens == ["Soph.", "OT", "151"]
        assert labels == ["B-BIBL", "I-BIBL", "I-BIBL"]

    def test_mixed_content_with_other_tags(self):
        """Test content outside citation tags gets O label."""
        xml = "Some text <bibl>Hdt. 1.1</bibl> more text"
        tokens, labels = parse_xml_to_bio(xml)

        assert tokens == ["Some", "text", "Hdt.", "1.1", "more", "text"]
        assert labels == ["O", "O", "B-BIBL", "I-BIBL", "O", "O"]

    def test_discontinuous_bibl_with_nested_element(self):
        """Test bibl containing other elements (multiple B- tags expected)."""
        xml = "<bibl>Hdt. <title>Histories</title> 1.1</bibl>"
        tokens, labels = parse_xml_to_bio(xml)

        # "Hdt." has parent=bibl, "Histories" has parent=title, "1.1" has parent=bibl
        # Each text node with bibl parent gets its own B- tag at position 0
        assert tokens == ["Hdt.", "Histories", "1.1"]
        assert labels[0] == "B-BIBL"  # First token in first bibl text node
        assert labels[1] == "O"  # title tag (not in our target tags)
        assert labels[2] == "B-BIBL"  # First token in second bibl text node

    def test_orphaned_closing_tag_ignored(self):
        """Test that orphaned closing tag is ignored by parser."""
        xml = "some text </cit> more words"
        tokens, labels = parse_xml_to_bio(xml)

        # Closing tag should be ignored, all text gets O labels
        assert tokens == ["some", "text", "more", "words"]
        assert labels == ["O", "O", "O", "O"]

    def test_unclosed_opening_tag_auto_closed(self):
        """Test that unclosed opening tag gets auto-closed."""
        xml = "regular text <bibl>citation reference"
        tokens, labels = parse_xml_to_bio(xml)

        # Parser auto-closes <bibl>, so "citation" and "reference" get BIBL labels
        assert tokens == ["regular", "text", "citation", "reference"]
        assert labels == ["O", "O", "B-BIBL", "I-BIBL"]

    def test_multiple_citations_in_sequence(self):
        """Test multiple separate citations each get their own B- tag."""
        xml = "<bibl>Soph. OT 1</bibl> and <bibl>Eur. Med. 2</bibl>"
        tokens, labels = parse_xml_to_bio(xml)

        assert tokens == ["Soph.", "OT", "1", "and", "Eur.", "Med.", "2"]
        assert labels == [
            "B-BIBL",
            "I-BIBL",
            "I-BIBL",  # First citation
            "O",  # "and"
            "B-BIBL",
            "I-BIBL",
            "I-BIBL",  # Second citation
        ]

    def test_empty_tag(self):
        """Test empty tag produces no tokens or labels."""
        xml = "<bibl></bibl>"
        tokens, labels = parse_xml_to_bio(xml)

        assert tokens == []
        assert labels == []

    def test_whitespace_only_tag(self):
        """Test tag with only whitespace produces no tokens."""
        xml = "<bibl>   \n\t  </bibl>"
        tokens, labels = parse_xml_to_bio(xml)

        assert tokens == []
        assert labels == []

    def test_plain_text_no_tags(self):
        """Test plain text without tags gets all O labels."""
        xml = "just some plain text"
        tokens, labels = parse_xml_to_bio(xml)

        assert tokens == ["just", "some", "plain", "text"]
        assert labels == ["O", "O", "O", "O"]

    def test_complex_nested_structure(self):
        """Test complex nested structure with quote inside cit."""
        xml = """
        <cit>
            <quote>τᾶς πολυχρύσου</quote>
            <bibl>Soph. OT 151</bibl>
        </cit>
        """
        tokens, labels = parse_xml_to_bio(xml)

        # Should have quote tokens with QUOTE labels, then bibl tokens with BIBL labels
        assert "τᾶς" in tokens
        assert "Soph." in tokens

        # Find index of first quote token and first bibl token
        quote_idx = tokens.index("τᾶς")
        bibl_idx = tokens.index("Soph.")

        assert labels[quote_idx] == "B-QUOTE"
        assert labels[quote_idx + 1] == "I-QUOTE"
        assert labels[bibl_idx] == "B-BIBL"

    def test_malformed_mixed_orphaned_tags(self):
        """Test handling of multiple malformed tag scenarios."""
        xml = "</quote> text <bibl>Hdt. 1.1 <cit>more"
        tokens, labels = parse_xml_to_bio(xml)

        # Orphaned </quote> ignored
        # <bibl> contains "Hdt. 1.1"
        # <cit> opened but auto-closed, contains "more"
        assert "text" in tokens
        assert "Hdt." in tokens
        assert "more" in tokens

        text_idx = tokens.index("text")
        hdt_idx = tokens.index("Hdt.")
        more_idx = tokens.index("more")

        assert labels[text_idx] == "O"  # Outside any citation tag
        assert labels[hdt_idx] == "B-BIBL"
        assert labels[more_idx] == "B-CIT"

    def test_tokens_and_labels_same_length(self):
        """Ensure tokens and labels always have same length."""
        test_cases = [
            "<bibl>Hdt. 1.1</bibl>",
            "plain text",
            "<quote>quote</quote> <bibl>bibl</bibl>",
            "</orphaned> <unclosed>text",
            "",
            "   ",
        ]

        for xml in test_cases:
            tokens, labels = parse_xml_to_bio(xml)
            assert len(tokens) == len(labels), f"Mismatch for: {xml}"

    def test_empty_string_input(self):
        """Test empty string input."""
        tokens, labels = parse_xml_to_bio("")

        assert tokens == []
        assert labels == []

    def test_real_world_example_from_dataset(self):
        """Test with realistic example from resolved.jsonl."""
        xml = '<bibl n="Hdt. 8.82">Hdt. 8.82</bibl>'
        tokens, labels = parse_xml_to_bio(xml)

        assert tokens == ["Hdt.", "8.82"]
        assert labels == ["B-BIBL", "I-BIBL"]

    def test_greek_and_latin_mixed_text(self):
        """Test handling of mixed Greek and Latin text."""
        xml = "<quote>γνῶθι σεαυτόν</quote> and <bibl>Plin. NH 15.30</bibl>"
        tokens, labels = parse_xml_to_bio(xml)

        assert "γνῶθι" in tokens
        assert "Plin." in tokens

        greek_idx = tokens.index("γνῶθι")
        latin_idx = tokens.index("Plin.")

        assert labels[greek_idx] == "B-QUOTE"
        assert labels[latin_idx] == "B-BIBL"
