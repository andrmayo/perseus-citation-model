"""Unit tests for xml_processing/tagger.py"""

from pathlib import Path
from lxml import etree as ET
from perscit_model.xml_processing.tagger import CitationTagger


class TestFixMalformedPredictions:
    """Test CitationTagger._fix_malformed_predictions static method."""

    def test_malformation_repair_with_fixtures(self):
        """Test malformation repair using real XML fixtures.

        This test uses tests/fixtures/malformation_test.xml, which contains
        special tokens like [BIBL_START] embedded in invalid positions
        (e.g. inside tags), and uses tests/fixtures/malformation_expected.xml
        to verify they are correctly repositioned and converted into
        XML tags.
        """
        fixtures_dir = Path(__file__).parent.parent / "fixtures"
        input_xml = (fixtures_dir / "malformation_test.xml").read_text(encoding="utf-8")
        expected_xml = (fixtures_dir / "malformation_expected.xml").read_text(
            encoding="utf-8"
        )

        result = CitationTagger._fix_malformed_predictions(input_xml)
        result_str = "".join(result)
        assert result_str == expected_xml
        # Test that output can be parsed with strict parser
        ET.fromstring(result_str.encode("utf-8"))
