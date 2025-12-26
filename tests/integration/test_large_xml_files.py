"""Test processing of large, complete XML files."""

from pathlib import Path

import pytest


class TestLargeXMLFiles:
    """Test that large, real-world XML files can be processed end-to-end."""

    @pytest.mark.slow
    def test_campbell_sophlanguage_file(self, real_tagger, tmp_path):
        """Test processing large campbell-sophlanguage-2.xml file (~400KB)."""
        xml_file = Path("notebooks/xml_cit_stripped/campbell-sophlanguage-2.xml")

        if not xml_file.exists():
            pytest.skip(f"XML file not found at {xml_file}")

        # Copy to temp location to avoid modifying original
        test_file = tmp_path / "campbell-sophlanguage-2.xml"
        test_file.write_text(xml_file.read_text(), encoding="utf-8")

        # This should not raise an XML parsing error
        real_tagger.process_xml_file(test_file, preserve_existing=True, overwrite=True)

        # Verify output
        result = test_file.read_text(encoding="utf-8")
        assert len(result) > 0

    @pytest.mark.slow
    def test_viaf_perseus_file(self, real_tagger, tmp_path):
        """Test processing large viaf2603144.viaf001.perseus-eng1.xml file (~1MB)."""
        xml_file = Path("notebooks/xml_cit_stripped/viaf2603144.viaf001.perseus-eng1.xml")

        if not xml_file.exists():
            pytest.skip(f"XML file not found at {xml_file}")

        # Copy to temp location to avoid modifying original
        test_file = tmp_path / "viaf2603144.viaf001.perseus-eng1.xml"
        test_file.write_text(xml_file.read_text(), encoding="utf-8")

        # This should not raise an XML parsing error
        real_tagger.process_xml_file(test_file, preserve_existing=True, overwrite=True)

        # Verify output
        result = test_file.read_text(encoding="utf-8")
        assert len(result) > 0
