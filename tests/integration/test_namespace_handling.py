"""Test namespace and attribute handling in XML reconstruction."""

from pathlib import Path

import pytest


class TestNamespaceHandling:
    """Test that namespaces and special attributes are preserved."""

    @pytest.mark.slow
    def test_xml_namespace_attributes(self, real_tagger, tmp_path):
        """Test that xml:lang and other namespace attributes are preserved."""
        xml_content = """<?xml version='1.0' encoding='UTF-8'?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
    <teiHeader xml:lang="eng">
        <fileDesc>
            <titleStmt>
                <title>Test Document</title>
            </titleStmt>
        </fileDesc>
    </teiHeader>
    <text>
        <body>
            <p>This is a test of namespace attributes.</p>
        </body>
    </text>
</TEI>"""

        input_file = tmp_path / "test.xml"
        input_file.write_text(xml_content, encoding="utf-8")

        # This should not lose the xml: prefix
        real_tagger.process_xml_file(input_file, preserve_existing=False, overwrite=True)

        # Verify output preserves namespace attributes
        result = input_file.read_text(encoding="utf-8")
        assert len(result) > 0
        assert "xml:lang" in result, f"xml:lang attribute was lost. Result: {result[:500]}"
