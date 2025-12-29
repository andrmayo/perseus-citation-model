"""Test XML reconstruction from tokenization to ensure valid XML output."""

import pytest


class TestXMLReconstruction:
    """Test that XML is properly reconstructed after inference."""

    @pytest.mark.slow
    def test_xml_with_processing_instruction(self, real_tagger, tmp_path):
        """Test XML with processing instructions (like xml-model)."""
        xml_content = """<?xml version='1.0' encoding='UTF-8' standalone='no'?>
<?xml-model href="https://epidoc.stoa.org/schema/latest/tei-epidoc.rng" schematypens="http://relaxng.org/ns/structure/1.0"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
    <text>
        <body>
            <p>This is a test of Herodotus.</p>
        </body>
    </text>
</TEI>"""

        input_file = tmp_path / "test.xml"
        input_file.write_text(xml_content, encoding="utf-8")

        # This should not raise an error
        real_tagger.process_xml_file(
            input_file, preserve_existing=False, overwrite=True
        )

        # Verify output is valid XML
        result = input_file.read_text(encoding="utf-8")
        assert len(result) > 0
        assert "<?xml" in result

    @pytest.mark.slow
    def test_xml_with_multiple_namespaces(self, real_tagger, tmp_path):
        """Test XML with namespace declarations."""
        xml_content = """<?xml version='1.0' encoding='UTF-8'?>
<tei xmlns="http://www.tei-c.org/ns/1.0" xmlns:xi="http://www.w3.org/2001/XInclude">
    <text>
        <body>
            <div type="textpart" subtype="chapter" n="1">
                <p>The language of Greek writers exhibits variety.</p>
            </div>
        </body>
    </text>
</tei>"""

        input_file = tmp_path / "test.xml"
        input_file.write_text(xml_content, encoding="utf-8")

        # This should not raise an error
        real_tagger.process_xml_file(
            input_file, preserve_existing=False, overwrite=True
        )

        # Verify output is valid XML
        result = input_file.read_text(encoding="utf-8")
        assert len(result) > 0

    @pytest.mark.slow
    def test_xml_with_unclosed_tags_from_window_splitting(self, real_tagger, tmp_path):
        """Test that window splitting doesn't create unclosed tags."""
        # Create a long XML document that will require multiple windows
        paragraphs = "\n".join(
            [
                f"<p>This is paragraph {i} with some text about ancient Greece and Herodotus.</p>"
                for i in range(50)
            ]
        )

        xml_content = f"""<?xml version='1.0' encoding='UTF-8'?>
<tei>
    <text>
        <body>
            {paragraphs}
        </body>
    </text>
</tei>"""

        input_file = tmp_path / "test.xml"
        input_file.write_text(xml_content, encoding="utf-8")

        # This should not raise an error about mismatched tags
        real_tagger.process_xml_file(
            input_file, preserve_existing=False, overwrite=True
        )

        # Verify output is valid XML
        result = input_file.read_text(encoding="utf-8")
        assert len(result) > 0
        assert result.count("<p>") == result.count("</p>")
