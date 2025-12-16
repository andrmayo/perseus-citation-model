import re
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup
from bs4.element import NavigableString

from perscit_model.extraction.evaluate import strip_xml_tags
from perscit_model.extraction.inference import InferenceModel


class CitationTagger:
    def __init__(
        self,
        model_path: str | Path,
        window_size: int = 512,
        stride: int | None = None,
        center: int | None = None,
    ):
        """
        Args:
            model_path: path to model to load
            window_size: this should probably be equal to the training context window
            stride: number of tokens to shift context window by, by default half of the window size
            center: defaults to stride, which ensures that every token winds up in center for some window
        """
        self.model = InferenceModel(model_path)
        self.window_size = window_size
        # this ensures that every token windw up in the reliable center of context window
        self.stride = stride if stride else window_size // 2
        self.center = center if center else self.stride

    def process_xml(
        self,
        xml_path: str | Path | Iterable[str] | Iterable[Path],
        preserve_existing: bool = True,
        copy: bool = False,
    ):
        """
        Args:
            xml_path: path or Iterable of paths
            preserve_existing: if True, do not overwrite existing citation tags
            copy: if True, create a copy of all XML files with taga added rather than modifying in place
        """
        if not isinstance(xml_path, Iterable):
            xml_path = [xml_path]

    # 1. Extract existing citations

    # 2. Create sliding windows
    #
    # 3. Run inference on each window
    #
    # 4. Merge predictions (character-level)
    #
    # 5. Filter conflicts with exisitng citations
    #
    # 6. Wrap bibl-quote pairs
    #
    # 7. Insert tags

    def extract_existing_citations(
        self, xml_content: str
    ) -> tuple[str, list[tuple[int, int, str]]]:
        """
        Extract existing citation tags and their positions in stripped text.

        Returns:
            (stripped_text, citations)
            where citations is a list of [start_char, end_char, tag_type)
            and positions are character offsets in stripped_text
        """

        citation_pattern = re.compile(
            r"<(bibl|quote|cit)(?:\s[^>]*)?>(.*?)</\1>", re.DOTALL | re.IGNORECASE
        )

        citations_original = []
        for match in citation_pattern.finditer(xml_content):
            tag_name = match.group(1).upper()
            content = match.group(2)
            start = match.start()
            end = match.end()

            citations_original.append(
                {
                    "type": tag_name,
                    "original_start": start,
                    "original_end": end,
                    "tag_length": end - start,
                    "content_length": len(content),
                }
            )

        # Strip citation tags
        stripped_text = strip_xml_tags(xml_content)

        # Calculate positions in stripped text
        citations_stripped = []
        chars_removed = 0

        for cit in citations_original:
            start_stripped = cit["original_start"] - chars_removed
            end_stripped = start_stripped + cit["content_length"]

            citations_stripped.append((start_stripped, end_stripped, cit["type"]))

            # tag overhead = opening tag + closing tag
            tag_overhead = cit["tag_length"] - cit["content_length"]
            chars_removed += tag_overhead

        return stripped_text, citations_stripped
