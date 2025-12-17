from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag

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

    def strip_citation_tags(
        self, xml_content: str
    ) -> tuple[str, list[tuple[int, int, str]]]:
        """
        Strip citation tags and track their positions in the stripped text.

        Returns:
            (stripped_text, citations)
            where citations is a list of (start_char, end_char, tag_type)
            and positions are character offsets in stripped_text
        """

        # Parse XML with BeautifulSoup
        soup = BeautifulSoup(xml_content, "lxml")

        # Build stripped text while tracking citation positions
        citations = []
        stripped_parts = []
        current_pos = 0

        def traverse(element: Tag) -> None:
            """Recursively traverse the parse tree, building stripped text."""
            nonlocal current_pos

            # Check if this element is a citation tag
            is_citation = element.name.lower() in (
                "bibl",
                "quote",
                "cit",
            )

            start_pos = None
            if is_citation:
                # Record the start position for citation
                start_pos = current_pos
            else:
                # Foer non-citation tags, include the opening tag with attributes
                attrs = "".join(
                    f' {k}="{v}"' if isinstance(v, str) else f' {k}="{" ".join(v)}"'
                    for k, v in element.attrs.items()
                )
                opening_tag = f"<{element.name}{attrs}>"
                stripped_parts.append(opening_tag)
                current_pos += len(opening_tag)

            # Process the element's contents
            for child in element.children:
                if isinstance(child, NavigableString):
                    # Add text content to stripped output
                    text = str(child)
                    stripped_parts.append(text)
                    current_pos += len(text)
                elif isinstance(child, Tag):
                    # Recursively process child tags
                    traverse(child)

            if is_citation:
                # Record the citation span
                end_pos = current_pos
                citations.append((start_pos, end_pos, element.name.upper()))
            else:
                # For non-citation tags, include the closing tag
                closing_tag = f"</{element.name}>"
                stripped_parts.append(closing_tag)
                current_pos += len(closing_tag)

        # Start traversal from the root
        # lxml parser wraps content in <html><body>, so process body's children
        body = soup.find("body")
        if body:
            for child in body.children:
                if isinstance(child, NavigableString):
                    text = str(child)
                    stripped_parts.append(text)
                    current_pos += len(text)
                elif isinstance(child, Tag):
                    traverse(child)
        else:
            # Fallback in case no body is found
            traverse(soup)
        stripped_text = "".join(stripped_parts)

        # outer tags come before inner tags
        citations.sort(key=lambda x: (x[0], -x[1]))

        return stripped_text, citations
