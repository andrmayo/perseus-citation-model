from typing import cast, Iterator

from bs4 import BeautifulSoup
from bs4.element import PageElement


def parse_xml_to_bio(xml_context: str) -> tuple[list[str], list[str]]:
    """
    Parse XML and create BIO tags for citation elements.

    Handles malformed XML by wrapping in root element and using lxml HTML mode.
    Extracts tokens and assigns BIO tags based on immediate parent element:
    - <bibl> → B-BIBL, I-BIBL
    - <quote> → B-QUOTE, I-QUOTE
    - <cit> → B-CIT, I-CIT
    - Other → O

    Note on nested tags:
        For nested structures like <cit><bibl>text</bibl></cit>, we use the
        immediate parent (innermost tag). So "text" gets BIBL tags, not CIT.
        This is the "innermost wins" strategy.

    Note on tags orphaned in excerpting:
        The lxml HTML mode parser will ignore closing orphaned tags at the start,
        and will close orphaned opening tags at the end of a context. This will
        lead to some mislabelled tokens.

    Note on discontinuous tags:
        If a tag contains other elements (e.g., <bibl>Hdt. <title>El.</title> 751</bibl>),
        each text node is labeled independently. This may result in multiple B- tags
        for the same logical entity, which the model will learn to handle.

    Args:
        xml_context: XML snippet (may be malformed from excerpting)

    Returns:
        Tuple of (tokens, labels) where both are lists of strings
    """

    # Use lxml instead of xml - more forgiving of malformed markup
    soup = BeautifulSoup(xml_context, "lxml")

    tokens = []
    labels = []

    descendents = soup.descendants
    if descendents is None:
        return tokens, labels
    cast(Iterator[PageElement], descendents)

    for element in descendents:
        if isinstance(element, str):
            text = element.strip()
            if text:
                words = text.split()
                parent_tag = element.parent.name if element.parent else None

                for i, word in enumerate(words):
                    tokens.append(word)

                    # Assign BIO tags based on parent element
                    if parent_tag == "bibl":
                        labels.append("B-BIBL" if i == 0 else "I-BIBL")
                    elif parent_tag == "quote":
                        labels.append("B-QUOTE" if i == 0 else "I-QUOTE")
                    elif parent_tag == "cit":
                        labels.append("B-CIT" if i == 0 else "I-CIT")
                    else:
                        labels.append("O")

    return tokens, labels
