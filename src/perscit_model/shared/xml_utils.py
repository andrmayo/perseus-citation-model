import json
from typing import TextIO, cast, Iterable
from lxml import etree
from xml import sax

from perscit_model.shared.data_loader import SharedDataLoader


# NOTE: Slight risk of circular imports if SharedDataLoader ever uses a util in this module
class CitXMLHandler(sax.ContentHandler):
    def __init__(self, out: TextIO, chunk_size: int) -> None:
        self.total_counts = {"cit": 0, "bibl": 0, "quote": 0}
        self.counts = {}
        # Maps doc idx to its counts
        self.doc_counts = {}
        self.total = 0
        self.doc_idx = -1  # so this can be incremented in startDocument
        self.total_char_count = 0
        self.out = out
        self.chunk_size = chunk_size
        self._buffer: list[int] = []
        self.data_loader = SharedDataLoader()
        self.filename: str | None = None

    def startDocument(self) -> None:
        if self.filename is None:
            print("Ideally, set self.filename before parsing document")
        self.counts = {"cit": 0, "bibl": 0, "quote": 0}
        self.doc_idx += 1
        # So we can easily see if the idx-th doc failed to parse
        self.doc_counts[self.doc_idx] = None
        self.char_count = 0

    def startElement(self, name, attrs):
        _ = attrs
        if name in self.counts:
            self.counts[name] += 1

    def characters(self, content):
        if not content:
            return

        self.char_count += len(content)
        # without a return_tensors argument, this should return a list[int]
        new_input_ids = cast(
            list[int], self.data_loader.tokenizer(content)["input_ids"]
        )
        assert isinstance(new_input_ids, list) and (
            len(new_input_ids) == 0 or isinstance(new_input_ids[0], int)
        ), f"new_input_ids has type {type(new_input_ids)}"
        self._buffer.extend(new_input_ids)

        while len(self._buffer) >= self.chunk_size:
            chunk = self._buffer[: self.chunk_size]
            json.dump(
                {
                    "window_text": self.data_loader.tokenizer.decode(
                        chunk, clean_up_tokenization_spaces=False
                    ),
                    "filename": self.filename,
                },
                self.out,
            )
            self.out.write("\n")
            self._buffer = self._buffer[self.chunk_size :]

    def endDocument(self) -> None:
        print("Document counts -- ", end="")
        self.doc_counts[self.doc_idx] = {}
        self.total_char_count += self.char_count
        for k, v in self.counts.items():
            self.total_counts[k] += v
            self.total += v
            self.doc_counts[self.doc_idx][k] = v
            print(f"{k}: {v} -- ", end="")
        self.doc_counts[self.doc_idx]["char_count"] = self.char_count

        # handle leftover buffer
        if self._buffer:
            json.dump(
                {
                    "window_text": self.data_loader.tokenizer.decode(
                        self._buffer, clean_up_tokenization_spaces=False
                    ),
                    "filename": self.filename,
                },
                self.out,
            )
            self.out.write("\n")
            self._buffer.clear()


def get_opening_tag(elem: etree._Element) -> str:
    """Return the opening tag on an lxml element as a string, preserving namespaces."""

    # Element name with prefix - handle non-standard tag types safely
    try:
        tag_name = etree.QName(elem).localname
    except (ValueError, TypeError):
        # Fallback for non-standard tag types
        tag_name = str(elem.tag) if hasattr(elem, "tag") else "unknown"

    ns_prefix = f"{elem.prefix}:" if elem.prefix else ""
    full_tag_name = f"{ns_prefix}{tag_name}"

    return f"<{full_tag_name}{get_attrs_as_string(elem)}>"


def get_attrs_as_string(elem: etree._Element) -> str:
    """Return the attributes of elem as single str, with leading space."""

    # Special XML namespace that's always predefined
    XML_NAMESPACE = "http://www.w3.org/XML/1998/namespace"

    # Reverse nsmap: namespace URI -> prefix
    reverse_nsmap = {uri: prefix for prefix, uri in elem.nsmap.items() if prefix}
    # Add the special XML namespace
    reverse_nsmap[XML_NAMESPACE] = "xml"

    # Attributes
    attrs = []
    for k, v in elem.attrib.items():
        # Check if this is already a prefixed name (from HTML parser fallback)
        # HTML parser returns 'xml:lang' while XML parser returns '{namespace}lang'
        if ":" in cast(str, k) and not cast(str, k).startswith("{"):
            # Already in prefixed form (e.g., 'xml:lang'), use as-is
            attr_name = k
        else:
            # Fully qualified form (e.g., '{http://...}lang'), extract components
            qname = etree.QName(k)
            # Find a prefix for this namespace if there i one
            attr_prefix = (
                f"{reverse_nsmap.get(qname.namespace, '')}:" if qname.namespace else ""
            )
            attr_name = f"{attr_prefix}{qname.localname}"
        attrs.append(f'{attr_name}="{v}"')

    if attrs:
        return f" {' '.join(attrs)}"  # Note leading space
    return ""


def strip_spec_elems(
    root: etree._Element | etree._ElementTree, names_to_remove: Iterable[str]
) -> None:
    """Remove tags from all elements specified in names_to_remove, while keeping contents and children."""
    if isinstance(root, etree._ElementTree):
        root = root.getroot()

    ns = etree.QName(root).namespace
    if ns:
        names_to_remove = [f"{{{ns}}}{name}" for name in names_to_remove]

    etree.strip_tags(root, *names_to_remove)
    return


def strip_spec_elem_attrs(
    root: etree._Element | etree._ElementTree, names_to_remove: Iterable[str]
) -> None:
    """Takes an Iterable of element names, and removes attributes of all elements with matching names. Namespsaces are ignored."""
    if isinstance(root, etree._ElementTree):
        root = root.getroot()

    if not isinstance(names_to_remove, set):
        names_to_remove = set(names_to_remove)

    for elem in root.iter():
        if etree.QName(elem).localname in names_to_remove:
            elem.attrib.clear()
    return
