from typing import cast, Iterable
from lxml import etree
from xml import sax


class CitXMLHandler(sax.ContentHandler):
    def __init__(self) -> None:
        self.total_counts = {"cit": 0, "bibl": 0, "quote": 0}
        self.counts = {}
        # Maps doc idx to its counts
        self.doc_counts = {}
        self.total = 0
        self.doc_idx = -1  # so this can be incremented in startDocument
        self.total_char_count = 0
        self.filename: str | None = None
        self.doc_idx_to_filename = {}

    def startDocument(self) -> None:
        if self.filename is None:
            print("Warning: ideally, set self.filename before parsing document")
        self.counts = {"cit": 0, "bibl": 0, "quote": 0}
        self.doc_idx += 1
        # So we can easily see if the idx-th doc failed to parse
        self.doc_counts[self.doc_idx] = None
        self.doc_idx_to_filename[self.doc_idx] = self.filename
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
