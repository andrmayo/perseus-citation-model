from typing import Iterable
from lxml import etree


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
