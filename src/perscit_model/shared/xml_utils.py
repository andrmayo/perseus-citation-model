from lxml import etree


def get_opening_tag(elem: etree._Element) -> str:
    """Return the opening tag on an lxml element as a string, preserving namespaces."""

    # Element name with prefix - handle non-standard tag types safely
    try:
        tag_name = etree.QName(elem).localname
    except (ValueError, TypeError):
        # Fallback for non-standard tag types
        tag_name = str(elem.tag) if hasattr(elem, 'tag') else 'unknown'

    ns_prefix = f"{elem.prefix}:" if elem.prefix else ""
    full_tag_name = f"{ns_prefix}{tag_name}"

    return f"<{full_tag_name}{get_attrs_as_string(elem)}>"


def get_attrs_as_string(elem: etree._Element) -> str:
    """Return the attributes of elem as single str, with leading space."""

    # Reverse nsmap: namespace URI -> prefix
    reverse_nsmap = {uri: prefix for prefix, uri in elem.nsmap.items() if prefix}
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
