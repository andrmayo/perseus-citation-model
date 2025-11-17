def extract_fields(tag_extraction_data: list[dict[str, str]]) -> list[dict[str, str]]:
    """Extract the fields needed for training citation tagging model."""

    def extract(extraction_item: dict[str, str]) -> dict[str, str]:
        return {
            "xml_context": extraction_item["xml_context"],
            "filename": extraction_item["filename"],
        }

    return [extract(item) for item in tag_extraction_data]
