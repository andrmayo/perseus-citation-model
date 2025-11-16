def get_author_data(urn_resolution_data: list[dict[str, str]]) -> list[dict[str, str]]:
    def parse_urn_to_author(urn: str) -> str:
        parts = urn.split(":")
        author_work = parts[3]
        return author_work.split(".")[0]

    def extract_author_classification_fields(item: dict) -> dict[str, str]:
        """Extract fields needed for author classification"""
        return {
            "bibl": item["bibl"],
            "quote": item.get("quote", ""),
            "author": parse_urn_to_author(item["urn"]),
        }

    return [extract_author_classification_fields(item) for item in urn_resolution_data]
