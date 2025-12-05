#!/usr/bin/env python3
"""Test that data loader produces correct labels without extra spaces."""

from perscit_model.extraction.data_loader import ExtractionDataLoader

loader = ExtractionDataLoader()

# Test XML with citation
test_xml = "cf. <bibl>Thuc. III.38</bibl> text"

# Parse to BIO format
parsed = loader.parse_xml_to_bio(test_xml)
print("Original XML:")
print(f"  {repr(test_xml)}")
print("\nParsed (with special tokens):")
print(f"  {repr(parsed)}")

# Check for extra spaces
if " [BIBL_START] " in parsed:
    print("\n❌ FAIL: Extra spaces around special tokens detected!")
    print("   The data loader still has the bug.")
elif "[BIBL_START]Thuc" in parsed:
    print("\n✓ PASS: No extra spaces - special tokens are directly adjacent to content")
    print("   The data loader fix is working correctly.")
else:
    print("\n⚠ WARNING: Unexpected pattern")
    print("   Manual inspection needed.")
