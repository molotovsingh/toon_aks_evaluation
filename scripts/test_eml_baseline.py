#!/usr/bin/env python3
"""
Baseline Test: Current .EML Extraction Quality
Tests current broken .eml handling to establish baseline
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.document_processor import DocumentProcessor
from src.core.config import load_config

def test_current_eml_extraction():
    """Test current .eml extraction to capture baseline failures"""

    # Initialize processor
    config, _, _ = load_config()
    processor = DocumentProcessor(config)

    # Test file
    test_file = Path("sample_pdf/famas_dispute/RE_ FaMAS GmbH Vs Elcomponics Sales Pvt. Ltd, O_s Amount Euro 245,000, File Ref # 29260CFIN_2024.eml")

    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return

    print("=" * 80)
    print("BASELINE TEST: Current .EML Extraction")
    print("=" * 80)
    print(f"File: {test_file.name}")
    print(f"Size: {test_file.stat().st_size / 1024:.1f} KB")
    print()

    # Extract using current broken code
    text, method = processor.extract_text(test_file, "eml")

    print(f"Extraction Method: {method}")
    print(f"Extracted Text Length: {len(text)} characters")
    print()
    print("=" * 80)
    print("SAMPLE OUTPUT (first 1000 chars):")
    print("=" * 80)
    print(text[:1000])
    print()
    print("=" * 80)
    print("ISSUES DETECTED:")
    print("=" * 80)

    issues = []

    # Check for MIME boundaries
    if "------=" in text or "NextPart" in text:
        issues.append("❌ MIME boundary markers present")

    # Check for quoted-printable encoding
    if "=20" in text or "=\n" in text:
        issues.append("❌ Quoted-printable encoding not decoded")

    # Check for HTML tags
    if "<html" in text.lower() or "</div>" in text.lower():
        issues.append("❌ HTML tags present (not stripped)")

    # Check for Content-Type headers
    if "Content-Type:" in text:
        issues.append("❌ MIME headers included in body")

    # Check for base64
    if "base64" in text.lower():
        issues.append("⚠️ Possible base64 content")

    if issues:
        for issue in issues:
            print(issue)
    else:
        print("✅ No obvious issues detected (unexpected!)")

    print()
    print("=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    print("Current .eml extraction reads raw MIME content without parsing.")
    print("This produces unusable text for legal event extraction.")
    print()

if __name__ == "__main__":
    test_current_eml_extraction()
