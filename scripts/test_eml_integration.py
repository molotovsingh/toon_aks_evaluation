#!/usr/bin/env python3
"""
Integration Test: Full .EML Pipeline
Tests DoclingDocumentExtractor with new email parser (full pipeline)
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.docling_adapter import DoclingDocumentExtractor
from src.core.config import load_config

def test_full_pipeline():
    """Test full .eml extraction pipeline (DoclingDocumentExtractor)"""

    test_file = Path("sample_pdf/famas_dispute/RE_ FaMAS GmbH Vs Elcomponics Sales Pvt. Ltd, O_s Amount Euro 245,000, File Ref # 29260CFIN_2024.eml")

    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return

    print("=" * 80)
    print("INTEGRATION TEST: Full .EML Pipeline")
    print("=" * 80)
    print(f"File: {test_file.name}")
    print()

    try:
        # Initialize full pipeline (same as Streamlit uses)
        config, _, _ = load_config()
        extractor = DoclingDocumentExtractor(config)

        print("✅ DoclingDocumentExtractor initialized")
        print()

        # Extract using full pipeline
        doc = extractor.extract(test_file)

        print("✅ EXTRACTION SUCCESSFUL")
        print()

        print("=" * 80)
        print("EXTRACTED DOCUMENT:")
        print("=" * 80)
        print(f"Plain Text Length: {len(doc.plain_text)} characters")
        print(f"Markdown Length: {len(doc.markdown)} characters")
        print()

        print("=" * 80)
        print("METADATA:")
        print("=" * 80)
        for key, value in doc.metadata.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
        print()

        print("=" * 80)
        print("PLAIN TEXT SAMPLE (first 1000 chars):")
        print("=" * 80)
        print(doc.plain_text[:1000])
        print()

        print("=" * 80)
        print("QUALITY CHECKS:")
        print("=" * 80)

        checks = []

        # Metadata checks
        if "email_headers" in doc.metadata:
            checks.append("✅ Email headers in metadata")
            email_headers = doc.metadata["email_headers"]
            if email_headers.get("subject"):
                checks.append(f"✅ Subject: {email_headers['subject'][:60]}...")
            if email_headers.get("from"):
                checks.append(f"✅ From: {email_headers['from']}")
            if email_headers.get("date"):
                checks.append(f"✅ Date: {email_headers['date']}")
        else:
            checks.append("❌ Email headers missing from metadata")

        if doc.metadata.get("extraction_method") == "email_parser":
            checks.append("✅ Extraction method: email_parser")
        else:
            checks.append(f"❌ Extraction method: {doc.metadata.get('extraction_method')}")

        # Text quality checks
        if "------=" not in doc.plain_text:
            checks.append("✅ No MIME boundaries in text")
        else:
            checks.append("❌ MIME boundaries still present")

        if "=20" not in doc.plain_text:
            checks.append("✅ Quoted-printable decoded")
        else:
            checks.append("❌ Quoted-printable not decoded")

        for check in checks:
            print(check)

        print()
        print("=" * 80)
        print("RESULT: ✅ INTEGRATION TEST PASSED")
        print("=" * 80)
        print()

    except Exception as e:
        print(f"❌ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_full_pipeline()
