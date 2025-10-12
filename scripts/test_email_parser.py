#!/usr/bin/env python3
"""
Quick Test: New Email Parser
Validates parser on Famas .eml file
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.email_parser import parse_email_file, format_email_as_text, get_email_metadata

def test_new_parser():
    """Test new email parser on Famas email"""

    test_file = Path("sample_pdf/famas_dispute/RE_ FaMAS GmbH Vs Elcomponics Sales Pvt. Ltd, O_s Amount Euro 245,000, File Ref # 29260CFIN_2024.eml")

    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return

    print("=" * 80)
    print("PARSER TEST: New .EML Parser")
    print("=" * 80)
    print(f"File: {test_file.name}")
    print()

    try:
        # Parse email
        parsed = parse_email_file(test_file)

        print("✅ PARSING SUCCESSFUL")
        print()
        print("=" * 80)
        print("EXTRACTED METADATA:")
        print("=" * 80)
        print(f"Subject: {parsed.subject}")
        print(f"From: {parsed.from_addr}")
        print(f"To: {parsed.to_addr[:80]}...")  # Truncate long to field
        print(f"Date: {parsed.date}")
        print(f"Body Format: {parsed.body_format}")
        print(f"Has Attachments: {parsed.has_attachments}")
        print(f"Attachment Count: {parsed.attachment_count}")
        print()

        # Format as text
        formatted_text = format_email_as_text(parsed)

        print("=" * 80)
        print("FORMATTED TEXT (first 1000 chars):")
        print("=" * 80)
        print(formatted_text[:1000])
        print()

        print("=" * 80)
        print("QUALITY CHECKS:")
        print("=" * 80)

        checks = []

        # Check for MIME boundaries (should be GONE)
        if "------=" not in formatted_text and "NextPart" not in formatted_text:
            checks.append("✅ No MIME boundary markers")
        else:
            checks.append("❌ MIME boundaries still present")

        # Check for quoted-printable (should be DECODED)
        if "=20" not in formatted_text and "=E2=80=" not in formatted_text:
            checks.append("✅ Quoted-printable decoded")
        else:
            checks.append("❌ Quoted-printable still encoded")

        # Check for Content-Type headers (should be GONE)
        if "Content-Type:" not in formatted_text:
            checks.append("✅ No MIME headers in body")
        else:
            checks.append("❌ MIME headers still present")

        # Check for clean email headers
        if "Subject:" in formatted_text and "From:" in formatted_text:
            checks.append("✅ Email headers present")
        else:
            checks.append("❌ Email headers missing")

        for check in checks:
            print(check)

        print()
        print("=" * 80)
        print("METADATA FOR ExtractedDocument:")
        print("=" * 80)
        metadata = get_email_metadata(parsed)
        for key, value in metadata.items():
            print(f"{key}: {value}")

        print()

    except Exception as e:
        print(f"❌ PARSING FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_new_parser()
