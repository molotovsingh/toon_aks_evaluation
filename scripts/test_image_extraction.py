#!/usr/bin/env python3
"""
Test Image Extraction: JPEG/PNG OCR Pipeline
Validates Docling image processing with OCR
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables BEFORE importing config
from dotenv import load_dotenv
load_dotenv()

from src.core.docling_adapter import DoclingDocumentExtractor
from src.core.config import load_config

def find_test_images():
    """Find available test images in the project"""
    base_dir = Path(__file__).parent.parent

    # Search for image files
    search_paths = [
        base_dir / "tests" / "test_documents",
        base_dir / "sample_pdf",
        base_dir,
    ]

    image_files = []
    for search_path in search_paths:
        if search_path.exists():
            image_files.extend(search_path.glob("**/*.jpg"))
            image_files.extend(search_path.glob("**/*.jpeg"))
            image_files.extend(search_path.glob("**/*.png"))

    return image_files


def test_image_extraction():
    """Test image OCR extraction"""

    print("=" * 80)
    print("IMAGE EXTRACTION TEST")
    print("=" * 80)

    # Find test images
    test_images = find_test_images()

    if not test_images:
        print("⚠️ No test images found in project")
        print()
        print("To test image extraction:")
        print("1. Take a screenshot of a legal document (PDF page)")
        print("2. Save as .jpg or .png in tests/test_documents/")
        print("3. Run this script again")
        print()
        print("Alternatively, test via Streamlit:")
        print("   uv run streamlit run app.py")
        print("   Upload any JPEG/PNG file with text")
        print()
        return

    print(f"✅ Found {len(test_images)} test image(s)")
    print()

    # Initialize extractor
    config, _, _ = load_config()
    extractor = DoclingDocumentExtractor(config)

    print("✅ DoclingDocumentExtractor initialized")
    print()

    # Test each image
    for idx, image_path in enumerate(test_images[:3], 1):  # Limit to 3 images
        print("=" * 80)
        print(f"TEST {idx}: {image_path.name}")
        print("=" * 80)
        print(f"Path: {image_path}")
        print(f"Size: {image_path.stat().st_size / 1024:.1f} KB")
        print()

        # Time the extraction
        start_time = time.perf_counter()

        try:
            # Extract
            doc = extractor.extract(image_path)

            elapsed_time = time.perf_counter() - start_time

            # Display results
            print(f"✅ EXTRACTION SUCCESSFUL ({elapsed_time:.2f}s)")
            print()

            print("-" * 80)
            print("METADATA")
            print("-" * 80)
            print(f"Extraction Method: {doc.metadata.get('extraction_method')}")
            print(f"OCR Enabled: {doc.metadata.get('needs_ocr', 'N/A')}")
            print(f"Plain Text Length: {len(doc.plain_text)} characters")
            print(f"Markdown Length: {len(doc.markdown)} characters")
            print()

            # Quality checks
            print("-" * 80)
            print("QUALITY CHECKS")
            print("-" * 80)

            checks = []

            # Check extraction method
            if doc.metadata.get('extraction_method') == 'docling_image_ocr':
                checks.append("✅ Extraction method: docling_image_ocr")
            else:
                checks.append(f"❌ Unexpected extraction method: {doc.metadata.get('extraction_method')}")

            # Check text extracted
            if len(doc.plain_text) > 50:
                checks.append(f"✅ Substantial text extracted ({len(doc.plain_text)} chars)")
            elif len(doc.plain_text) > 10:
                checks.append(f"⚠️ Minimal text extracted ({len(doc.plain_text)} chars) - low quality image?")
            else:
                checks.append(f"❌ Almost no text extracted ({len(doc.plain_text)} chars)")

            # Check performance
            if elapsed_time < 30:
                checks.append(f"✅ Fast processing ({elapsed_time:.1f}s < 30s target)")
            elif elapsed_time < 60:
                checks.append(f"⚠️ Acceptable processing ({elapsed_time:.1f}s, target was <30s)")
            else:
                checks.append(f"❌ Slow processing ({elapsed_time:.1f}s > 60s)")

            for check in checks:
                print(check)

            # Show sample text
            if len(doc.plain_text) > 0:
                print()
                print("-" * 80)
                print("EXTRACTED TEXT SAMPLE (first 500 chars)")
                print("-" * 80)
                print(doc.plain_text[:500])
                if len(doc.plain_text) > 500:
                    print(f"\n... ({len(doc.plain_text) - 500} more characters)")

            print()

        except Exception as e:
            elapsed_time = time.perf_counter() - start_time
            print(f"❌ EXTRACTION FAILED ({elapsed_time:.2f}s)")
            print(f"Error: {e}")
            print()
            import traceback
            traceback.print_exc()
            print()

    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tested {min(len(test_images), 3)} image(s)")
    print()
    print("Next steps:")
    print("1. Review extracted text quality above")
    print("2. Test via Streamlit: uv run streamlit run app.py")
    print("3. Upload test images and extract legal events")
    print()


if __name__ == "__main__":
    test_image_extraction()
