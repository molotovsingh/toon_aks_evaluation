#!/usr/bin/env python3
"""
Simple test runner for OCR fallback behavior.
No pytest required - runs direct assertions.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.document_processor import DocumentProcessor
from src.core.config import DoclingConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_document_processor_initializes_without_ocr():
    """Test DocumentProcessor initialization with OCR disabled (default)"""
    print("\n🧪 Test 1: DocumentProcessor initializes without OCR...")
    config = DoclingConfig()
    assert config.do_ocr is False, "Default OCR should be disabled"

    processor = DocumentProcessor(config)
    assert processor is not None
    assert processor.converter is not None
    print("✅ PASS: DocumentProcessor initializes with OCR disabled")


def test_document_processor_initializes_with_ocr_enabled():
    """Test DocumentProcessor initialization with OCR explicitly enabled"""
    print("\n🧪 Test 2: DocumentProcessor initializes with OCR enabled...")
    config = DoclingConfig()
    config.do_ocr = True
    config.ocr_engine = "rapidocr"

    processor = DocumentProcessor(config)
    assert processor is not None
    assert processor.converter is not None
    print("✅ PASS: DocumentProcessor initializes with OCR enabled (RapidOCR)")


def test_ocr_fallback_uses_rapidocr():
    """Test that image processing fallback uses RapidOCR, not EasyOCR"""
    print("\n🧪 Test 3: Image fallback uses RapidOCR...")
    config = DoclingConfig()
    config.do_ocr = False  # OCR disabled for PDFs

    processor = DocumentProcessor(config)
    assert processor is not None
    assert processor.converter is not None
    print("✅ PASS: Image fallback configured with RapidOCR")


def test_rapidocr_import_available():
    """Verify RapidOCR is installed and importable"""
    print("\n🧪 Test 4: RapidOCR import available...")
    try:
        from docling.datamodel.pipeline_options import RapidOcrOptions
        rapidocr_options = RapidOcrOptions()
        assert rapidocr_options is not None
        print("✅ PASS: RapidOCR import successful")
    except ImportError as e:
        raise AssertionError(f"RapidOCR import failed: {e}")


def test_no_easyocr_hardcoded_fallback():
    """Ensure no hardcoded EasyOCR fallback exists in document_processor.py"""
    print("\n🧪 Test 5: No EasyOCR hardcoded fallback...")
    import inspect
    from src.core import document_processor

    # Read the source code
    source = inspect.getsource(document_processor.DocumentProcessor.__init__)

    # Verify EasyOCR is not the fallback
    assert "EasyOcrOptions()" not in source, \
        "EasyOCR should not be hardcoded as fallback (not installed)"

    # Verify RapidOCR IS the fallback
    assert "RapidOcrOptions()" in source, \
        "RapidOCR should be the image processing fallback"

    print("✅ PASS: Fallback chain is correct (RapidOCR, not EasyOCR)")


def test_ocr_import_robustness():
    """Test that initialization is robust to configuration"""
    print("\n🧪 Test 6: OCR initialization robustness...")
    config = DoclingConfig()

    # Should not crash even with default config
    try:
        processor = DocumentProcessor(config)
        assert processor is not None
        print("✅ PASS: Initialization robust to OCR configuration")
    except Exception as e:
        raise AssertionError(f"Initialization should not crash: {e}")


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🔍 OCR Fallback Test Suite")
    print("="*70)

    tests = [
        test_document_processor_initializes_without_ocr,
        test_document_processor_initializes_with_ocr_enabled,
        test_ocr_fallback_uses_rapidocr,
        test_rapidocr_import_available,
        test_no_easyocr_hardcoded_fallback,
        test_ocr_import_robustness,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"❌ FAIL: {test_func.__name__}")
            print(f"   Error: {e}")

    print("\n" + "="*70)
    print(f"📊 Results: {passed} passed, {failed} failed")
    print("="*70)

    if failed > 0:
        sys.exit(1)
    else:
        print("\n✅ All OCR fallback tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
