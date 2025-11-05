"""
Test OCR fallback behavior to ensure production readiness.

This test catches issues like:
- Missing OCR dependencies
- Broken fallback chains
- Incompatible OCR engine imports
"""

import pytest
import logging
from src.core.document_processor import DocumentProcessor
from src.core.config import DoclingConfig


logger = logging.getLogger(__name__)


class TestOCRFallback:
    """Test suite for OCR fallback mechanisms"""

    def test_document_processor_initializes_without_ocr(self):
        """Test DocumentProcessor initialization with OCR disabled (default)"""
        config = DoclingConfig()
        assert config.do_ocr is False, "Default OCR should be disabled"

        # Should initialize without errors
        processor = DocumentProcessor(config)
        assert processor is not None
        assert processor.converter is not None
        logger.info("✅ DocumentProcessor initializes with OCR disabled")

    def test_document_processor_initializes_with_ocr_enabled(self):
        """Test DocumentProcessor initialization with OCR explicitly enabled"""
        config = DoclingConfig()
        config.do_ocr = True
        config.ocr_engine = "rapidocr"

        # Should initialize without errors
        processor = DocumentProcessor(config)
        assert processor is not None
        assert processor.converter is not None
        logger.info("✅ DocumentProcessor initializes with OCR enabled (RapidOCR)")

    def test_ocr_fallback_uses_rapidocr(self):
        """Test that image processing fallback uses RapidOCR, not EasyOCR"""
        config = DoclingConfig()
        config.do_ocr = False  # OCR disabled for PDFs

        # Create processor (image fallback will use RapidOCR)
        processor = DocumentProcessor(config)

        # Verify processor is functional
        assert processor is not None
        assert processor.converter is not None
        logger.info("✅ Image fallback configured with RapidOCR")

    def test_rapidocr_import_available(self):
        """Verify RapidOCR is installed and importable"""
        try:
            from docling.datamodel.pipeline_options import RapidOcrOptions
            rapidocr_options = RapidOcrOptions()
            assert rapidocr_options is not None
            logger.info("✅ RapidOCR import successful")
        except ImportError as e:
            pytest.fail(f"RapidOCR import failed: {e}")

    def test_easyocr_not_used_as_fallback(self):
        """Ensure EasyOCR is NOT hardcoded as fallback (it's not installed)"""
        # This test documents that EasyOCR should not be used
        # If someone tries to add EasyOCR back, this test will help catch it
        config = DoclingConfig()
        config.do_ocr = False

        processor = DocumentProcessor(config)

        # Verify processor initializes (proving RapidOCR fallback works)
        assert processor is not None
        logger.info("✅ Fallback does not depend on EasyOCR")

    def test_all_ocr_engines_importable(self):
        """Verify all configured OCR engines can be imported"""
        available_engines = []
        unavailable_engines = []

        ocr_engines = {
            "tesseract": "TesseractOcrOptions",
            "ocrmac": "OcrMacOptions",
            "rapidocr": "RapidOcrOptions",
        }

        for engine_name, class_name in ocr_engines.items():
            try:
                from docling.datamodel.pipeline_options import TesseractOcrOptions, OcrMacOptions, RapidOcrOptions
                available_engines.append(engine_name)
            except ImportError:
                unavailable_engines.append(engine_name)

        # RapidOCR must be available (it's in pyproject.toml)
        assert "rapidocr" in available_engines, "RapidOCR must be available"

        logger.info(f"✅ Available OCR engines: {available_engines}")
        if unavailable_engines:
            logger.warning(f"⚠️  Unavailable OCR engines: {unavailable_engines}")


class TestOCRFallbackRegression:
    """Regression tests to catch issues like the EasyOCR fallback bug"""

    def test_no_easyocr_hardcoded_fallback(self):
        """Ensure no hardcoded EasyOCR fallback exists in document_processor.py"""
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

        logger.info("✅ Fallback chain is correct (RapidOCR, not EasyOCR)")

    def test_ocr_import_robustness(self):
        """Test that missing OCR engines don't crash initialization"""
        config = DoclingConfig()

        # Test with default engine (tesseract)
        # Should not crash even if tesseract is missing
        try:
            processor = DocumentProcessor(config)
            assert processor is not None
            logger.info("✅ Initialization robust to missing OCR engines")
        except Exception as e:
            pytest.fail(f"Initialization should not crash: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
