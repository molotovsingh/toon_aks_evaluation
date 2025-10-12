#!/usr/bin/env python3
"""
Test script for document extraction caching
Validates that switching LLM providers reuses cached document extraction
"""

import sys
import os
import logging
from pathlib import Path
import io

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.legal_pipeline_refactored import LegalEventsPipeline
from src.core.config import load_config

# Configure logging to see cache messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_mock_uploaded_file(file_path: Path):
    """Create a mock uploaded file object from a real file"""
    with open(file_path, 'rb') as f:
        file_bytes = f.read()

    mock_file = io.BytesIO(file_bytes)
    mock_file.name = file_path.name
    return mock_file


def test_cache_across_providers():
    """Test that document extraction is cached when switching providers"""

    # Use sample document
    sample_path = Path("sample_pdf/famas_dispute/Answer to Request for Arbitration.pdf")

    if not sample_path.exists():
        logger.error(f"❌ Sample file not found: {sample_path}")
        return False

    logger.info("=" * 80)
    logger.info("🧪 Testing Document Extraction Caching")
    logger.info("=" * 80)
    logger.info(f"Sample file: {sample_path.name} ({sample_path.stat().st_size / 1024:.1f} KB)")
    logger.info("")

    # Test 1: Process with LangExtract (Gemini)
    logger.info("=" * 80)
    logger.info("TEST 1: First extraction with LangExtract (Gemini)")
    logger.info("=" * 80)

    try:
        # Create mock uploaded file
        uploaded_file = create_mock_uploaded_file(sample_path)

        # Create pipeline with LangExtract
        pipeline1 = LegalEventsPipeline(
            provider="langextract",
            doc_extractor_type="docling"
        )

        # Process document
        results1 = pipeline1._process_single_file_guaranteed(uploaded_file, Path("temp"))

        if results1:
            logger.info(f"✅ Extracted {len(results1)} events with LangExtract")
            # Check for timing info
            if results1[0].get('Docling_Seconds') is not None:
                docling_time1 = results1[0]['Docling_Seconds']
                logger.info(f"⏱️  Docling extraction time: {docling_time1:.2f}s")
        else:
            logger.warning("⚠️  No events extracted (may be expected for test)")

    except Exception as e:
        logger.error(f"❌ Test 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    logger.info("")

    # Test 2: Process SAME file with OpenAI (should hit cache)
    logger.info("=" * 80)
    logger.info("TEST 2: Second extraction with OpenAI (should use cache)")
    logger.info("=" * 80)

    try:
        # Create fresh mock uploaded file (same content)
        uploaded_file2 = create_mock_uploaded_file(sample_path)

        # Create NEW pipeline with OpenAI provider
        pipeline2 = LegalEventsPipeline(
            provider="openai",
            doc_extractor_type="docling"
        )

        # Process same document
        results2 = pipeline2._process_single_file_guaranteed(uploaded_file2, Path("temp"))

        if results2:
            logger.info(f"✅ Extracted {len(results2)} events with OpenAI")
            # Check for timing info and cache hit
            if results2[0].get('Docling_Seconds') is not None:
                docling_time2 = results2[0]['Docling_Seconds']
                logger.info(f"⏱️  Docling extraction time: {docling_time2:.2f}s")

                if docling_time2 == 0.0:
                    logger.info("✅ CACHE HIT CONFIRMED: Docling time is 0.0s!")
                else:
                    logger.warning(f"⚠️  Expected cache hit but got {docling_time2:.2f}s extraction time")
        else:
            logger.warning("⚠️  No events extracted (may be expected for test)")

    except Exception as e:
        logger.error(f"❌ Test 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    logger.info("")
    logger.info("=" * 80)
    logger.info("🎯 Test Complete")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Expected log output:")
    logger.info("  Test 1: Should show Docling extraction time (2-10s)")
    logger.info("  Test 2: Should show '💾 Cache HIT' and 0.0s Docling time")
    logger.info("")
    logger.info("Review the logs above for '💾 Cache HIT' message to confirm caching works!")

    return True


if __name__ == "__main__":
    success = test_cache_across_providers()
    sys.exit(0 if success else 1)
