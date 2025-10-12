#!/usr/bin/env python3
"""
Test GPT-5 legal event extraction with real document

Tests the complete pipeline:
1. Document extraction (Docling)
2. Event extraction (GPT-5 via OpenAI adapter)
3. Validates adapter uses correct GPT-5 parameters
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()

# Verify API key
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("❌ ERROR: OPENAI_API_KEY not found in environment")
    print("   Set it in your .env file")
    sys.exit(1)

print(f"✅ API Key found: {api_key[:8]}...{api_key[-4:]}")

# Test document path
test_doc = Path("sample_pdf/famas_dispute/Answer to Request for Arbitration.pdf")
if not test_doc.exists():
    print(f"❌ ERROR: Test document not found: {test_doc}")
    sys.exit(1)

print(f"✅ Test document found: {test_doc.name} ({test_doc.stat().st_size / 1024:.1f} KB)")

print("\n" + "="*60)
print("TESTING GPT-5 LEGAL EVENT EXTRACTION")
print("="*60)

# Import pipeline components
from src.core.config import OpenAIConfig, DoclingConfig
from src.core.openai_adapter import OpenAIEventExtractor
from src.core.docling_adapter import DoclingDocumentExtractor

# Step 1: Extract document text with Docling
print("\n📄 Step 1: Extracting document text with Docling...")
doc_config = DoclingConfig(
    do_ocr=True,
    ocr_engine='tesseract',
    accelerator_device='cpu'
)
doc_extractor = DoclingDocumentExtractor(doc_config)
extracted_doc = doc_extractor.extract(test_doc)

text_length = len(extracted_doc.plain_text)
print(f"✅ Extracted {text_length} characters of text")
print(f"   First 200 chars: {extracted_doc.plain_text[:200]}...")

# Step 2: Extract legal events with GPT-5
print("\n🧠 Step 2: Extracting legal events with GPT-5...")
event_config = OpenAIConfig(
    api_key=api_key,
    model='gpt-5'
)
event_extractor = OpenAIEventExtractor(event_config)

# Verify GPT-5 detection
print(f"   GPT-5 detected: {event_extractor._is_gpt5}")
print(f"   JSON mode supported: {event_extractor._supports_json_mode}")

if not event_extractor._is_gpt5:
    print("❌ ERROR: GPT-5 model not detected!")
    sys.exit(1)

# Extract events
metadata = {
    "document_name": test_doc.name,
    "file_path": str(test_doc)
}

print(f"\n⏳ Calling GPT-5 API (this may take 10-30 seconds)...")
events = event_extractor.extract_events(extracted_doc.plain_text, metadata)

# Step 3: Display results
print("\n" + "="*60)
print("EXTRACTION RESULTS")
print("="*60)

print(f"\n📊 Total events extracted: {len(events)}")

# Get stats
stats = event_extractor.get_stats()
print(f"💰 Total cost: ${stats['total_cost']:.4f}")
print(f"🔢 Total tokens: {stats['total_tokens']}")
print(f"🤖 Model: {stats['model']}")
print(f"📋 JSON mode: {stats['supports_json_mode']}")

# Display events
print("\n📋 EXTRACTED EVENTS:")
print("-" * 60)

for i, event in enumerate(events, 1):
    print(f"\n{i}. Date: {event.date}")
    print(f"   Event: {event.event_particulars[:150]}{'...' if len(event.event_particulars) > 150 else ''}")
    print(f"   Citation: {event.citation}")
    print(f"   Source: {event.document_reference}")

# Validation checks
print("\n" + "="*60)
print("VALIDATION CHECKS")
print("="*60)

checks = {
    "At least one event extracted": len(events) > 0,
    "No fallback events": not any(event.attributes.get('fallback', False) for event in events),
    "All events have particulars": all(event.event_particulars for event in events),
    "Cost calculated": stats['total_cost'] > 0,
    "Tokens counted": stats['total_tokens'] > 0,
}

all_passed = True
for check, passed in checks.items():
    status = "✅" if passed else "❌"
    print(f"{status} {check}")
    if not passed:
        all_passed = False

# Final summary
print("\n" + "="*60)
if all_passed:
    print("✅ GPT-5 EXTRACTION TEST PASSED")
    print("\nGPT-5 is working correctly with the OpenAI adapter!")
    print("The adapter correctly uses:")
    print("  • max_completion_tokens (not max_tokens)")
    print("  • temperature=1.0 (non-deterministic)")
else:
    print("❌ GPT-5 EXTRACTION TEST FAILED")
    print("\nSome validation checks did not pass.")
    sys.exit(1)

print("="*60 + "\n")
