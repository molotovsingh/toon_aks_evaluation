#!/usr/bin/env python3
"""
Test Gemini Direct API JSON Mode Capability
Critical validation before adding Gemini as document extractor

Context: Oct 2025 testing showed google/gemini-* models scored 0/10 via OpenRouter
Question: Was this OpenRouter's issue or Gemini's fundamental capability?
Purpose: Test direct Gemini API to determine if multimodal doc extraction is viable
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment
load_dotenv()

# Import after path setup
from src.core.constants import LEGAL_EVENTS_PROMPT


def test_basic_json():
    """Test 1: Can Gemini produce valid JSON via direct API?"""
    print("\n" + "=" * 80)
    print("TEST 1: Basic JSON Mode")
    print("=" * 80)

    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = """
Return a JSON object with the following structure:
{
  "test": "success",
  "message": "Gemini JSON mode working"
}

Return ONLY the JSON object, no additional text.
        """.strip()

        response = model.generate_content(prompt)

        print(f"✓ Response received ({len(response.text)} chars)")
        print(f"  Raw: {response.text[:200]}")

        # Try to parse JSON (with markdown wrapper stripping)
        try:
            # Strip markdown wrappers if present
            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:]
            elif text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            data = json.loads(text)
            print(f"  ✅ PASS: Valid JSON parsed (markdown wrapper stripped)")
            print(f"     {json.dumps(data, indent=2)}")
            return True
        except json.JSONDecodeError as e:
            print(f"  ❌ FAIL: Invalid JSON - {e}")
            print(f"     Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


def test_legal_extraction():
    """Test 2: Can Gemini extract legal events in JSON format?"""
    print("\n" + "=" * 80)
    print("TEST 2: Legal Event Extraction (JSON)")
    print("=" * 80)

    legal_text = """
ANSWER TO REQUEST FOR ARBITRATION
Case No. 100/2017
Date: 15 March 2017

This arbitration was filed on 22 February 2017 by Famas GmbH against Elcomponics Sales Pvt Ltd.
The contract was signed on 15 January 2016 in Mumbai.
Payment of USD 500,000 was due on 1 March 2017.
    """.strip()

    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = f"""{LEGAL_EVENTS_PROMPT}

Extract legal events from this document:

{legal_text}

Return ONLY a JSON array of events with the following structure for each event:
{{
  "event_particulars": "description",
  "citation": "",
  "document_reference": "source",
  "date": "YYYY-MM-DD or empty string"
}}
        """

        print("✓ Sending legal text to Gemini...")
        response = model.generate_content(prompt)

        print(f"✓ Response received ({len(response.text)} chars)")

        # Try to parse JSON
        try:
            # Strip markdown wrappers if present
            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            # Parse JSON
            events = json.loads(text)

            # Handle both array and object responses
            if isinstance(events, dict):
                if "events" in events:
                    events = events["events"]
                elif "extractions" in events:
                    events = events["extractions"]
                else:
                    events = [events]  # Single event object

            if not isinstance(events, list):
                print(f"  ❌ FAIL: Not a list - got {type(events)}")
                return False

            print(f"  ✅ PASS: Valid JSON array with {len(events)} events")

            # Validate structure
            all_fields_present = True
            for i, event in enumerate(events, 1):
                has_fields = all(k in event for k in ["event_particulars", "citation", "document_reference", "date"])
                if not has_fields:
                    print(f"     Event {i}: ⚠️  Missing required fields")
                    all_fields_present = False
                else:
                    print(f"     Event {i}: ✓ All fields present")
                    print(f"        Date: {event['date']}")
                    print(f"        Particulars: {event['event_particulars'][:60]}...")

            if all_fields_present:
                print(f"  ✅ BONUS: All events have required fields")

            return True

        except json.JSONDecodeError as e:
            print(f"  ❌ FAIL: Invalid JSON - {e}")
            print(f"     Response: {response.text[:500]}")
            return False

    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comparison_to_openrouter():
    """Test 3: Compare direct API to Oct 2025 OpenRouter results"""
    print("\n" + "=" * 80)
    print("TEST 3: Comparison to Oct 2025 OpenRouter Results")
    print("=" * 80)

    print("\n📊 Oct 2025 OpenRouter Results:")
    print("  • google/gemini-pro-1.5: 0/10 quality, 2/10 reliability")
    print("  • google/gemini-flash-1.5: 0/10 quality, 2/10 reliability")
    print("  • google/gemini-2.0-flash-exp:free: 0/10 quality, 2/10 reliability")
    print("\n  Issue: Models returned responses but JSON mode appeared broken")
    print("  Question: Was this OpenRouter's routing issue or Gemini's capability?")

    print("\n🔬 Direct API Test Results (this run):")
    # Results from tests above will be printed

    return True


def main():
    """Run all validation tests"""
    print("\n" + "=" * 80)
    print("🧪 GEMINI DIRECT API JSON MODE VALIDATION")
    print("=" * 80)
    print("\nPurpose: Determine if Gemini can replace Docling for document extraction")
    print("Critical: Must pass JSON mode tests to proceed with integration")
    print("\nContext: google/gemini-* scored 0/10 via OpenRouter in Oct 2025")
    print("Question: Can direct Gemini API produce valid JSON?")

    # Check API key
    if not os.getenv("GEMINI_API_KEY"):
        print("\n❌ GEMINI_API_KEY not set. Please configure .env file.")
        sys.exit(1)

    results = []

    # Test 1: Basic JSON
    test1_pass = test_basic_json()
    results.append(("Basic JSON Mode", test1_pass))

    # Test 2: Legal Extraction
    test2_pass = test_legal_extraction()
    results.append(("Legal Event Extraction", test2_pass))

    # Test 3: Comparison
    test_comparison_to_openrouter()

    # Summary
    print("\n" + "=" * 80)
    print("📋 TEST SUMMARY")
    print("=" * 80)

    all_passed = all(result[1] for result in results)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        print("\n🎉 RECOMMENDATION: PROCEED with Gemini document extractor integration")
        print("\nNext steps:")
        print("  1. Run A/B comparison: Docling vs Gemini on real PDF")
        print("  2. Measure cost difference (4.5x expected)")
        print("  3. Evaluate quality improvement on complex layouts")
        print("  4. Add UI selector if quality justifies cost")
    else:
        print("❌ TESTS FAILED")
        print("=" * 80)
        print("\n⚠️  RECOMMENDATION: SKIP Gemini document extractor")
        print("\nReasons:")
        print("  • Direct Gemini API has same JSON issues as OpenRouter")
        print("  • Fundamental capability limitation, not routing issue")
        print("  • Cannot reliably produce structured event extraction")
        print("\nAlternatives:")
        print("  • Keep Docling as sole document extractor (proven, free, fast)")
        print("  • Focus on improving event extractor models instead")
        print("  • Revisit when Gemini 3.0 launches with better JSON support")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
