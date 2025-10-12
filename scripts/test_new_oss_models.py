#!/usr/bin/env python3
"""
New Open-Source Models Testing Script
Tests promising OSS models (gpt-oss, Qwen QwQ, Mistral 3.1) for legal event extraction

Based on test_fallback_models.py with added Real Document Test (Test 4)
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime
from dataclasses import dataclass, field

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("❌ Error: requests library not available")
    sys.exit(1)

# Load environment FIRST (before imports that need it)
from dotenv import load_dotenv
load_dotenv()

from src.core.constants import LEGAL_EVENTS_PROMPT

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ModelTestResult:
    """Results for a single model test"""
    model_id: str
    display_name: str
    license_type: str  # "Apache 2.0", "MIT", etc.
    basic_chat_passed: bool = False
    json_mode_passed: bool = False
    legal_extraction_passed: bool = False
    real_doc_passed: bool = False  # NEW
    response_time: float = 0.0
    tokens_used: int = 0
    quality_score: int = 0
    cost_input_per_million: float = 0.0
    cost_output_per_million: float = 0.0
    error_message: str = ""
    notes: List[str] = field(default_factory=list)
    json_clean: bool = False
    all_fields_present: bool = False
    real_doc_event_count: int = 0  # NEW
    reliability_score: int = 0  # 0-10 based on quirks


class NewOSSModelTester:
    """Test new open-source models for legal extraction"""

    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.base_url = "https://openrouter.ai/api/v1"
        self.log_file = Path(__file__).parent / "new_oss_models_test.log"
        self.results: List[ModelTestResult] = []

        # Setup file logging
        file_handler = logging.FileHandler(self.log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

        # Priority models to test (cost per M is input/output)
        self.models_to_test = [
            # OpenAI GPT-OSS (Apache 2.0, Aug 2025)
            ("openai/gpt-oss-20b", "GPT-OSS 20B", "Apache 2.0", 0.03, 0.14),
            ("openai/gpt-oss-120b", "GPT-OSS 120B", "Apache 2.0", 0.04, 0.40),

            # Qwen QwQ (Open Source, Reasoning)
            ("qwen/qwq-32b", "Qwen QwQ 32B", "Open Source", 0.04, 0.14),

            # Mistral Small 3.1 (Apache 2.0, Jan 2025 - upgrade to existing)
            ("mistralai/mistral-small-24b-instruct-2501", "Mistral Small 3.1", "Apache 2.0", 0.20, 0.20),
        ]

        # Legal text for Test 3
        self.legal_text = """
        Motion to Dismiss filed on January 15, 2024, pursuant to Federal Rules of Civil Procedure 12(b)(6).
        The defendant argues failure to state a claim upon which relief can be granted.
        Plaintiff has 21 days to respond per Local Rule 7.1. Hearing scheduled for March 1, 2024
        in the United States District Court, Northern District of California, Case No. 3:24-cv-00123.
        """

    def log(self, message: str):
        """Log to both console and file"""
        logger.info(message)

    def print_header(self):
        """Print test header"""
        print("\n" + "=" * 85)
        print("🧪 New Open-Source Models Testing for Legal Extraction")
        print("=" * 85)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Testing {len(self.models_to_test)} priority OSS models")
        print(f"Log file: {self.log_file}")
        print("=" * 85 + "\n")

    def test_basic_chat(self, model_id: str) -> Tuple[bool, float, int, str]:
        """Test 1: Basic chat completion"""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model_id,
            "messages": [
                {"role": "user", "content": "Reply with only the word 'OK'."}
            ],
            "max_tokens": 10,
            "temperature": 0.0
        }

        try:
            start_time = time.time()
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            elapsed = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                tokens = data.get("usage", {}).get("total_tokens", 0)
                return True, elapsed, tokens, ""
            else:
                return False, elapsed, 0, f"HTTP {response.status_code}"

        except Exception as e:
            return False, 0.0, 0, str(e)[:100]

    def test_json_mode(self, model_id: str) -> Tuple[bool, bool, float, int, str]:
        """Test 2: JSON mode with response_format"""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": 'Return: {"test": "ok", "value": 42}'}
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": 50,
            "temperature": 0.0
        }

        try:
            start_time = time.time()
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            elapsed = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                tokens = data.get("usage", {}).get("total_tokens", 0)

                clean = not ("```" in content)

                try:
                    if "```json" in content:
                        json_text = content.split("```json")[1].split("```")[0].strip()
                        json.loads(json_text)
                        return True, False, elapsed, tokens, ""
                    else:
                        json.loads(content.strip())
                        return True, clean, elapsed, tokens, ""
                except json.JSONDecodeError as e:
                    return False, False, elapsed, tokens, f"Invalid JSON: {str(e)[:50]}"
            else:
                return False, False, elapsed, 0, f"HTTP {response.status_code}"

        except Exception as e:
            return False, False, 0.0, 0, str(e)[:100]

    def test_legal_extraction(self, model_id: str) -> Tuple[bool, bool, bool, float, int, str]:
        """Test 3: Legal event extraction with LEGAL_EVENTS_PROMPT"""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model_id,
            "messages": [
                {
                    "role": "system",
                    "content": LEGAL_EVENTS_PROMPT + "\n\nReturn ONLY valid JSON array."
                },
                {
                    "role": "user",
                    "content": f"Extract legal events:\n\n{self.legal_text}"
                }
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "max_tokens": 800
        }

        try:
            start_time = time.time()
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            elapsed = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                tokens = data.get("usage", {}).get("total_tokens", 0)

                clean = not ("```" in content)

                try:
                    clean_content = content.strip()
                    if "```json" in clean_content:
                        clean_content = clean_content.split("```json")[1].split("```")[0].strip()
                        clean = False
                    elif "```" in clean_content:
                        clean_content = clean_content.split("```")[1].split("```")[0].strip()
                        clean = False

                    parsed = json.loads(clean_content)

                    events = parsed
                    if isinstance(parsed, dict):
                        if "events" in parsed:
                            events = parsed["events"]
                        else:
                            events = [parsed]

                    if not isinstance(events, list):
                        return False, False, False, elapsed, tokens, "Not a list"

                    all_fields_present = True
                    for event in events:
                        if not isinstance(event, dict):
                            all_fields_present = False
                            break
                        required = ["event_particulars", "citation", "document_reference", "date"]
                        for field in required:
                            if field not in event:
                                all_fields_present = False
                                break

                    return True, clean, all_fields_present, elapsed, tokens, ""

                except json.JSONDecodeError as e:
                    return False, False, False, elapsed, tokens, f"JSON error: {str(e)[:50]}"
            else:
                return False, False, False, elapsed, 0, f"HTTP {response.status_code}"

        except Exception as e:
            return False, False, False, 0.0, 0, str(e)[:100]

    def test_real_document(self, model_id: str) -> Tuple[bool, int, float, int, str]:
        """Test 4: Real document extraction (Famas arbitration PDF excerpt)

        Uses a sample excerpt from the Famas dispute PDF to test real-world extraction
        """
        # Sample excerpt from Famas arbitration case (shortened for API call)
        real_legal_text = """
        ANSWER TO REQUEST FOR ARBITRATION

        Case No. 100/2017
        Date: 15 March 2017

        RESPONDENT: Elcomponics Sales Pvt. Ltd.
        CLAIMANT: Famas GmbH

        1. Respondent respectfully submits this Answer to the Request for Arbitration dated
        10 February 2017 filed by Famas GmbH pursuant to the Agreement dated 1 January 2015.

        2. On 1 January 2015, the parties entered into a Distribution Agreement whereby the
        Claimant appointed Respondent as exclusive distributor for electronic components in India.

        3. Clause 12 of the Agreement provides for arbitration under ICC Rules, with seat in Geneva.

        4. Claimant alleges breach of payment obligations under Invoices No. 2016/045 (EUR 50,000)
        and 2016/078 (EUR 75,000), both dated 15 September 2016.

        5. Respondent denies liability and counterclaims EUR 100,000 for defective goods supplied
        in July 2016, which resulted in customer complaints and reputational damage.
        """

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model_id,
            "messages": [
                {
                    "role": "system",
                    "content": LEGAL_EVENTS_PROMPT + "\n\nReturn ONLY valid JSON array."
                },
                {
                    "role": "user",
                    "content": f"Extract legal events:\n\n{real_legal_text}"
                }
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "max_tokens": 1500
        }

        try:
            start_time = time.time()
            response = requests.post(url, headers=headers, json=payload, timeout=90)
            elapsed = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                tokens = data.get("usage", {}).get("total_tokens", 0)

                try:
                    clean_content = content.strip()
                    if "```json" in clean_content:
                        clean_content = clean_content.split("```json")[1].split("```")[0].strip()
                    elif "```" in clean_content:
                        clean_content = clean_content.split("```")[1].split("```")[0].strip()

                    parsed = json.loads(clean_content)

                    events = parsed
                    if isinstance(parsed, dict):
                        if "events" in parsed:
                            events = parsed["events"]
                        else:
                            events = [parsed]

                    if not isinstance(events, list):
                        return False, 0, elapsed, tokens, "Response not a list"

                    # Check if events have meaningful content
                    valid_events = 0
                    for event in events:
                        if isinstance(event, dict):
                            particulars = event.get("event_particulars", "")
                            if particulars and len(particulars) > 10:  # At least some description
                                valid_events += 1

                    passed = valid_events >= 3  # Expect at least 3 events from this text
                    return passed, valid_events, elapsed, tokens, ""

                except json.JSONDecodeError as e:
                    return False, 0, elapsed, tokens, f"JSON error: {str(e)[:50]}"
            else:
                return False, 0, elapsed, 0, f"HTTP {response.status_code}"

        except Exception as e:
            return False, 0, 0.0, 0, str(e)[:100]

    def calculate_scores(self, result: ModelTestResult) -> Tuple[int, int]:
        """Calculate quality and reliability scores"""
        quality = 0
        reliability = 10  # Start at max, deduct for quirks

        if result.basic_chat_passed:
            quality += 2
        else:
            reliability -= 3

        if result.json_mode_passed:
            quality += 2
            if result.json_clean:
                quality += 1
            else:
                reliability -= 2  # Markdown wrapping is a quirk
        else:
            reliability -= 5

        if result.legal_extraction_passed:
            quality += 2
            if result.all_fields_present:
                quality += 1
            else:
                reliability -= 1

        # NEW: Real document test
        if result.real_doc_passed:
            quality += 2  # Major test
        else:
            quality -= 1  # Penalize if failed

        return min(quality, 10), max(reliability, 0)

    def test_model(self, model_id: str, display_name: str, license_type: str,
                   cost_in: float, cost_out: float) -> ModelTestResult:
        """Run all 4 tests for a single model"""
        print(f"\n{'─' * 85}")
        print(f"🧪 Testing: {display_name}")
        print(f"   Model: {model_id}")
        print(f"   License: {license_type}")
        print(f"   Cost: ${cost_in}/M input, ${cost_out}/M output")
        print(f"{'─' * 85}")

        result = ModelTestResult(
            model_id=model_id,
            display_name=display_name,
            license_type=license_type,
            cost_input_per_million=cost_in,
            cost_output_per_million=cost_out
        )

        # Test 1: Basic Chat
        print("   [1/4] Basic chat...", end=" ", flush=True)
        passed, elapsed, tokens, error = self.test_basic_chat(model_id)
        result.basic_chat_passed = passed
        result.response_time += elapsed
        result.tokens_used += tokens
        if passed:
            print(f"✅ ({elapsed:.2f}s)")
        else:
            print(f"❌ {error}")
            result.error_message = error
            result.notes.append(f"Basic chat failed: {error}")

        # Test 2: JSON Mode
        if passed:
            print("   [2/4] JSON mode...", end=" ", flush=True)
            passed, clean, elapsed, tokens, error = self.test_json_mode(model_id)
            result.json_mode_passed = passed
            result.json_clean = clean
            result.response_time += elapsed
            result.tokens_used += tokens
            if passed:
                status = "clean ✨" if clean else "wrapped ⚠️"
                print(f"✅ {status} ({elapsed:.2f}s)")
                if not clean:
                    result.notes.append("JSON in markdown")
            else:
                print(f"❌ {error}")
                result.notes.append(f"JSON failed: {error}")

        # Test 3: Legal Extraction
        if result.json_mode_passed:
            print("   [3/4] Legal extraction...", end=" ", flush=True)
            passed, clean, all_fields, elapsed, tokens, error = self.test_legal_extraction(model_id)
            result.legal_extraction_passed = passed
            result.json_clean = result.json_clean and clean
            result.all_fields_present = all_fields
            result.response_time += elapsed
            result.tokens_used += tokens
            if passed:
                status = "all fields ✓" if all_fields else "missing fields ⚠️"
                print(f"✅ {status} ({elapsed:.2f}s)")
            else:
                print(f"❌ {error}")
                result.notes.append(f"Extraction failed: {error}")
        else:
            print("   [3/4] Legal extraction... ⏭️ Skipped")

        # Test 4: Real Document (NEW)
        if result.legal_extraction_passed:
            print("   [4/4] Real document (Famas case)...", end=" ", flush=True)
            passed, event_count, elapsed, tokens, error = self.test_real_document(model_id)
            result.real_doc_passed = passed
            result.real_doc_event_count = event_count
            result.response_time += elapsed
            result.tokens_used += tokens
            if passed:
                print(f"✅ {event_count} events ✓ ({elapsed:.2f}s)")
            else:
                print(f"❌ {event_count} events (expected ≥3) - {error if error else 'low quality'}")
                result.notes.append(f"Real doc failed: {event_count} events, expected ≥3")
        else:
            print("   [4/4] Real document... ⏭️ Skipped")

        # Calculate scores
        result.quality_score, result.reliability_score = self.calculate_scores(result)

        # Calculate blended cost (3:1 output:input ratio typical for extraction)
        blended_cost = (cost_in + (cost_out * 3)) / 4

        print(f"\n   Quality: {result.quality_score}/10 | Reliability: {result.reliability_score}/10")
        print(f"   Blended Cost: ${blended_cost:.3f}/M | Cost-Eff: {result.quality_score / (blended_cost / 100):.1f}")
        if result.notes:
            print(f"   Notes: {'; '.join(result.notes)}")

        return result

    def print_summary(self):
        """Print comprehensive summary with recommendations"""
        print("\n" + "=" * 85)
        print("📊 NEW OSS MODELS TEST RESULTS")
        print("=" * 85)

        # Sort by quality score
        self.results.sort(key=lambda x: (x.quality_score, x.reliability_score), reverse=True)

        print(f"\n{'Model':<35}{'License':<15}{'Q':<8}{'R':<8}{'Cost':<12}{'Status'}")
        print("─" * 85)
        for r in self.results:
            name = r.display_name[:33]
            license_short = r.license_type[:13]
            quality = f"{r.quality_score}/10"
            reliability = f"{r.reliability_score}/10"
            blended_cost = (r.cost_input_per_million + (r.cost_output_per_million * 3)) / 4
            cost = f"${blended_cost:.3f}/M"

            if r.quality_score >= 9 and r.reliability_score >= 8:
                status = "🥇 EXCELLENT - ADD"
            elif r.quality_score >= 7 and r.reliability_score >= 7:
                status = "🥈 GOOD - CONSIDER"
            elif r.quality_score >= 5:
                status = "🥉 OK - MAYBE"
            else:
                status = "❌ FAILED - SKIP"

            print(f"{name:<35}{license_short:<15}{quality:<8}{reliability:<8}{cost:<12}{status}")

        # Recommendations
        print("\n" + "=" * 85)
        print("🎯 RECOMMENDATIONS")
        print("=" * 85)

        excellent = [r for r in self.results if r.quality_score >= 9 and r.reliability_score >= 8]
        good = [r for r in self.results if 7 <= r.quality_score < 9 and r.reliability_score >= 7]
        failed = [r for r in self.results if r.quality_score < 7]

        if excellent:
            print("\n✅ MODELS TO ADD (≥9/10 quality, ≥8/10 reliability):")
            for r in excellent:
                blended = (r.cost_input_per_million + (r.cost_output_per_million * 3)) / 4
                cost_eff = r.quality_score / (blended / 100)
                print(f"  • {r.model_id}")
                print(f"    {r.license_type} | ${blended:.3f}/M | Cost-Eff: {cost_eff:.1f}")
                print(f"    Update category: ", end="")
                if blended < 0.15:
                    print("'💰 Budget Conscious' (new champion!)")
                elif "mistral" in r.model_id.lower():
                    print("'🌍 Open Source / EU Hosting' (upgrade existing)")
                else:
                    print("'🌍 Open Source / EU Hosting' (new option)")

        if good:
            print("\n⚠️  MODELS TO CONSIDER (7-8/10 quality, may have quirks):")
            for r in good:
                blended = (r.cost_input_per_million + (r.cost_output_per_million * 3)) / 4
                print(f"  • {r.model_id} - ${blended:.3f}/M")
                print(f"    Quirks: {', '.join(r.notes) if r.notes else 'None'}")

        if failed:
            print("\n❌ MODELS TO SKIP (<7/10 quality):")
            for r in failed:
                print(f"  • {r.model_id} - {r.error_message if r.error_message else 'Low quality extraction'}")

        if not excellent and not good:
            print("\n⚠️  NO MODELS PASSED (all <7/10 quality)")
            print("   Recommendation: Keep current 9 models unchanged")
            print("   Document results to help community avoid these models")

        # Cost comparison
        print("\n" + "=" * 85)
        print("💰 COST-EFFECTIVENESS COMPARISON")
        print("=" * 85)
        print(f"{'Model':<40}{'Quality':<10}{'Blended $/M':<15}{'Cost-Eff'}")
        print("─" * 85)

        # Add baseline comparisons
        baselines = [
            ("DeepSeek R1 Distill (current champion)", 10, 0.26),
            ("GPT-4o-mini (current default)", 9, 0.31),
        ]

        for model, quality, cost in baselines:
            cost_eff = quality / (cost / 100)
            print(f"{model:<40}{quality}/10{'':<4}${cost:.3f}{'':<10}{cost_eff:.1f} ⭐")

        print()
        for r in self.results:
            if r.quality_score >= 7:  # Only show passing models
                blended = (r.cost_input_per_million + (r.cost_output_per_million * 3)) / 4
                cost_eff = r.quality_score / (blended / 100)
                name = r.display_name[:38]
                print(f"{name:<40}{r.quality_score}/10{'':<4}${blended:.3f}{'':<10}{cost_eff:.1f}")

        print(f"\n📄 Detailed log: {self.log_file}")
        print("=" * 85 + "\n")

    def run_all_tests(self):
        """Run tests for all models"""
        self.print_header()

        for model_id, display_name, license_type, cost_in, cost_out in self.models_to_test:
            result = self.test_model(model_id, display_name, license_type, cost_in, cost_out)
            self.results.append(result)
            self.log(f"Completed: {model_id} | Q:{result.quality_score}/10 R:{result.reliability_score}/10")

        self.print_summary()


def main():
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ Error: OPENROUTER_API_KEY not found")
        print("   Set it in your .env file to test OpenRouter models")
        sys.exit(1)

    tester = NewOSSModelTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
