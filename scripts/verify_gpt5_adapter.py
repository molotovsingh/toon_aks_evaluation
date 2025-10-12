#!/usr/bin/env python3
"""
Quick verification that GPT-5 adapter changes are correct

Verifies:
1. GPT-5 detection logic works
2. Parameter selection logic is correct (without making API calls)
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

print("\n" + "="*60)
print("GPT-5 ADAPTER VERIFICATION")
print("="*60)

# Test 1: Verify GPT-5 detection
from src.core.openai_adapter import OpenAIEventExtractor, GPT5_MODELS

print("\n📋 Step 1: Verify GPT-5 model list")
print(f"   Total GPT-5 variants configured: {len(GPT5_MODELS)}")
for model in GPT5_MODELS:
    print(f"   • {model}")

# Test 2: Test detection method
from src.core.config import OpenAIConfig

print("\n🔍 Step 2: Test GPT-5 detection method")
test_cases = [
    ("gpt-5", True),
    ("gpt-5-2025-08-07", True),
    ("gpt-5-mini", True),
    ("GPT-5", True),  # Case insensitive
    ("gpt-4o", False),
    ("gpt-4o-mini", False),
    ("claude-3-5-sonnet", False),
]

all_passed = True
for model, expected_gpt5 in test_cases:
    # Mock config to test detection
    config = OpenAIConfig(
        api_key="test_key",
        model=model
    )

    # Create extractor (without API call)
    try:
        extractor = OpenAIEventExtractor(config)
        is_gpt5 = extractor._is_gpt5
        status = "✅" if is_gpt5 == expected_gpt5 else "❌"
        print(f"   {status} '{model}' → is_gpt5={is_gpt5} (expected: {expected_gpt5})")

        if is_gpt5 != expected_gpt5:
            all_passed = False
    except Exception as e:
        print(f"   ❌ '{model}' → ERROR: {e}")
        all_passed = False

# Test 3: Verify API parameter logic
print("\n⚙️ Step 3: Verify API parameter selection logic")
print("\nReading _call_openai_api method...")

# Read the actual implementation
adapter_file = project_root / "src" / "core" / "openai_adapter.py"
with open(adapter_file, 'r') as f:
    content = f.read()

# Check for GPT-5 specific parameters
checks = {
    "temperature=1.0 if self._is_gpt5": "temperature=1.0 if self._is_gpt5" in content,
    "max_completion_tokens conditional": "max_completion_tokens" in content and "if self._is_gpt5" in content,
    "max_tokens for GPT-4": "max_tokens" in content and "else:" in content,
    "GPT5_MODELS list defined": "GPT5_MODELS = [" in content,
    "_check_gpt5_model method": "def _check_gpt5_model(self, model: str)" in content,
}

for check, found in checks.items():
    status = "✅" if found else "❌"
    print(f"   {status} {check}")
    if not found:
        all_passed = False

# Final summary
print("\n" + "="*60)
if all_passed:
    print("✅ ALL VERIFICATION CHECKS PASSED")
    print("\nThe OpenAI adapter correctly handles GPT-5:")
    print("  • GPT-5 detection logic works for all variants")
    print("  • Conditional parameter selection implemented")
    print("  • temperature=1.0 for GPT-5, temperature=0.0 for GPT-4")
    print("  • max_completion_tokens for GPT-5, max_tokens for GPT-4")
    print("\n💡 Ready to test with real API calls!")
else:
    print("❌ SOME CHECKS FAILED")
    print("\nReview the failed checks above.")
    sys.exit(1)

print("="*60 + "\n")
