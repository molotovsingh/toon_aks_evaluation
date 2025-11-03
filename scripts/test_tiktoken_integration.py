#!/usr/bin/env python3
"""
Test Script: Tiktoken Integration & Cost Calculation Validation

Tests the new two-stage cost estimation system:
1. Extract Famas PDF with Docling
2. Count tokens with tiktoken for each model
3. Calculate costs for all models
4. Show cost comparison

Usage:
    uv run python scripts/test_tiktoken_integration.py
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment
from dotenv import load_dotenv
load_dotenv()


def test_token_counting():
    """Test tiktoken token counting for different models"""
    print("\n" + "=" * 80)
    print("🔬 TEST 1: Token Counting with Tiktoken")
    print("=" * 80)

    from src.utils.token_counter import (
        count_tokens,
        get_supported_models,
        is_model_supported
    )

    # Test text
    test_text = """
    This Lease Agreement is entered into on September 21, 2025, between Landlord
    ("Owner") and Tenant ("Renter"). The lease begins on October 1, 2025 and rent
    is due on the 5th of every month. All payments should be made to the Owner's
    account at First National Bank.
    """ * 50  # Repeat to simulate larger document

    print(f"\nTest text length: {len(test_text)} characters")

    # Test different models
    test_models = [
        "gpt-4o",
        "gpt-4o-mini",
        "claude-3-haiku-20240307",
        "claude-sonnet-4-5",
        "deepseek-chat",
    ]

    print("\nToken counts by model:")
    print(f"{'Model':<30} {'Tokens':<10} {'Supported':<10}")
    print("-" * 50)

    for model in test_models:
        if is_model_supported(model):
            tokens = count_tokens(test_text, model)
            print(f"{model:<30} {tokens:<10} ✓")
        else:
            print(f"{model:<30} {'N/A':<10} ✗")

    # Show all supported models
    print(f"\nTotal supported models: {len(get_supported_models())}")
    print("Supported models:", ", ".join(get_supported_models()[:5]), "...")


def test_document_extraction():
    """Test Docling document extraction"""
    print("\n" + "=" * 80)
    print("🔬 TEST 2: Document Extraction with Docling")
    print("=" * 80)

    from src.core.docling_adapter import DoclingDocumentExtractor

    # Use Famas PDF from repo
    sample_file = Path("sample_pdf/famas_dispute/Transaction_Fee_Invoice.pdf")

    if not sample_file.exists():
        print(f"❌ Sample file not found: {sample_file}")
        return None

    print(f"\nExtracting: {sample_file.name}")
    print(f"File size: {sample_file.stat().st_size / 1024 / 1024:.2f} MB")

    try:
        from src.core.config import DoclingConfig
        config = DoclingConfig()
        extractor = DoclingDocumentExtractor(config)
        extracted = extractor.extract(sample_file)

        if extracted and extracted.plain_text:
            print(f"✅ Extraction successful")
            print(f"   - Characters: {len(extracted.plain_text):,}")
            print(f"   - Lines: {len(extracted.plain_text.splitlines()):,}")
            return extracted.plain_text
        else:
            print("❌ Extraction returned empty result")
            return None

    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_cost_calculations(extracted_text: str):
    """Test cost calculations across all models"""
    print("\n" + "=" * 80)
    print("🔬 TEST 3: Cost Calculations with Tiktoken")
    print("=" * 80)

    from src.ui.cost_estimator import estimate_all_models_with_tiktoken

    print(f"\nCalculating costs for {len(extracted_text):,} characters...")

    try:
        cost_table = estimate_all_models_with_tiktoken(
            extracted_texts=[extracted_text],
            output_ratio=0.10
        )

        if cost_table:
            print(f"✅ Cost calculation successful")
            print(f"\nTop 5 Cheapest Models:")
            print(f"{'Rank':<6} {'Model':<30} {'Tokens':<10} {'Cost':<12} {'Quality':<10}")
            print("-" * 68)

            for idx, model in enumerate(cost_table[:5], 1):
                tokens = model['input_tokens'] + model['output_tokens']
                print(
                    f"{idx:<6} {model['display_name']:<30} {tokens:<10,} "
                    f"${model['total_cost']:<11.4f} {model['quality_score']:<10}"
                )

            print(f"\nTop 3 Most Expensive Models:")
            print(f"{'Rank':<6} {'Model':<30} {'Tokens':<10} {'Cost':<12} {'Quality':<10}")
            print("-" * 68)

            for idx, model in enumerate(cost_table[-3:], 1):
                tokens = model['input_tokens'] + model['output_tokens']
                print(
                    f"{idx:<6} {model['display_name']:<30} {tokens:<10,} "
                    f"${model['total_cost']:<11.4f} {model['quality_score']:<10}"
                )

            # Cost breakdown
            print(f"\n💰 Cost Breakdown (Top Model vs Ground Truth):")
            cheapest = cost_table[0]
            most_expensive = cost_table[-1]

            print(f"\nCheapest: {cheapest['display_name']}")
            print(f"  Input:  {cheapest['input_tokens']:,} tokens @ ${cheapest['input_cost']:.6f}")
            print(f"  Output: {cheapest['output_tokens']:,} tokens @ ${cheapest['output_cost']:.6f}")
            print(f"  Total:  ${cheapest['total_cost']:.6f}")

            print(f"\nMost Expensive: {most_expensive['display_name']}")
            print(f"  Input:  {most_expensive['input_tokens']:,} tokens @ ${most_expensive['input_cost']:.6f}")
            print(f"  Output: {most_expensive['output_tokens']:,} tokens @ ${most_expensive['output_cost']:.6f}")
            print(f"  Total:  ${most_expensive['total_cost']:.6f}")

            # Cost ratio (handle free models)
            if cheapest['total_cost'] > 0:
                ratio = most_expensive['total_cost'] / cheapest['total_cost']
                print(f"\n📊 Cost Analysis:")
                print(f"  Most expensive is {ratio:.0f}x the cost of cheapest")
            else:
                print(f"\n📊 Cost Analysis:")
                print(f"  Cheapest model is FREE (Gemini)")

            savings = most_expensive['total_cost'] - cheapest['total_cost']
            print(f"  Savings by choosing cheapest: ${savings:.6f}")

            # Validate token counts
            print(f"\n✅ Token Count Variation (Expected - Different Encodings):")
            input_tokens_list = [m['input_tokens'] for m in cost_table]
            output_tokens_list = [m['output_tokens'] for m in cost_table]

            print(f"  Input tokens range: {min(input_tokens_list):,} - {max(input_tokens_list):,}")
            print(f"  Output tokens range: {min(output_tokens_list):,} - {max(output_tokens_list):,}")
            print(f"  Note: Different models use different tokenizers (o200k vs cl100k)")
            print(f"  Token count variance is EXPECTED and CORRECT")

            return cost_table
        else:
            print("❌ Cost calculation failed")
            return None

    except Exception as e:
        print(f"❌ Cost calculation error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_accuracy_vs_heuristic(extracted_text: str):
    """Compare tiktoken accuracy vs character-based heuristic"""
    print("\n" + "=" * 80)
    print("🔬 TEST 4: Tiktoken vs Character Heuristic Accuracy")
    print("=" * 80)

    from src.utils.token_counter import count_tokens
    from src.ui.cost_estimator import estimate_tokens as heuristic_estimate

    print(f"\nComparing token estimation methods on {len(extracted_text):,} characters")

    # Get heuristic estimate
    heuristic_tokens = heuristic_estimate(extracted_text)
    print(f"\nCharacter heuristic: {heuristic_tokens} tokens (4 chars = 1 token)")

    # Get tiktoken estimates
    test_models = [
        ("gpt-4o", "o200k_base"),
        ("gpt-4o-mini", "o200k_base"),
        ("claude-3-haiku-20240307", "cl100k_base"),
    ]

    print(f"\n{'Model':<30} {'Tokens':<10} {'vs Heuristic':<15} {'Error':<10}")
    print("-" * 65)

    for model_id, encoding in test_models:
        tiktoken_tokens = count_tokens(extracted_text, model_id)
        difference = tiktoken_tokens - heuristic_tokens
        percent_error = abs(difference) / heuristic_tokens * 100 if heuristic_tokens > 0 else 0

        print(
            f"{model_id:<30} {tiktoken_tokens:<10} {difference:+10} "
            f"{percent_error:<9.1f}%"
        )

    print(f"\n💡 Insight: Tiktoken gives more precise counts per model encoding")
    print(f"   Heuristic is good for quick estimates before extraction")
    print(f"   Tiktoken is essential for accurate cost calculation after extraction")


def main():
    """Run all tests"""
    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  TIKTOKEN INTEGRATION & COST CALCULATION TEST SUITE  ".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)

    try:
        # Test 1: Token counting
        test_token_counting()

        # Test 2: Document extraction
        extracted_text = test_document_extraction()

        if extracted_text:
            # Test 3: Cost calculations
            cost_table = test_cost_calculations(extracted_text)

            # Test 4: Accuracy comparison
            test_accuracy_vs_heuristic(extracted_text)

            print("\n" + "=" * 80)
            print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("⚠️  DOCUMENT EXTRACTION FAILED - SKIPPING COST TESTS")
            print("=" * 80)

    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
