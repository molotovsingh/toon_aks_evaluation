#!/usr/bin/env python3
"""
GPT-5 Availability Smoke Test

Tests if GPT-5 is accessible via OpenAI API and determines:
1. Correct model identifier
2. JSON mode support
3. Actual pricing/token usage
4. Available GPT-5 variants
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


def test_model_availability(client, model_id: str) -> dict:
    """
    Test if a specific model is available

    Args:
        client: OpenAI client instance
        model_id: Model identifier to test

    Returns:
        Dict with test results
    """
    print(f"\n{'='*60}")
    print(f"Testing: {model_id}")
    print(f"{'='*60}")

    try:
        # Minimal test message
        # GPT-5 has special requirements:
        # - Uses max_completion_tokens (not max_tokens)
        # - Requires temperature=1.0 (no other values allowed)
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "user", "content": "Say 'OK' if you can read this."}
                ],
                max_completion_tokens=10,
                temperature=1.0  # GPT-5 requires temperature=1.0
            )
        except Exception as e:
            # Fallback to old parameter name for GPT-4 models
            if 'max_tokens' in str(e) or 'max_completion_tokens' in str(e):
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "user", "content": "Say 'OK' if you can read this."}
                    ],
                    max_tokens=10,
                    temperature=0.0
                )
            else:
                raise

        # Extract details
        result = {
            "model_id": model_id,
            "status": "✅ AVAILABLE",
            "actual_model": response.model,
            "response": response.choices[0].message.content,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        print(f"✅ Status: AVAILABLE")
        print(f"📋 Actual model: {result['actual_model']}")
        print(f"💬 Response: {result['response']}")
        print(f"🔢 Tokens: {result['total_tokens']} (prompt: {result['prompt_tokens']}, completion: {result['completion_tokens']})")

        return result

    except Exception as e:
        error_msg = str(e)
        result = {
            "model_id": model_id,
            "status": "❌ UNAVAILABLE",
            "error": error_msg
        }

        print(f"❌ Status: UNAVAILABLE")
        print(f"🔴 Error: {error_msg}")

        return result


def test_json_mode(client, model_id: str) -> dict:
    """
    Test if model supports JSON mode

    Args:
        client: OpenAI client instance
        model_id: Model identifier to test

    Returns:
        Dict with test results
    """
    print(f"\n{'='*60}")
    print(f"Testing JSON Mode: {model_id}")
    print(f"{'='*60}")

    try:
        # Try GPT-5 parameters first
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Return all responses as valid JSON."},
                    {"role": "user", "content": 'Return a JSON object with one field: {"status": "ok"}'}
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=20,
                temperature=1.0  # GPT-5 requires temperature=1.0
            )
        except Exception as e:
            # Fallback to old parameter name and temperature for GPT-4
            if 'max_tokens' in str(e) or 'max_completion_tokens' in str(e) or 'temperature' in str(e):
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Return all responses as valid JSON."},
                        {"role": "user", "content": 'Return a JSON object with one field: {"status": "ok"}'}
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=20,
                    temperature=0.0
                )
            else:
                raise

        content = response.choices[0].message.content

        result = {
            "model_id": model_id,
            "json_mode": "✅ SUPPORTED",
            "response": content,
            "total_tokens": response.usage.total_tokens
        }

        print(f"✅ JSON Mode: SUPPORTED")
        print(f"📋 Response: {content}")

        return result

    except Exception as e:
        error_msg = str(e)
        result = {
            "model_id": model_id,
            "json_mode": "❌ NOT SUPPORTED",
            "error": error_msg
        }

        print(f"❌ JSON Mode: NOT SUPPORTED")
        print(f"🔴 Error: {error_msg}")

        return result


def list_available_models(client) -> list:
    """
    List all available models from OpenAI API

    Args:
        client: OpenAI client instance

    Returns:
        List of model IDs
    """
    print(f"\n{'='*60}")
    print(f"Listing Available Models")
    print(f"{'='*60}")

    try:
        models = client.models.list()
        model_ids = [model.id for model in models.data]

        # Filter for GPT-5 related models
        gpt5_models = [m for m in model_ids if 'gpt-5' in m.lower() or 'gpt5' in m.lower()]

        if gpt5_models:
            print(f"\n🎯 Found {len(gpt5_models)} GPT-5 related model(s):")
            for model in sorted(gpt5_models):
                print(f"   - {model}")
        else:
            print(f"\n⚠️ No GPT-5 related models found in API")
            print(f"📋 Total models available: {len(model_ids)}")
            print(f"\nMost recent GPT models:")
            gpt_models = [m for m in model_ids if m.startswith('gpt-')]
            for model in sorted(gpt_models)[-10:]:  # Show last 10
                print(f"   - {model}")

        return gpt5_models

    except Exception as e:
        print(f"❌ Failed to list models: {e}")
        return []


def main():
    """Run GPT-5 availability smoke tests"""
    print("\n" + "="*60)
    print("GPT-5 AVAILABILITY SMOKE TEST")
    print("="*60)

    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY not found in environment")
        print("   Set it in your .env file or export it:")
        print("   export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)

    print(f"✅ API Key found: {api_key[:8]}...{api_key[-4:]}")

    # Initialize OpenAI client
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        print(f"✅ OpenAI client initialized")
    except ImportError:
        print("❌ ERROR: openai library not installed")
        print("   Install with: uv pip install openai")
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: Failed to initialize OpenAI client: {e}")
        sys.exit(1)

    # Test candidate model identifiers
    candidate_models = [
        'gpt-5',                    # Our assumed identifier
        'gpt-5-turbo',              # Possible turbo variant
        'gpt-5-2025-08-07',         # Versioned identifier
        'gpt-5-preview',            # Preview version
    ]

    results = []

    # Step 1: Test each candidate model
    print("\n" + "="*60)
    print("STEP 1: Testing Candidate Model IDs")
    print("="*60)

    for model_id in candidate_models:
        result = test_model_availability(client, model_id)
        results.append(result)

    # Step 2: Test JSON mode on successful models
    print("\n" + "="*60)
    print("STEP 2: Testing JSON Mode Support")
    print("="*60)

    available_models = [r for r in results if r['status'] == '✅ AVAILABLE']

    if available_models:
        for result in available_models:
            json_result = test_json_mode(client, result['model_id'])
            result['json_mode_test'] = json_result
    else:
        print("⚠️ No models available to test JSON mode")

    # Step 3: List all available models
    gpt5_models = list_available_models(client)

    # Final Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    if available_models:
        print(f"\n✅ GPT-5 IS AVAILABLE!")
        print(f"\nWorking model identifier(s):")
        for result in available_models:
            print(f"   - {result['model_id']} → {result['actual_model']}")
            if 'json_mode_test' in result:
                json_status = result['json_mode_test'].get('json_mode', 'UNKNOWN')
                print(f"     JSON Mode: {json_status}")

        print(f"\n💡 RECOMMENDATION:")
        best_model = available_models[0]
        print(f"   Use: {best_model['model_id']}")
        print(f"   Actual model: {best_model['actual_model']}")

    else:
        print(f"\n❌ GPT-5 NOT AVAILABLE")
        print(f"\nTested identifiers:")
        for result in results:
            print(f"   - {result['model_id']}: {result['status']}")

        if gpt5_models:
            print(f"\n💡 Alternative GPT-5 models found:")
            for model in gpt5_models:
                print(f"   - {model}")
        else:
            print(f"\n⚠️ No GPT-5 models found in API")
            print(f"   GPT-5 may not be released yet or requires special access")

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
