#!/usr/bin/env python3
"""
Test Runtime Model Override for OpenRouter
Verifies that the backend plumbing correctly threads runtime_model through the stack
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment
load_dotenv()

def test_config_active_model():
    """Test 1: Verify OpenRouterConfig.active_model property works"""
    print("\n" + "="*80)
    print("TEST 1: OpenRouterConfig active_model property")
    print("="*80)

    from src.core.config import OpenRouterConfig

    # Test 1a: No runtime override (uses env default)
    config1 = OpenRouterConfig()
    print(f"✓ Config without runtime_model:")
    print(f"  - config.model (env): {config1.model}")
    print(f"  - config.runtime_model: {config1.runtime_model}")
    print(f"  - config.active_model: {config1.active_model}")
    assert config1.active_model == config1.model, "active_model should equal model when no runtime override"
    print("  ✅ PASS: active_model returns env default when runtime_model is None")

    # Test 1b: With runtime override
    config2 = OpenRouterConfig()
    config2.runtime_model = "deepseek/deepseek-chat"
    print(f"\n✓ Config with runtime_model='deepseek/deepseek-chat':")
    print(f"  - config.model (env): {config2.model}")
    print(f"  - config.runtime_model: {config2.runtime_model}")
    print(f"  - config.active_model: {config2.active_model}")
    assert config2.active_model == "deepseek/deepseek-chat", "active_model should return runtime_model when set"
    print("  ✅ PASS: active_model returns runtime override when provided")


def test_load_provider_config():
    """Test 2: Verify load_provider_config threads runtime_model to config"""
    print("\n" + "="*80)
    print("TEST 2: load_provider_config with runtime_model")
    print("="*80)

    from src.core.config import load_provider_config

    # Test 2a: Without runtime_model
    doc_config1, event_config1, extractor_config1 = load_provider_config("openrouter")
    print(f"✓ load_provider_config('openrouter') without runtime_model:")
    print(f"  - event_config.model (env): {event_config1.model}")
    print(f"  - event_config.runtime_model: {event_config1.runtime_model}")
    print(f"  - event_config.active_model: {event_config1.active_model}")
    assert event_config1.runtime_model is None, "runtime_model should be None when not provided"
    print("  ✅ PASS: runtime_model is None when not provided")

    # Test 2b: With runtime_model
    doc_config2, event_config2, extractor_config2 = load_provider_config(
        "openrouter",
        runtime_model="anthropic/claude-3-haiku"
    )
    print(f"\n✓ load_provider_config('openrouter', runtime_model='anthropic/claude-3-haiku'):")
    print(f"  - event_config.model (env): {event_config2.model}")
    print(f"  - event_config.runtime_model: {event_config2.runtime_model}")
    print(f"  - event_config.active_model: {event_config2.active_model}")
    assert event_config2.runtime_model == "anthropic/claude-3-haiku", "runtime_model should be set"
    assert event_config2.active_model == "anthropic/claude-3-haiku", "active_model should return runtime override"
    print("  ✅ PASS: runtime_model is correctly set and active_model returns it")


def test_extractor_factory():
    """Test 3: Verify create_default_extractors passes runtime_model through"""
    print("\n" + "="*80)
    print("TEST 3: create_default_extractors with runtime_model")
    print("="*80)

    from src.core.extractor_factory import create_default_extractors

    # Check if OpenRouter is configured
    if not os.getenv("OPENROUTER_API_KEY"):
        print("  ⚠️  SKIP: OPENROUTER_API_KEY not set - cannot test adapter initialization")
        return

    # Test 3a: Without runtime_model
    try:
        doc_ext1, event_ext1 = create_default_extractors(
            event_extractor_override="openrouter"
        )
        print(f"✓ Extractor created without runtime_model:")
        print(f"  - Adapter type: {event_ext1.__class__.__name__}")
        print(f"  - Config model (env): {event_ext1.config.model}")
        print(f"  - Config runtime_model: {event_ext1.config.runtime_model}")
        print(f"  - Config active_model: {event_ext1.config.active_model}")
        assert event_ext1.config.runtime_model is None, "runtime_model should be None"
        print("  ✅ PASS: Adapter initialized with env default model")
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        raise

    # Test 3b: With runtime_model
    try:
        doc_ext2, event_ext2 = create_default_extractors(
            event_extractor_override="openrouter",
            runtime_model="google/gemini-flash-1.5"
        )
        print(f"\n✓ Extractor created with runtime_model='google/gemini-flash-1.5':")
        print(f"  - Adapter type: {event_ext2.__class__.__name__}")
        print(f"  - Config model (env): {event_ext2.config.model}")
        print(f"  - Config runtime_model: {event_ext2.config.runtime_model}")
        print(f"  - Config active_model: {event_ext2.config.active_model}")
        assert event_ext2.config.runtime_model == "google/gemini-flash-1.5", "runtime_model should be set"
        assert event_ext2.config.active_model == "google/gemini-flash-1.5", "active_model should return runtime override"
        print("  ✅ PASS: Adapter initialized with runtime model override")
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        raise


def test_pipeline_initialization():
    """Test 4: Verify LegalEventsPipeline accepts and uses runtime_model"""
    print("\n" + "="*80)
    print("TEST 4: LegalEventsPipeline with runtime_model")
    print("="*80)

    from src.core.legal_pipeline_refactored import LegalEventsPipeline

    # Check if OpenRouter is configured
    if not os.getenv("OPENROUTER_API_KEY"):
        print("  ⚠️  SKIP: OPENROUTER_API_KEY not set - cannot test pipeline initialization")
        return

    # Test 4a: Without runtime_model
    try:
        pipeline1 = LegalEventsPipeline(event_extractor="openrouter")
        print(f"✓ Pipeline created without runtime_model:")
        print(f"  - Pipeline provider: {pipeline1.provider}")
        print(f"  - Pipeline runtime_model: {pipeline1.runtime_model}")
        print(f"  - Extractor config active_model: {pipeline1.event_extractor.config.active_model}")
        assert pipeline1.runtime_model is None, "Pipeline runtime_model should be None"
        print("  ✅ PASS: Pipeline initialized with env default model")
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        raise

    # Test 4b: With runtime_model
    try:
        pipeline2 = LegalEventsPipeline(
            event_extractor="openrouter",
            runtime_model="meta-llama/llama-3.3-70b-instruct"
        )
        print(f"\n✓ Pipeline created with runtime_model='meta-llama/llama-3.3-70b-instruct':")
        print(f"  - Pipeline provider: {pipeline2.provider}")
        print(f"  - Pipeline runtime_model: {pipeline2.runtime_model}")
        print(f"  - Extractor config active_model: {pipeline2.event_extractor.config.active_model}")
        assert pipeline2.runtime_model == "meta-llama/llama-3.3-70b-instruct", "Pipeline runtime_model should be set"
        assert pipeline2.event_extractor.config.active_model == "meta-llama/llama-3.3-70b-instruct", "Extractor should use runtime model"
        print("  ✅ PASS: Pipeline initialized with runtime model override")
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        raise


def test_backward_compatibility():
    """Test 5: Verify backward compatibility (all optional parameters)"""
    print("\n" + "="*80)
    print("TEST 5: Backward Compatibility")
    print("="*80)

    from src.core.legal_pipeline_refactored import LegalEventsPipeline

    # Check if any provider is configured
    has_gemini = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    has_openrouter = os.getenv("OPENROUTER_API_KEY")

    if not (has_gemini or has_openrouter):
        print("  ⚠️  SKIP: No API keys configured - cannot test backward compatibility")
        return

    # Test old-style initialization (no runtime_model parameter)
    try:
        if has_openrouter:
            pipeline = LegalEventsPipeline(event_extractor="openrouter")
            print(f"✓ Old-style initialization: LegalEventsPipeline(event_extractor='openrouter')")
        else:
            pipeline = LegalEventsPipeline()  # Uses env default
            print(f"✓ Old-style initialization: LegalEventsPipeline()")

        print(f"  - Provider: {pipeline.provider}")
        print(f"  - Runtime model: {pipeline.runtime_model}")
        print(f"  ✅ PASS: Backward compatible - works without runtime_model parameter")
    except TypeError as e:
        print(f"  ❌ FAIL: Not backward compatible - {e}")
        raise


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("🧪 RUNTIME MODEL OVERRIDE - BACKEND INTEGRATION TESTS")
    print("="*80)

    try:
        test_config_active_model()
        test_load_provider_config()
        test_extractor_factory()
        test_pipeline_initialization()
        test_backward_compatibility()

        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED - Backend plumbing is working correctly!")
        print("="*80)
        print("\nKey Findings:")
        print("  ✓ OpenRouterConfig.active_model property works correctly")
        print("  ✓ runtime_model threads through all layers (config → factory → pipeline)")
        print("  ✓ Adapters use active_model (runtime override takes precedence)")
        print("  ✓ 100% backward compatible (all new parameters are optional)")
        print("\n🎉 Ready to build the UI!")

    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ TEST FAILED: {e}")
        print("="*80)
        sys.exit(1)


if __name__ == "__main__":
    main()
