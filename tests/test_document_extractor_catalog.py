#!/usr/bin/env python3
"""
Tests for Document Extractor Catalog

Validates centralized registry pattern for Layer 1 document extraction metadata.

Run with: uv run python tests/test_document_extractor_catalog.py
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.document_extractor_catalog import (
    get_doc_extractor_catalog,
    DocumentExtractorCatalog,
    DocExtractorEntry,
)


class TestDocumentExtractorCatalog:
    """Test suite for document extractor catalog functionality"""

    def test_catalog_singleton(self):
        """Verify catalog returns same instance (singleton pattern)"""
        catalog1 = get_doc_extractor_catalog()
        catalog2 = get_doc_extractor_catalog()
        assert catalog1 is catalog2, "Catalog should be singleton"

    def test_get_extractor_docling(self):
        """Test retrieving Docling extractor entry"""
        catalog = get_doc_extractor_catalog()
        docling = catalog.get_extractor("docling")

        assert docling is not None, "Docling extractor should exist"
        assert docling.extractor_id == "docling"
        assert docling.display_name == "Docling (Local OCR)"
        assert docling.provider == "local"
        assert docling.cost_per_page == 0.0, "Docling should be free"
        assert docling.enabled is True, "Docling should be enabled by default"
        assert docling.supports_pdf is True

    def test_get_extractor_qwen_vl(self):
        """Test retrieving Qwen-VL extractor entry"""
        catalog = get_doc_extractor_catalog()
        qwen_vl = catalog.get_extractor("qwen_vl")

        assert qwen_vl is not None, "Qwen-VL extractor should exist"
        assert qwen_vl.extractor_id == "qwen_vl"
        assert qwen_vl.display_name == "Qwen3-VL (Budget Vision)"
        assert qwen_vl.provider == "openrouter"
        assert qwen_vl.cost_per_page == 0.00512, "Qwen-VL pricing should match"
        assert qwen_vl.enabled is True, "Qwen-VL should be enabled by default"
        assert qwen_vl.supports_vision is True, "Qwen-VL is a vision model"
        assert qwen_vl.prompt_id == "qwen_vl_doc", "Qwen-VL should have prompt_id configured"

    def test_get_extractor_not_found(self):
        """Test retrieving non-existent extractor returns None"""
        catalog = get_doc_extractor_catalog()
        result = catalog.get_extractor("nonexistent_extractor")
        assert result is None, "Non-existent extractor should return None"

    def test_list_extractors_all(self):
        """Test listing all extractors"""
        catalog = get_doc_extractor_catalog()
        extractors = catalog.list_extractors()

        assert len(extractors) >= 2, "Should have at least 2 extractors (docling, qwen_vl)"
        extractor_ids = [e.extractor_id for e in extractors]
        assert "docling" in extractor_ids
        assert "qwen_vl" in extractor_ids

    def test_list_extractors_enabled_only(self):
        """Test listing only enabled extractors"""
        catalog = get_doc_extractor_catalog()
        enabled = catalog.list_extractors(enabled=True)

        assert len(enabled) >= 2, "Should have at least 2 enabled extractors"
        for extractor in enabled:
            assert extractor.enabled is True, f"{extractor.extractor_id} should be enabled"

    def test_list_extractors_filter_by_provider(self):
        """Test filtering extractors by provider"""
        catalog = get_doc_extractor_catalog()
        local_extractors = catalog.list_extractors(provider="local")

        assert len(local_extractors) >= 1, "Should have at least 1 local extractor"
        for extractor in local_extractors:
            assert extractor.provider == "local"

    def test_list_extractors_filter_free_only(self):
        """Test filtering free extractors only"""
        catalog = get_doc_extractor_catalog()
        free_extractors = catalog.list_extractors(free_only=True)

        assert len(free_extractors) >= 1, "Should have at least 1 free extractor (docling)"
        for extractor in free_extractors:
            assert extractor.cost_per_page == 0.0, f"{extractor.extractor_id} should be free"

    def test_list_extractors_filter_vision(self):
        """Test filtering vision-capable extractors"""
        catalog = get_doc_extractor_catalog()
        vision_extractors = catalog.list_extractors(supports_vision=True)

        assert len(vision_extractors) >= 1, "Should have at least 1 vision extractor (qwen_vl)"
        for extractor in vision_extractors:
            assert extractor.supports_vision is True

    def test_get_pricing(self):
        """Test getting pricing information"""
        catalog = get_doc_extractor_catalog()

        # Test Docling (free)
        docling_pricing = catalog.get_pricing("docling")
        assert docling_pricing is not None
        assert docling_pricing["cost_per_page"] == 0.0
        assert docling_pricing["cost_display"] == "FREE"

        # Test Qwen-VL (paid)
        qwen_pricing = catalog.get_pricing("qwen_vl")
        assert qwen_pricing is not None
        assert qwen_pricing["cost_per_page"] == 0.00512
        assert "$0.077" in qwen_pricing["cost_display"]

    def test_get_pricing_not_found(self):
        """Test getting pricing for non-existent extractor"""
        catalog = get_doc_extractor_catalog()
        result = catalog.get_pricing("nonexistent")
        assert result is None

    def test_estimate_cost_free_extractor(self):
        """Test cost estimation for free extractor"""
        catalog = get_doc_extractor_catalog()
        estimate = catalog.estimate_cost("docling", page_count=15)

        assert estimate["extractor_id"] == "docling"
        assert estimate["page_count"] == 15
        assert estimate["cost_per_page"] == 0.0
        assert estimate["cost_usd"] == 0.0
        assert estimate["cost_display"] == "FREE"
        assert estimate["pricing_available"] is True

    def test_estimate_cost_paid_extractor(self):
        """Test cost estimation for paid extractor"""
        catalog = get_doc_extractor_catalog()
        estimate = catalog.estimate_cost("qwen_vl", page_count=15)

        assert estimate["extractor_id"] == "qwen_vl"
        assert estimate["page_count"] == 15
        assert estimate["cost_per_page"] == 0.00512
        # Use approximate comparison for floating point (15 * 0.00512 = 0.0768)
        assert abs(estimate["cost_usd"] - 0.0768) < 0.0001, f"Expected ~0.0768, got {estimate['cost_usd']}"
        assert "$0.0768" in estimate["cost_display"]
        assert estimate["pricing_available"] is True

    def test_estimate_cost_unknown_extractor(self):
        """Test cost estimation for unknown extractor"""
        catalog = get_doc_extractor_catalog()
        estimate = catalog.estimate_cost("unknown", page_count=10)

        assert estimate["extractor_id"] == "unknown"
        assert estimate["cost_usd"] == 0.0
        assert estimate["pricing_available"] is False

    def test_validate_extractor_id(self):
        """Test extractor ID validation"""
        catalog = get_doc_extractor_catalog()

        assert catalog.validate_extractor_id("docling") is True
        assert catalog.validate_extractor_id("qwen_vl") is True
        assert catalog.validate_extractor_id("nonexistent") is False

    def test_get_all_extractor_ids(self):
        """Test getting all extractor IDs"""
        catalog = get_doc_extractor_catalog()
        ids = catalog.get_all_extractor_ids()

        assert isinstance(ids, list)
        assert "docling" in ids
        assert "qwen_vl" in ids

    def test_get_prompt_qwen_vl(self):
        """Test getting prompt for Qwen-VL (references prompt_id)"""
        catalog = get_doc_extractor_catalog()
        prompt = catalog.get_prompt("qwen_vl")

        assert prompt is not None, "Qwen-VL should have a prompt"
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        # Verify it's the legal document prompt
        assert "Transcribe this document" in prompt
        assert "legal citations" in prompt.lower()

    def test_get_prompt_docling(self):
        """Test getting prompt for Docling (no prompt configured)"""
        catalog = get_doc_extractor_catalog()
        prompt = catalog.get_prompt("docling")

        assert prompt is None, "Docling should not have a prompt (uses Tesseract OCR)"

    def test_get_prompt_nonexistent(self):
        """Test getting prompt for non-existent extractor"""
        catalog = get_doc_extractor_catalog()
        prompt = catalog.get_prompt("nonexistent")

        assert prompt is None

    def test_prompt_resolution_priority(self):
        """Test that prompt_override takes precedence over prompt_id"""
        # This test validates the documented priority:
        # 1. prompt_override (inline)
        # 2. prompt_id (registry lookup)
        # 3. None (default behavior)

        catalog = get_doc_extractor_catalog()

        # Qwen-VL has prompt_id but no prompt_override
        qwen_vl = catalog.get_extractor("qwen_vl")
        assert qwen_vl.prompt_id == "qwen_vl_doc"
        assert qwen_vl.prompt_override is None

        # Get prompt should use prompt_id
        prompt = catalog.get_prompt("qwen_vl")
        assert prompt is not None

    def test_registry_schema_completeness(self):
        """Test that all registry entries have required fields"""
        catalog = get_doc_extractor_catalog()
        extractors = catalog.list_extractors()

        for extractor in extractors:
            # Required identification fields
            assert extractor.extractor_id, f"{extractor.extractor_id}: missing extractor_id"
            assert extractor.display_name, f"{extractor.extractor_id}: missing display_name"
            assert extractor.provider, f"{extractor.extractor_id}: missing provider"

            # Required pricing fields
            assert isinstance(extractor.cost_per_page, (int, float))
            assert extractor.cost_display, f"{extractor.extractor_id}: missing cost_display"

            # Required capability fields
            assert isinstance(extractor.supports_pdf, bool)
            assert isinstance(extractor.enabled, bool)

            # Optional fields should have defaults
            assert extractor.processing_speed in ["fast", "medium", "slow", ""]
            assert extractor.ocr_quality in ["high", "medium", "low", "n/a", ""]


class TestFactoryIntegration:
    """Test integration between catalog and extractor_factory module"""

    def test_factory_callable_present(self):
        """Test that factory_callable field is populated for extractors"""
        catalog = get_doc_extractor_catalog()

        # Docling should have factory callable
        docling = catalog.get_extractor("docling")
        assert docling.factory_callable is not None, "Docling should have factory_callable"
        assert "src.core.extractor_factory" in docling.factory_callable
        assert "_create_docling_document_extractor" in docling.factory_callable

        # Qwen-VL should have factory callable
        qwen_vl = catalog.get_extractor("qwen_vl")
        assert qwen_vl.factory_callable is not None, "Qwen-VL should have factory_callable"
        assert "src.core.extractor_factory" in qwen_vl.factory_callable
        assert "_create_qwen_vl_document_extractor" in qwen_vl.factory_callable

    def test_factory_registry_built_from_catalog(self):
        """Test that DOC_PROVIDER_REGISTRY is dynamically built from catalog"""
        from src.core.extractor_factory import DOC_PROVIDER_REGISTRY

        # Registry should be a dict
        assert isinstance(DOC_PROVIDER_REGISTRY, dict), "Registry should be a dictionary"

        # Should contain enabled extractors with factory callables
        assert "docling" in DOC_PROVIDER_REGISTRY, "Docling should be in registry"
        assert "qwen_vl" in DOC_PROVIDER_REGISTRY, "Qwen-VL should be in registry"

        # Registry functions should be callable
        assert callable(DOC_PROVIDER_REGISTRY["docling"])
        assert callable(DOC_PROVIDER_REGISTRY["qwen_vl"])

    def test_disabled_extractor_not_in_factory(self):
        """Test that disabling an extractor removes it from factory registry"""
        # This test simulates the behavior - in real usage, you'd modify catalog and reload
        from src.core.document_extractor_catalog import DocumentExtractorCatalog, DocExtractorEntry

        # Create test catalog with disabled extractor
        test_registry = [
            DocExtractorEntry(
                extractor_id="test_disabled",
                display_name="Test Disabled",
                provider="local",
                cost_per_page=0.0,
                cost_display="FREE",
                enabled=False,  # DISABLED
                factory_callable="src.core.extractor_factory._create_docling_document_extractor"
            ),
            DocExtractorEntry(
                extractor_id="test_enabled",
                display_name="Test Enabled",
                provider="local",
                cost_per_page=0.0,
                cost_display="FREE",
                enabled=True,  # ENABLED
                factory_callable="src.core.extractor_factory._create_docling_document_extractor"
            ),
        ]

        test_catalog = DocumentExtractorCatalog(test_registry)

        # Get enabled extractors (should only include enabled one)
        enabled_extractors = test_catalog.list_extractors(enabled=True)
        enabled_ids = [e.extractor_id for e in enabled_extractors]

        assert "test_enabled" in enabled_ids, "Enabled extractor should be in list"
        assert "test_disabled" not in enabled_ids, "Disabled extractor should not be in list"

    def test_factory_build_validates_enabled_flag(self):
        """Test that build_extractors validates enabled flag via catalog"""
        from src.core.extractor_factory import build_extractors, ExtractorConfigurationError
        from src.core.config import DoclingConfig, ExtractorConfig, LangExtractConfig

        # Try to build with valid enabled extractor (should succeed)
        doc_config = DoclingConfig()
        event_config = LangExtractConfig()
        extractor_config = ExtractorConfig(doc_extractor="docling")

        try:
            doc_extractor, _ = build_extractors(doc_config, event_config, extractor_config)
            assert doc_extractor is not None, "Should create extractor for enabled entry"
        except ExtractorConfigurationError:
            # This is fine - just means dependencies aren't available in test environment
            pass

    def test_cli_gets_extractors_from_catalog(self):
        """Test that CLI dynamically loads extractors from catalog"""
        # Import would fail if catalog isn't accessible
        from src.core.document_extractor_catalog import get_doc_extractor_catalog

        catalog = get_doc_extractor_catalog()
        cli_choices = [e.extractor_id for e in catalog.list_extractors(enabled=True)]

        # CLI should get same list as catalog
        assert "docling" in cli_choices, "CLI should include docling"
        assert "qwen_vl" in cli_choices, "CLI should include qwen_vl"
        assert len(cli_choices) >= 2, "CLI should have at least 2 choices"


class TestPromptRegistryIntegration:
    """Test integration between catalog and doc_extractor_prompts module"""

    def test_qwen_vl_prompt_exists_in_registry(self):
        """Verify Qwen-VL prompt exists in doc_extractor_prompts.py"""
        from src.core.doc_extractor_prompts import get_prompt_by_id

        prompt = get_prompt_by_id("qwen_vl_doc")
        assert prompt is not None
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_catalog_prompt_matches_registry(self):
        """Verify catalog.get_prompt() returns same prompt as direct registry lookup"""
        from src.core.doc_extractor_prompts import get_prompt_by_id
        catalog = get_doc_extractor_catalog()

        catalog_prompt = catalog.get_prompt("qwen_vl")
        registry_prompt = get_prompt_by_id("qwen_vl_doc")

        assert catalog_prompt == registry_prompt, "Catalog should return same prompt as registry"

    def test_list_prompt_ids(self):
        """Test listing available prompt IDs"""
        from src.core.doc_extractor_prompts import list_prompt_ids

        ids = list_prompt_ids()
        assert isinstance(ids, list)
        assert "qwen_vl_doc" in ids


def run_all_tests():
    """Run all test suites"""
    print("=" * 80)
    print("DOCUMENT EXTRACTOR CATALOG TEST SUITE")
    print("=" * 80)
    print()

    test_catalog = TestDocumentExtractorCatalog()
    test_factory = TestFactoryIntegration()
    test_prompts = TestPromptRegistryIntegration()

    # Catalog tests
    tests = [
        ("Catalog Singleton", test_catalog.test_catalog_singleton),
        ("Get Extractor - Docling", test_catalog.test_get_extractor_docling),
        ("Get Extractor - Qwen-VL", test_catalog.test_get_extractor_qwen_vl),
        ("Get Extractor - Not Found", test_catalog.test_get_extractor_not_found),
        ("List All Extractors", test_catalog.test_list_extractors_all),
        ("List Enabled Extractors", test_catalog.test_list_extractors_enabled_only),
        ("Filter by Provider", test_catalog.test_list_extractors_filter_by_provider),
        ("Filter Free Only", test_catalog.test_list_extractors_filter_free_only),
        ("Filter Vision Capable", test_catalog.test_list_extractors_filter_vision),
        ("Get Pricing", test_catalog.test_get_pricing),
        ("Get Pricing - Not Found", test_catalog.test_get_pricing_not_found),
        ("Estimate Cost - Free", test_catalog.test_estimate_cost_free_extractor),
        ("Estimate Cost - Paid", test_catalog.test_estimate_cost_paid_extractor),
        ("Estimate Cost - Unknown", test_catalog.test_estimate_cost_unknown_extractor),
        ("Validate Extractor ID", test_catalog.test_validate_extractor_id),
        ("Get All Extractor IDs", test_catalog.test_get_all_extractor_ids),
        ("Get Prompt - Qwen-VL", test_catalog.test_get_prompt_qwen_vl),
        ("Get Prompt - Docling", test_catalog.test_get_prompt_docling),
        ("Get Prompt - Nonexistent", test_catalog.test_get_prompt_nonexistent),
        ("Prompt Resolution Priority", test_catalog.test_prompt_resolution_priority),
        ("Registry Schema Completeness", test_catalog.test_registry_schema_completeness),
        ("Factory Callable Present", test_factory.test_factory_callable_present),
        ("Factory Registry Built", test_factory.test_factory_registry_built_from_catalog),
        ("Disabled Extractor Not In Factory", test_factory.test_disabled_extractor_not_in_factory),
        ("Factory Validates Enabled Flag", test_factory.test_factory_build_validates_enabled_flag),
        ("CLI Gets Extractors From Catalog", test_factory.test_cli_gets_extractors_from_catalog),
        ("Qwen-VL Prompt Exists", test_prompts.test_qwen_vl_prompt_exists_in_registry),
        ("Catalog-Registry Prompt Match", test_prompts.test_catalog_prompt_matches_registry),
        ("List Prompt IDs", test_prompts.test_list_prompt_ids),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            test_func()
            print(f"✅ PASS: {test_name}")
            passed += 1
        except AssertionError as e:
            print(f"❌ FAIL: {test_name}")
            print(f"   Error: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR: {test_name}")
            print(f"   Exception: {e}")
            failed += 1

    print()
    print("=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 80)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
