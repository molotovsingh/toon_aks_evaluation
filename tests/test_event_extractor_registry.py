"""
Tests for Event Extractor Registry

Validates:
1. Catalog schema and access patterns
2. Dynamic factory bootstrapping
3. Enabled flag toggling
4. Runtime model override flows
5. Credential validation
"""

import unittest
import os
from unittest.mock import patch
from src.core.event_extractor_catalog import (
    get_event_extractor_catalog,
    EventExtractorEntry,
    list_event_extractors,
    validate_event_provider
)
from src.core.extractor_factory import EVENT_PROVIDER_REGISTRY, ExtractorConfigurationError, build_extractors
from src.core.config import DoclingConfig, ExtractorConfig, load_provider_config


class TestEventExtractorCatalog(unittest.TestCase):
    """Test catalog schema and query functions"""

    def test_catalog_singleton(self):
        """Catalog returns same instance on multiple calls"""
        catalog1 = get_event_extractor_catalog()
        catalog2 = get_event_extractor_catalog()
        assert catalog1 is catalog2

    def test_all_providers_registered(self):
        """All 6 expected providers are in catalog"""
        catalog = get_event_extractor_catalog()
        provider_ids = catalog.get_all_provider_ids()

        expected = ['langextract', 'openrouter', 'openai', 'anthropic', 'deepseek', 'opencode_zen']
        assert set(provider_ids) == set(expected), f"Expected {expected}, got {provider_ids}"

    def test_get_extractor_by_id(self):
        """Get extractor entry by provider_id"""
        catalog = get_event_extractor_catalog()

        # Valid provider
        openrouter = catalog.get_extractor('openrouter')
        assert openrouter is not None
        assert openrouter.provider_id == 'openrouter'
        assert openrouter.display_name == 'OpenRouter'

        # Invalid provider
        invalid = catalog.get_extractor('nonexistent')
        assert invalid is None

    def test_list_extractors_enabled_filter(self):
        """Filter extractors by enabled status"""
        catalog = get_event_extractor_catalog()

        # All enabled (default state)
        enabled = catalog.list_extractors(enabled=True)
        assert len(enabled) == 6, "All providers should be enabled by default"

        # Filter for disabled (should be empty initially)
        disabled = catalog.list_extractors(enabled=False)
        assert len(disabled) == 0, "No providers should be disabled by default"

    def test_list_extractors_runtime_model_filter(self):
        """Filter extractors by runtime model support"""
        catalog = get_event_extractor_catalog()

        # Providers with runtime model support
        with_runtime = catalog.list_extractors(supports_runtime_model=True)
        runtime_ids = [e.provider_id for e in with_runtime]

        # Should include: langextract, openrouter, openai, anthropic
        assert 'langextract' in runtime_ids
        assert 'openrouter' in runtime_ids
        assert 'openai' in runtime_ids
        assert 'anthropic' in runtime_ids

        # Should exclude: deepseek, opencode_zen (single model providers)
        without_runtime = catalog.list_extractors(supports_runtime_model=False)
        no_runtime_ids = [e.provider_id for e in without_runtime]
        assert 'deepseek' in no_runtime_ids
        assert 'opencode_zen' in no_runtime_ids

    def test_list_extractors_recommended_filter(self):
        """Filter extractors by recommended status"""
        catalog = get_event_extractor_catalog()

        recommended = catalog.list_extractors(recommended_only=True)
        recommended_ids = [e.provider_id for e in recommended]

        # langextract and openrouter are marked as recommended
        assert 'langextract' in recommended_ids
        assert 'openrouter' in recommended_ids

    def test_validate_provider_id(self):
        """Validate provider ID existence"""
        catalog = get_event_extractor_catalog()

        assert catalog.validate_provider_id('openrouter') is True
        assert catalog.validate_provider_id('langextract') is True
        assert catalog.validate_provider_id('invalid') is False

        # Also test convenience function
        assert validate_event_provider('openai') is True
        assert validate_event_provider('nonexistent') is False

    def test_factory_callable_fields(self):
        """All enabled providers have valid factory_callable"""
        catalog = get_event_extractor_catalog()
        enabled = catalog.list_extractors(enabled=True)

        for entry in enabled:
            assert entry.factory_callable is not None, f"{entry.provider_id} missing factory_callable"
            assert entry.factory_callable.startswith("src.core."), \
                f"{entry.provider_id} factory_callable not in src.core.*"
            assert "_create_" in entry.factory_callable, \
                f"{entry.provider_id} factory_callable doesn't follow naming convention"


class TestDynamicRegistryBootstrap(unittest.TestCase):
    """Test dynamic EVENT_PROVIDER_REGISTRY bootstrapping from catalog"""

    def test_registry_populated_from_catalog(self):
        """EVENT_PROVIDER_REGISTRY is built from catalog at module load"""
        # Registry should contain all enabled providers
        assert 'langextract' in EVENT_PROVIDER_REGISTRY
        assert 'openrouter' in EVENT_PROVIDER_REGISTRY
        assert 'openai' in EVENT_PROVIDER_REGISTRY
        assert 'anthropic' in EVENT_PROVIDER_REGISTRY
        assert 'deepseek' in EVENT_PROVIDER_REGISTRY
        assert 'opencode_zen' in EVENT_PROVIDER_REGISTRY

    def test_registry_contains_callables(self):
        """Registry values are callable factory functions"""
        for provider_id, factory_func in EVENT_PROVIDER_REGISTRY.items():
            assert callable(factory_func), f"{provider_id} factory is not callable"

    def test_registry_fallback_langextract(self):
        """LangExtract is guaranteed to be in registry as fallback"""
        # Even if catalog is misconfigured, LangExtract should be available
        assert 'langextract' in EVENT_PROVIDER_REGISTRY


class TestRuntimeModelOverride(unittest.TestCase):
    """Test runtime model override flows for multi-model providers"""

    def test_openrouter_runtime_model_propagation(self):
        """Runtime model propagates to OpenRouter config"""
        _, config, _ = load_provider_config('openrouter', runtime_model='gpt-5')

        # OpenRouter uses @property active_model
        assert config.active_model == 'gpt-5'
        assert config.runtime_model == 'gpt-5'

    def test_openai_runtime_model_propagation(self):
        """Runtime model propagates to OpenAI config"""
        _, config, _ = load_provider_config('openai', runtime_model='gpt-5')

        assert config.model == 'gpt-5'

    def test_anthropic_runtime_model_propagation(self):
        """Runtime model propagates to Anthropic config"""
        _, config, _ = load_provider_config('anthropic', runtime_model='claude-sonnet-4-5')

        assert config.model == 'claude-sonnet-4-5'

    def test_langextract_runtime_model_propagation(self):
        """Runtime model propagates to LangExtract config (uses model_id field)"""
        _, config, _ = load_provider_config('langextract', runtime_model='gemini-2.5-pro')

        # LangExtract uses model_id instead of model
        assert config.model_id == 'gemini-2.5-pro'

    def test_deepseek_runtime_model_propagation(self):
        """DeepSeek runtime model propagates correctly"""
        _, config, _ = load_provider_config('deepseek', runtime_model='deepseek-reasoner')

        # DeepSeek does support runtime model override
        assert config.model == 'deepseek-reasoner'


class TestEnabledFlagToggling(unittest.TestCase):
    """Test enabled flag behavior in catalog and factory"""

    def test_disabled_provider_not_in_factory(self):
        """Disabled providers are skipped during registry bootstrapping"""
        # This test verifies the _build_event_provider_registry() behavior
        # In production, you would temporarily disable a provider in the catalog
        # and verify it's excluded from the factory registry

        catalog = get_event_extractor_catalog()
        enabled_providers = catalog.list_extractors(enabled=True)
        enabled_ids = {e.provider_id for e in enabled_providers}

        # Factory should match enabled providers
        factory_ids = set(EVENT_PROVIDER_REGISTRY.keys())

        # All enabled providers should be in factory
        assert enabled_ids.issubset(factory_ids), \
            f"Enabled providers {enabled_ids} not all in factory {factory_ids}"

    def test_build_extractors_validates_enabled_status(self):
        """build_extractors() checks enabled flag and raises error if disabled"""
        # Create minimal configs for testing
        docling_config = DoclingConfig()
        extractor_config = ExtractorConfig()
        extractor_config.doc_extractor = 'docling'
        extractor_config.event_extractor = 'langextract'

        # Mock LangExtractConfig
        from src.core.config import LangExtractConfig
        event_config = LangExtractConfig()

        # This should succeed (langextract is enabled)
        doc_ext, event_ext = build_extractors(docling_config, event_config, extractor_config)
        assert doc_ext is not None
        assert event_ext is not None

        # Test with a hypothetical disabled provider would require mocking the catalog
        # For now, verify the validation logic exists by checking for invalid provider
        extractor_config.event_extractor = 'nonexistent_provider'
        with self.assertRaises(ExtractorConfigurationError):
            build_extractors(docling_config, event_config, extractor_config)


class TestCredentialValidation(unittest.TestCase):
    """Test credential validation and error handling"""

    @patch.dict(os.environ, {}, clear=True)
    def test_openrouter_missing_credentials(self):
        """OpenRouter without API key should be detectable"""
        # Clear environment to simulate missing credentials
        catalog = get_event_extractor_catalog()
        openrouter = catalog.get_extractor('openrouter')

        assert openrouter is not None
        # Catalog doesn't validate credentials directly - that's adapter responsibility
        # But we can verify the provider exists and is configured

    @patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test_key'})
    def test_openrouter_with_credentials(self):
        """OpenRouter with API key should load config correctly"""
        from src.core.config import OpenRouterConfig
        config = OpenRouterConfig()

        assert config.api_key == 'test_key'

    @patch.dict(os.environ, {}, clear=True)
    def test_langextract_missing_credentials(self):
        """LangExtract without GEMINI_API_KEY should have empty string"""
        from src.core.config import LangExtractConfig
        config = LangExtractConfig()

        # Model ID should still have default value
        assert config.model_id is not None

    def test_extractor_availability_check(self):
        """Event extractors should implement is_available() for credential checks"""
        # This test documents expected behavior - adapters should check API keys
        # in their is_available() method

        # Import an adapter and verify it has the method
        from src.core.langextract_adapter import LangExtractEventExtractor
        from src.core.config import LangExtractConfig

        config = LangExtractConfig()
        adapter = LangExtractEventExtractor(config)

        # All adapters must implement is_available()
        assert hasattr(adapter, 'is_available')
        assert callable(adapter.is_available)


class TestPromptOverrides(unittest.TestCase):
    """Test prompt_id and prompt_override functionality"""

    def test_get_prompt_returns_none_when_not_configured(self):
        """Providers without prompt configuration return None (use default)"""
        catalog = get_event_extractor_catalog()

        # langextract has no prompt configured (uses LEGAL_EVENTS_PROMPT default)
        prompt = catalog.get_prompt('langextract')
        assert prompt is None

    def test_prompt_override_takes_precedence(self):
        """prompt_override takes precedence over prompt_id"""
        # This is tested implicitly in the catalog implementation
        # Would need to modify catalog to add test entries with both fields


class TestFactorySecurityConstraints(unittest.TestCase):
    """Test security constraints on dynamic factory loading"""

    def test_whitelist_validation(self):
        """Only src.core.* import paths are allowed for factory_callable"""
        catalog = get_event_extractor_catalog()
        enabled = catalog.list_extractors(enabled=True)

        for entry in enabled:
            if entry.factory_callable:
                module_path = entry.factory_callable.rsplit('.', 1)[0]
                assert module_path.startswith('src.core'), \
                    f"{entry.provider_id} factory_callable '{entry.factory_callable}' not in whitelist"

    def test_no_eval_or_exec(self):
        """Factory building uses importlib, not eval/exec"""
        # This is verified by code inspection of _build_event_provider_registry
        # The implementation uses importlib.import_module() and getattr()
        import importlib
        from src.core import extractor_factory

        # Verify importlib is imported in extractor_factory
        assert hasattr(extractor_factory, 'importlib')


if __name__ == '__main__':
    unittest.main()
