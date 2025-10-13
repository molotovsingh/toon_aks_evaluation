"""
Unit tests for PipelineMetadata runtime model extraction

Verifies that metadata correctly captures runtime model overrides
from all provider types (OpenRouter, OpenAI, Anthropic, LangExtract, DeepSeek).
"""

import unittest
from datetime import datetime
from unittest.mock import Mock, patch
from dataclasses import dataclass

from src.core.pipeline_metadata import PipelineMetadata


# Mock config classes (mimicking real config structures)
@dataclass
class MockOpenRouterConfig:
    """Mock OpenRouter config with active_model property"""
    model: str = "openai/gpt-4o-mini"
    runtime_model: str = None

    @property
    def active_model(self):
        """Returns runtime_model if set, else model (same as real OpenRouterConfig)"""
        return self.runtime_model or self.model


@dataclass
class MockOpenAIConfig:
    """Mock OpenAI config"""
    model: str = "gpt-4o-mini"


@dataclass
class MockLangExtractConfig:
    """Mock LangExtract config"""
    model_id: str = "gemini-2.0-flash"


@dataclass
class MockAnthropicConfig:
    """Mock Anthropic config"""
    model: str = "claude-3-haiku-20240307"


@dataclass
class MockDeepSeekConfig:
    """Mock DeepSeek config"""
    model: str = "deepseek-chat"


class TestPipelineMetadataRuntimeModel(unittest.TestCase):
    """Test runtime model extraction from different provider types"""

    def setUp(self):
        """Set up mock pipeline for testing"""
        self.mock_pipeline = Mock()
        self.mock_pipeline.provider = "openrouter"

        # Mock document extractor (not testing this part)
        self.mock_pipeline.document_extractor = Mock()

    def test_openrouter_runtime_model_override(self):
        """Test that OpenRouter runtime model override is captured"""
        # Setup: OpenRouter adapter with runtime model override
        mock_extractor = Mock()
        mock_extractor.config = MockOpenRouterConfig(
            model="openai/gpt-4o-mini",
            runtime_model="openai/gpt-oss-120b"  # Runtime override
        )
        self.mock_pipeline.event_extractor = mock_extractor
        self.mock_pipeline.provider = "openrouter"

        # Execute: Extract metadata
        metadata = PipelineMetadata.from_pipeline(self.mock_pipeline)

        # Verify: Should capture runtime override, not env default
        self.assertEqual(metadata.provider_model, "openai/gpt-oss-120b")
        self.assertEqual(metadata.provider_name, "openrouter")

    def test_openrouter_no_runtime_override(self):
        """Test that OpenRouter uses env default when no runtime override"""
        # Setup: OpenRouter adapter without runtime model override
        mock_extractor = Mock()
        mock_extractor.config = MockOpenRouterConfig(
            model="openai/gpt-4o-mini",
            runtime_model=None  # No runtime override
        )
        self.mock_pipeline.event_extractor = mock_extractor
        self.mock_pipeline.provider = "openrouter"

        # Execute: Extract metadata
        metadata = PipelineMetadata.from_pipeline(self.mock_pipeline)

        # Verify: Should use env default from active_model property
        self.assertEqual(metadata.provider_model, "openai/gpt-4o-mini")

    def test_openai_config_model(self):
        """Test that OpenAI config.model is captured"""
        # Setup: OpenAI adapter with config.model
        mock_extractor = Mock()
        mock_extractor.config = MockOpenAIConfig(model="gpt-5")
        self.mock_pipeline.event_extractor = mock_extractor
        self.mock_pipeline.provider = "openai"

        # Execute: Extract metadata
        metadata = PipelineMetadata.from_pipeline(self.mock_pipeline)

        # Verify: Should capture config.model
        self.assertEqual(metadata.provider_model, "gpt-5")
        self.assertEqual(metadata.provider_name, "openai")

    def test_anthropic_config_model(self):
        """Test that Anthropic config.model is captured"""
        # Setup: Anthropic adapter with config.model
        mock_extractor = Mock()
        mock_extractor.config = MockAnthropicConfig(model="claude-sonnet-4-5")
        self.mock_pipeline.event_extractor = mock_extractor
        self.mock_pipeline.provider = "anthropic"

        # Execute: Extract metadata
        metadata = PipelineMetadata.from_pipeline(self.mock_pipeline)

        # Verify: Should capture config.model
        self.assertEqual(metadata.provider_model, "claude-sonnet-4-5")

    def test_langextract_config_model_id(self):
        """Test that LangExtract config.model_id is captured"""
        # Setup: LangExtract adapter with config.model_id
        mock_extractor = Mock()
        mock_extractor.config = MockLangExtractConfig(model_id="gemini-2.5-pro")
        self.mock_pipeline.event_extractor = mock_extractor
        self.mock_pipeline.provider = "langextract"

        # Execute: Extract metadata
        metadata = PipelineMetadata.from_pipeline(self.mock_pipeline)

        # Verify: Should capture config.model_id
        self.assertEqual(metadata.provider_model, "gemini-2.5-pro")

    def test_deepseek_config_model(self):
        """Test that DeepSeek config.model is captured"""
        # Setup: DeepSeek adapter with config.model
        mock_extractor = Mock()
        mock_extractor.config = MockDeepSeekConfig(model="deepseek-r1")
        self.mock_pipeline.event_extractor = mock_extractor
        self.mock_pipeline.provider = "deepseek"

        # Execute: Extract metadata
        metadata = PipelineMetadata.from_pipeline(self.mock_pipeline)

        # Verify: Should capture config.model
        self.assertEqual(metadata.provider_model, "deepseek-r1")

    @patch.dict('os.environ', {'OPENROUTER_MODEL': 'openai/gpt-4o'})
    def test_fallback_to_environment_variable(self):
        """Test fallback to environment variable when no config found"""
        # Setup: Extractor without config attribute
        mock_extractor = Mock(spec=[])  # No attributes
        self.mock_pipeline.event_extractor = mock_extractor
        self.mock_pipeline.provider = "openrouter"

        # Execute: Extract metadata
        metadata = PipelineMetadata.from_pipeline(self.mock_pipeline)

        # Verify: Should fall back to environment variable
        self.assertEqual(metadata.provider_model, "openai/gpt-4o")

    def test_strategy_priority_order(self):
        """Test that strategy 1 (active_model) takes priority over strategy 2 (model)"""
        # Setup: Extractor with both active_model and model attributes
        mock_extractor = Mock()
        mock_config = MockOpenRouterConfig(
            model="openai/gpt-4o-mini",
            runtime_model="openai/qwen-qwq-32b"  # Runtime override
        )
        # Add both active_model property and model attribute
        mock_extractor.config = mock_config
        mock_extractor.model = "should-not-use-this"  # Should be ignored

        self.mock_pipeline.event_extractor = mock_extractor
        self.mock_pipeline.provider = "openrouter"

        # Execute: Extract metadata
        metadata = PipelineMetadata.from_pipeline(self.mock_pipeline)

        # Verify: Should use active_model (strategy 1), not model (fallback)
        self.assertEqual(metadata.provider_model, "openai/qwen-qwq-32b")

    def test_backward_compatibility_direct_properties(self):
        """Test backward compatibility with direct model_id property"""
        # Setup: Old-style extractor with direct model_id attribute (no config)
        mock_extractor = Mock()
        mock_extractor.model_id = "legacy-model-id"
        # Remove config to simulate old adapter
        mock_extractor.config = None

        self.mock_pipeline.event_extractor = mock_extractor
        self.mock_pipeline.provider = "langextract"

        # Execute: Extract metadata
        metadata = PipelineMetadata.from_pipeline(self.mock_pipeline)

        # Verify: Should fall back to direct property
        self.assertEqual(metadata.provider_model, "legacy-model-id")


if __name__ == '__main__':
    unittest.main()
