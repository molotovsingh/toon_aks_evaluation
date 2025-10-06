"""
Extractor Factory - Creates document and event extractors based on configuration
Supports environment variable overrides for different implementations
"""

import logging
from typing import Tuple, Callable, Dict, Any, Optional

from .interfaces import DocumentExtractor, EventExtractor
from .config import (
    DoclingConfig, LangExtractConfig, ExtractorConfig, GeminiDocConfig,
    OpenRouterConfig, OpenCodeZenConfig, OpenAIConfig, AnthropicConfig, DeepSeekConfig,
    load_config, load_provider_config
)
from .docling_adapter import DoclingDocumentExtractor
from .gemini_doc_extractor import GeminiDocumentExtractor
from .langextract_adapter import LangExtractEventExtractor
from .openrouter_adapter import OpenRouterEventExtractor
from .opencode_zen_adapter import OpenCodeZenEventExtractor
from .openai_adapter import OpenAIEventExtractor
from .anthropic_adapter import AnthropicEventExtractor
from .deepseek_adapter import DeepSeekEventExtractor

logger = logging.getLogger(__name__)


class ExtractorConfigurationError(ValueError):
    """Raised when extractor provider configuration is invalid."""


# Document Extractor Factories

def _create_docling_document_extractor(
    doc_config: DoclingConfig,
    _event_config: Any,
    _extractor_config: ExtractorConfig
) -> DocumentExtractor:
    """Factory for the Docling document extractor (local PDF processing)."""
    return DoclingDocumentExtractor(doc_config)


def _create_gemini_document_extractor(
    _doc_config: DoclingConfig,
    _event_config: Any,
    _extractor_config: ExtractorConfig
) -> DocumentExtractor:
    """Factory for the Gemini document extractor (cloud multimodal vision)."""
    gemini_config = GeminiDocConfig()
    return GeminiDocumentExtractor(gemini_config)


DOC_PROVIDER_REGISTRY: Dict[str, Callable[[DoclingConfig, Any, ExtractorConfig], DocumentExtractor]] = {
    "docling": _create_docling_document_extractor,
    "gemini": _create_gemini_document_extractor,
}


# Event Extractor Factories

def _create_langextract_event_extractor(
    _doc_config: DoclingConfig,
    event_config: Any,
    _extractor_config: ExtractorConfig
) -> EventExtractor:
    """Factory for the default LangExtract adapter."""
    return LangExtractEventExtractor(event_config)


def _create_openrouter_event_extractor(
    _doc_config: DoclingConfig,
    event_config: Any,
    _extractor_config: ExtractorConfig
) -> EventExtractor:
    """Factory for the OpenRouter adapter."""
    return OpenRouterEventExtractor(event_config)


def _create_opencode_zen_event_extractor(
    _doc_config: DoclingConfig,
    event_config: Any,
    _extractor_config: ExtractorConfig
) -> EventExtractor:
    """Factory for the OpenCode Zen adapter."""
    return OpenCodeZenEventExtractor(event_config)


def _create_openai_event_extractor(
    _doc_config: DoclingConfig,
    event_config: Any,
    _extractor_config: ExtractorConfig
) -> EventExtractor:
    """Factory for the OpenAI adapter."""
    return OpenAIEventExtractor(event_config)


def _create_anthropic_event_extractor(
    _doc_config: DoclingConfig,
    event_config: Any,
    _extractor_config: ExtractorConfig
) -> EventExtractor:
    """Factory for the Anthropic adapter."""
    return AnthropicEventExtractor(event_config)


def _create_deepseek_event_extractor(
    _doc_config: DoclingConfig,
    event_config: Any,
    _extractor_config: ExtractorConfig
) -> EventExtractor:
    """Factory for the DeepSeek adapter."""
    return DeepSeekEventExtractor(event_config)


EVENT_PROVIDER_REGISTRY: Dict[str, Callable[[DoclingConfig, Any, ExtractorConfig], EventExtractor]] = {
    "langextract": _create_langextract_event_extractor,
    "openrouter": _create_openrouter_event_extractor,
    "opencode_zen": _create_opencode_zen_event_extractor,
    "openai": _create_openai_event_extractor,
    "anthropic": _create_anthropic_event_extractor,
    "deepseek": _create_deepseek_event_extractor,
}


def build_extractors(
    docling_config: DoclingConfig,
    event_config: Any,
    extractor_config: ExtractorConfig
) -> Tuple[DocumentExtractor, EventExtractor]:
    """
    Build document and event extractors based on configuration

    Args:
        docling_config: Configuration for document processing
        event_config: Provider-specific configuration for event extraction
        extractor_config: Configuration for extractor selection

    Returns:
        Tuple of (DocumentExtractor, EventExtractor) instances

    Raises:
        ValueError: If an unsupported document extractor is requested.
        ExtractorConfigurationError: If the event extractor provider is not registered.
    """
    # Get extractor types from config
    doc_extractor_type = extractor_config.doc_extractor.lower()
    event_extractor_type = extractor_config.event_extractor.lower()

    logger.info(f"🏭 Building extractors: DOC={doc_extractor_type}, EVENT={event_extractor_type}")

    # Create document extractor
    doc_factory = DOC_PROVIDER_REGISTRY.get(doc_extractor_type)
    if not doc_factory:
        available = ", ".join(sorted(DOC_PROVIDER_REGISTRY)) or "none"
        logger.error(f"❌ Unsupported document extractor type: {doc_extractor_type}")
        raise ExtractorConfigurationError(
            f"Unsupported document extractor type: {doc_extractor_type}. Available providers: {available}"
        )
    doc_extractor = doc_factory(docling_config, None, extractor_config)
    logger.info(f"✅ Created {doc_extractor.__class__.__name__}")

    # Create event extractor
    event_factory = EVENT_PROVIDER_REGISTRY.get(event_extractor_type)
    if not event_factory:
        available = ", ".join(sorted(EVENT_PROVIDER_REGISTRY)) or "none"
        logger.error(f"❌ Unsupported event extractor type: {event_extractor_type}")
        raise ExtractorConfigurationError(
            f"Unsupported event extractor type: {event_extractor_type}. Available providers: {available}"
        )
    event_extractor = event_factory(docling_config, event_config, extractor_config)
    logger.info(f"✅ Created {event_extractor.__class__.__name__}")

    return doc_extractor, event_extractor


def create_default_extractors(
    event_extractor_override: Optional[str] = None,
    runtime_model: Optional[str] = None,
    doc_extractor_override: Optional[str] = None
) -> Tuple[DocumentExtractor, EventExtractor]:
    """
    Create extractors with default configuration from environment

    Args:
        event_extractor_override: Override the event extractor provider from environment
        runtime_model: Runtime model selection (for OpenRouter multi-model support)
        doc_extractor_override: Override the document extractor provider from environment

    Returns:
        Tuple of (DocumentExtractor, EventExtractor) instances
    """
    # First load base config to determine the event extractor type
    docling_config, _, extractor_config = load_config()

    if event_extractor_override:
        extractor_config.event_extractor = event_extractor_override

    if doc_extractor_override:
        extractor_config.doc_extractor = doc_extractor_override

    # Load provider-specific configuration based on the event extractor type
    docling_config, event_config, extractor_config = load_provider_config(
        extractor_config.event_extractor,
        docling_config=docling_config,
        extractor_config=extractor_config,
        runtime_model=runtime_model
    )

    return build_extractors(docling_config, event_config, extractor_config)


def validate_extractors(doc_extractor: DocumentExtractor, event_extractor: EventExtractor) -> bool:
    """
    Validate that extractors are properly configured

    Args:
        doc_extractor: Document extractor to validate
        event_extractor: Event extractor to validate

    Returns:
        True if both extractors are valid, False otherwise
    """
    try:
        # Check document extractor
        supported_types = doc_extractor.get_supported_types()
        if not supported_types:
            logger.error("❌ Document extractor supports no file types")
            return False

        # Check event extractor
        if not event_extractor.is_available():
            logger.error("❌ Event extractor is not available")
            return False

        logger.info(f"✅ Extractors validated: {len(supported_types)} file types supported")
        return True

    except Exception as e:
        logger.error(f"❌ Extractor validation failed: {e}")
        return False
