#!/usr/bin/env python3
"""
STANDARDIZED Legal Events Extraction - Granite Guardrails Compliant
Unified Five-Column Table: Docling + LangExtract + Shared Utilities
"""

import streamlit as st
import os
import logging
import time
from pathlib import Path

# Load environment variables FIRST (before imports that need it)
from dotenv import load_dotenv
load_dotenv()

# Import refactored shared utilities for guard-railed five-column exports
from src.ui.streamlit_common import (
    get_pipeline,
    process_documents_with_spinner,
    create_download_section,
    display_legal_events_table,
)
from src.ui.cost_estimator import estimate_all_models_with_tiktoken
from src.core.legal_pipeline_refactored import LegalEventsPipeline
from src.utils.file_handler import FileHandler
from src.ui.classification_ui import (
    create_classification_config,
    show_classification_results
)
from src.core.config import (
    OpenRouterConfig,
    OpenAIConfig,
    AnthropicConfig,
    DeepSeekConfig,
    GeminiEventConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reduce LangExtract AFC logging noise
logging.getLogger('langextract').setLevel(logging.WARNING)

# Page config
st.set_page_config(
    page_title="Paralegal Date Extraction Test - Docling + Langextract",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Compact Modern Design
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* === GLOBAL STYLES === */
.main-header {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 0.25rem;
}
.main-caption {
    font-family: 'Inter', sans-serif;
    font-size: 0.765rem;
    color: #64748b;
    text-align: center;
    margin-bottom: 1rem;
    font-weight: 400;
}

/* === PROVIDER BUTTON ENHANCEMENTS === */
/* NOTE: Using data-testid selectors for Streamlit elements. These are brittle and may break
   in future Streamlit versions, but are necessary as Streamlit doesn't provide stable CSS classes */
div[data-testid="column"] > div > div > button {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%) !important;
    border: 2px solid #e2e8f0 !important;
    border-radius: 8px !important;
    padding: 0.75rem 0.75rem !important;
    font-weight: 600 !important;
    font-size: 0.855rem !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
    height: auto !important;
    min-height: 45px !important;
}

div[data-testid="column"] > div > div > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.12) !important;
    border-color: #cbd5e1 !important;
}

/* === STATUS BADGES === */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.25rem 0.5rem;
    border-radius: 12px;
    font-size: 0.63rem;
    font-weight: 600;
    margin-top: 0.125rem;
}
.status-pill.ready {
    background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
    color: #065f46;
    border: 1px solid #6ee7b7;
}
.status-pill.setup {
    background: linear-gradient(135deg, #fed7aa 0%, #fdba74 100%);
    color: #92400e;
    border: 1px solid #fb923c;
}

/* === SECTION HEADERS === */
.section-header {
    font-size: 0.675rem;
    font-weight: 700;
    color: #1e293b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
    padding-bottom: 0.25rem;
    border-bottom: 2px solid #e2e8f0;
}

/* === ADVANCED SECTION STYLING === */
div[data-testid="stExpander"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 6px !important;
    background: #f8fafc !important;
    margin-top: 0.5rem !important;
}

/* === METRICS CARDS === */
.metric-card {
    padding: 1rem 0.75rem;
    border-radius: 8px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    transition: transform 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-1px);
}
.metric-card.success {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
}
.metric-card.info {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
}
.metric-card.warning {
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
}
.metric-value {
    font-size: 1.575rem;
    font-weight: 700;
    margin: 0.25rem 0;
    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.metric-label {
    font-size: 0.675rem;
    opacity: 0.95;
    font-weight: 500;
}

/* === DIVIDER ENHANCEMENT === */
hr {
    margin: 0.75rem 0 !important;
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent 0%, #e2e8f0 50%, transparent 100%) !important;
}

/* === CAPTIONS & TEXT === */
.stMarkdown {
    font-size: 0.72rem;
}
div[data-testid="stCaptionContainer"] {
    color: #64748b !important;
    font-size: 0.675rem !important;
    margin-top: 0.125rem !important;
}

/* === COMPACT SPACING === */
.element-container {
    margin-bottom: 0.5rem !important;
}

/* === SMOOTH ANIMATIONS === */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(5px); }
    to { opacity: 1; transform: translateY(0); }
}

.provider-card, .metric-card {
    animation: fadeIn 0.3s ease-out;
}

/* === PIPELINE STATUS TABLE === */
.pipeline-table {
    width: 100%;
    border-collapse: collapse;
    margin: 0.75rem 0;
    background: #fff;
    border-radius: 6px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
.pipeline-table td {
    border: 1px solid #e2e8f0;
    padding: 0.6rem 0.5rem;
    text-align: center;
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    font-size: 0.675rem;
    color: #64748b;
    vertical-align: middle;
}
.pipeline-stage-header {
    font-size: 0.585rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #475569;
    margin-bottom: 0.25rem;
}
.pipeline-stage-value {
    font-size: 0.765rem;
    font-weight: 600;
    color: #1e293b;
}
.pipeline-table td:first-child {
    border-left: 3px solid #0ea5e9;
}
.pipeline-table td:nth-child(3) {
    border-left: 3px solid #10b981;
}
.pipeline-table td:last-child {
    border-right: 3px solid #8b5cf6;
}
.pipeline-arrow {
    font-size: 1.08rem;
    color: #cbd5e1;
    font-weight: 300;
    padding: 0 0.25rem;
}
</style>
""", unsafe_allow_html=True)


def check_provider_status(provider_key: str) -> bool:
    """Check if provider API key is configured"""
    key_map = {
        'langextract': ['GEMINI_API_KEY', 'GOOGLE_API_KEY'],
        'openrouter': ['OPENROUTER_API_KEY'],
        'opencode_zen': ['OPENCODEZEN_API_KEY'],
        'openai': ['OPENAI_API_KEY'],
        'anthropic': ['ANTHROPIC_API_KEY'],
        'deepseek': ['DEEPSEEK_API_KEY']
    }

    required_keys = key_map.get(provider_key, [])
    return any(os.getenv(key) for key in required_keys)


def create_pipeline_status(doc_extractor: str, provider: str, model: str = None):
    """
    Display visual pipeline showing complete data flow:
    Document Processing → Event Extraction → Model Selection

    Args:
        doc_extractor: extractor_id from catalog (e.g., 'docling', 'qwen_vl')
        provider: 'openrouter', 'langextract', 'openai', 'anthropic', 'deepseek', 'opencode_zen'
        model: Selected model identifier (optional)
    """
    # Stage 1: Document Processing (dynamically loaded from catalog)
    from src.core.document_extractor_catalog import get_doc_extractor_catalog
    catalog = get_doc_extractor_catalog()
    doc_entry = catalog.get_extractor(doc_extractor)

    if doc_entry:
        # Use catalog metadata for display
        emoji = "🔧" if doc_entry.provider == "local" else "👁️"
        # Extract short name (e.g., "Docling" from "Docling (Local OCR)")
        short_name = doc_entry.display_name.split('(')[0].strip()
        provider_label = doc_entry.provider.title() if doc_entry.provider != "local" else "Local OCR"
        doc_display = f'{emoji} {short_name}<br><span style="font-size:0.63rem;">{provider_label}</span>'
    else:
        # Fallback if extractor not in catalog
        doc_display = doc_extractor

    # Stage 2: Event Extraction Provider
    provider_display_map = {
        'langextract': '🔮 Gemini<br><span style="font-size:0.63rem;">LangExtract</span>',
        'openrouter': '🌐 OpenRouter<br><span style="font-size:0.63rem;">18+ Models</span>',
        'opencode_zen': '⚖️ OpenCode Zen<br><span style="font-size:0.63rem;">Legal AI</span>',
        'openai': '🧠 OpenAI<br><span style="font-size:0.63rem;">GPT-4o</span>',
        'anthropic': '🎯 Anthropic<br><span style="font-size:0.63rem;">Claude 3.5</span>',
        'deepseek': '🔍 DeepSeek<br><span style="font-size:0.63rem;">R1 Chat</span>'
    }
    provider_display = provider_display_map.get(provider, provider)

    # Stage 3: Model Selection
    if model:
        # Extract friendly model name
        model_parts = model.split('/')
        model_short = model_parts[-1] if len(model_parts) > 1 else model
        # Lookup full display name from MODEL_CATALOG if available
        matching_models = [m for m in MODEL_CATALOG if m.model_id == model]
        if matching_models:
            model_obj = matching_models[0]
            model_display = f'{model_obj.display_name}<br><span style="font-size:0.63rem;">{model_obj.cost_per_1m}</span>'
        else:
            model_display = f'{model_short}<br><span style="font-size:0.63rem;">Selected</span>'
    else:
        model_display = '<span style="font-style:italic; color:#94a3b8;">Not selected</span>'

    # Render pipeline table
    st.markdown(f"""
    <table class="pipeline-table">
        <tr>
            <td>
                <div class="pipeline-stage-header">1. DOC PROCESSING</div>
                <div class="pipeline-stage-value">{doc_display}</div>
            </td>
            <td class="pipeline-arrow">→</td>
            <td>
                <div class="pipeline-stage-header">2. EVENT EXTRACTION</div>
                <div class="pipeline-stage-value">{provider_display}</div>
            </td>
            <td class="pipeline-arrow">→</td>
            <td>
                <div class="pipeline-stage-header">3. MODEL</div>
                <div class="pipeline-stage-value">{model_display}</div>
            </td>
        </tr>
    </table>
    """, unsafe_allow_html=True)


def create_pipeline_config_display(doc_extractor: str, provider: str, model: str = None):
    """
    Display collapsible pipeline configuration in JSON format.
    Useful for debugging, sharing configurations, and validating selections.

    Args:
        doc_extractor: 'docling' or 'qwen_vl'
        provider: 'openrouter', 'langextract', 'openai', 'anthropic', 'deepseek', 'opencode_zen'
        model: Selected model identifier (optional)
    """
    # Build comprehensive config object
    config = {
        "pipeline_version": "v2",
        "document_extraction": {
            "engine": doc_extractor,
            "ocr_enabled": doc_extractor == "docling",
            "cost_per_doc": "FREE" if doc_extractor == "docling" else "$0.0014"
        },
        "event_extraction": {
            "provider": provider,
            "model": model if model else "not_selected",
        }
    }

    # Enrich with model metadata if available
    if model:
        matching_models = [m for m in MODEL_CATALOG if m.model_id == model]
        if matching_models:
            model_obj = matching_models[0]
            config["event_extraction"].update({
                "model_display_name": model_obj.display_name,
                "category": model_obj.category,
                "cost": model_obj.cost_per_1m,
                "context_window": model_obj.context_window,
                "quality_score": model_obj.quality_score,
                "badges": model_obj.badges
            })

    # Display in collapsible expander
    with st.expander("📋 Pipeline Configuration (JSON)", expanded=False):
        st.caption("Copy this configuration to share or reproduce your setup")
        st.json(config)


# === MODEL CONFIGURATION ===
# Import centralized model catalog
from src.core.model_catalog import get_ui_model_config_list

# MODEL_CATALOG: Generated from centralized model catalog for backward compatibility
MODEL_CATALOG = get_ui_model_config_list()


def normalize_search_text(text: str) -> str:
    """
    Normalize text for flexible search matching.
    Converts separators (hyphens, dots, underscores, slashes) to spaces
    to enable searches like 'gpt 5' to match 'gpt-5'.
    """
    import re
    # Convert to lowercase
    text = text.lower()
    # Replace common separators with spaces
    text = re.sub(r'[-_./ ]', ' ', text)
    # Collapse multiple spaces to single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing spaces
    return text.strip()


def create_unified_model_selector(provider: str, container=st) -> str:
    """
    Unified model selector with search, advanced filters, and inline metadata.
    Replaces all 4 individual provider selectors with one reusable function.

    Args:
        provider: 'anthropic', 'openai', 'google', 'openrouter', 'deepseek'
        container: Streamlit container (st or st.sidebar)

    Returns:
        model_id string
    """
    # Filter catalog by provider
    provider_models = [m for m in MODEL_CATALOG if m.provider == provider]

    if not provider_models:
        container.error(f"No models found for provider: {provider}")
        return None

    # Search box
    search_query = container.text_input(
        "🔍 Search Models",
        placeholder=f"Search {len(provider_models)} models...",
        key=f"{provider}_search"
    )

    # Advanced filters (collapsible)
    show_filters = container.checkbox("🔧 Advanced Filters", key=f"{provider}_show_filters")

    # Category filter (when advanced filters enabled)
    category_filter = "All"
    if show_filters:
        # Get unique categories for this provider
        categories = sorted(set(m.category for m in provider_models))
        category_filter = container.selectbox(
            "Category",
            ["All"] + categories,
            key=f"{provider}_category"
        )

    # Apply filters
    filtered = provider_models

    # Filter by search (normalized)
    if search_query:
        normalized_query = normalize_search_text(search_query)
        filtered = [
            m for m in filtered
            if (normalized_query in normalize_search_text(m.display_name)
                or normalized_query in normalize_search_text(m.model_id)
                or normalized_query in normalize_search_text(m.category)
                or normalized_query in normalize_search_text(m.cost_per_1m)
                or normalized_query in normalize_search_text(m.context_window)
                or normalized_query in normalize_search_text(m.quality_score)
                or any(normalized_query in normalize_search_text(badge) for badge in m.badges))
        ]

    # Filter by category (when advanced filters enabled)
    if category_filter != "All":
        filtered = [m for m in filtered if m.category == category_filter]

    # Show results count when filtering
    if search_query or show_filters:
        container.caption(f"Found {len(filtered)} model(s)")

    if not filtered:
        container.warning("No models match your search.")
        # Return first provider model as fallback
        return provider_models[0].model_id

    # Session state key based on provider
    session_key = f'{provider}_model'

    # Get default from env or use config defaults (not first model in catalog)
    env_defaults = {
        'anthropic': os.getenv('ANTHROPIC_MODEL', AnthropicConfig().model),
        'openai': os.getenv('OPENAI_MODEL', OpenAIConfig().model),
        'google': os.getenv('GEMINI_MODEL_ID', GeminiEventConfig().model_id),
        'openrouter': os.getenv('OPENROUTER_MODEL', OpenRouterConfig().model),
        'deepseek': os.getenv('DEEPSEEK_MODEL', DeepSeekConfig().model)
    }
    default_model = env_defaults.get(provider, filtered[0].model_id)

    if session_key not in st.session_state:
        st.session_state[session_key] = default_model

    # Build selectbox options
    options = [m.model_id for m in filtered]
    display_map = {
        m.model_id: f"{m.display_name} • {m.format_inline()}"
        for m in filtered
    }

    # Ensure default is in filtered options
    if st.session_state[session_key] not in options:
        st.session_state[session_key] = options[0]

    # CONDITIONAL DISPLAY: Only show full selectbox when actively searching or filtering
    # This reduces clutter for providers with many models (e.g., OpenRouter with 18+ models)
    if search_query or show_filters:
        # Show full selectbox with filtered results
        selected = container.selectbox(
            "Select Model",
            options=options,
            index=options.index(st.session_state[session_key]),
            format_func=lambda x: display_map.get(x, x),
            key=f"{provider}_selector"
        )
    else:
        # Show compact default selection (no overwhelming dropdown)
        current_model = st.session_state[session_key]
        matching_models = [m for m in provider_models if m.model_id == current_model]
        if matching_models:
            model_obj = matching_models[0]
            container.markdown(f"**Selected**: {model_obj.display_name}")
            container.caption(f"{model_obj.format_inline()}")
        else:
            container.caption(f"**Selected**: {current_model}")

        # Use current selection
        selected = current_model

    # Update session state if changed
    if selected != st.session_state[session_key]:
        st.session_state[session_key] = selected
        if 'legal_events_df' in st.session_state:
            del st.session_state.legal_events_df

    return selected


def create_gemini_model_selector(container=st) -> str:
    """
    Custom Gemini model selector showing 4 options:
    1. Gemini 2.5 Pro (direct API)
    2. Gemini 2.5 Flash (direct API)
    3. Gemini 2.0 Flash (direct API)
    4. LangExtract (structured few-shot extraction)

    Returns:
        model_id string (or "langextract" for LangExtract option)
    """
    # Build options list: catalog models + LangExtract option
    options = []
    display_map = {}

    # Define Gemini models in preferred order: Pro → 2.5 Flash → 2.0 Flash
    gemini_model_ids = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"]

    # Add Gemini models that exist in catalog
    for model_id in gemini_model_ids:
        # Find model in catalog
        matching_models = [m for m in MODEL_CATALOG if m.model_id == model_id and m.provider == "google"]
        if matching_models:
            model = matching_models[0]
            options.append(model.model_id)
            display_map[model.model_id] = f"{model.display_name} • {model.format_inline()}"

    # Add LangExtract as special option
    options.append("langextract")
    display_map["langextract"] = "LangExtract (Structured Extraction) • Recommended • Free • 1M"

    # Session state key
    session_key = 'gemini_model'

    # Default: LangExtract (recommended for structured extraction)
    default_model = os.getenv('GEMINI_MODEL_ID', "langextract")

    if session_key not in st.session_state:
        st.session_state[session_key] = default_model

    # Ensure default is in options
    if st.session_state[session_key] not in options:
        st.session_state[session_key] = "langextract"  # Fallback to LangExtract

    # Selectbox with all 4 options
    selected = container.selectbox(
        "Select Model",
        options=options,
        index=options.index(st.session_state[session_key]),
        format_func=lambda x: display_map.get(x, x),
        key="gemini_selector"
    )

    # Update session state if changed
    if selected != st.session_state[session_key]:
        st.session_state[session_key] = selected
        if 'legal_events_df' in st.session_state:
            del st.session_state.legal_events_df

    return selected


def create_doc_extractor_selection():
    """Create document extractor selection UI (dynamically generated from registry)"""
    st.markdown('<div class="section-header">📄 Document Processing</div>', unsafe_allow_html=True)

    # Import catalog
    from src.core.document_extractor_catalog import get_doc_extractor_catalog

    # Initialize session state for document extractor
    default_doc_extractor = os.getenv('DOC_EXTRACTOR', 'docling').lower()
    if 'selected_doc_extractor' not in st.session_state:
        st.session_state.selected_doc_extractor = default_doc_extractor

    # Get enabled extractors from catalog
    catalog = get_doc_extractor_catalog()
    enabled_extractors = catalog.list_extractors(enabled=True)

    if not enabled_extractors:
        st.error("❌ No document extractors available. Check catalog configuration.")
        return default_doc_extractor

    # Build options dynamically from catalog
    doc_options = {}
    for extractor in enabled_extractors:
        # Check if API key required (for paid extractors)
        requires_api_key = extractor.cost_per_page > 0
        api_key_configured = bool(os.getenv('OPENROUTER_API_KEY')) if extractor.provider == 'openrouter' else True

        # Build display info
        emoji = "🔧" if extractor.provider == "local" else "👁️"
        doc_options[extractor.extractor_id] = {
            'name': f'{emoji} {extractor.display_name}',
            'subtitle': extractor.cost_display + (' • ' + extractor.processing_speed.title() if extractor.processing_speed else ''),
            'description': extractor.notes if api_key_configured else f'⚠️ Requires API key for {extractor.provider}'
        }

    # Get current index for radio button
    options_list = [e.extractor_id for e in enabled_extractors]
    try:
        current_index = options_list.index(st.session_state.selected_doc_extractor)
    except ValueError:
        current_index = 0

    # Radio button for document extractor selection
    selected_doc_extractor = st.radio(
        "Choose document processor:",
        options=options_list,
        format_func=lambda x: doc_options[x]['name'],
        index=current_index,
        help="Select document extraction engine. Local extractors are free, vision extractors offer better quality for poor scans."
    )

    # Show subtitle and description for selected option
    option_info = doc_options[selected_doc_extractor]
    st.caption(f"{option_info['subtitle']}")
    st.caption(f"ℹ️ {option_info['description']}")

    # Warn if paid extractor selected but not configured
    selected_entry = catalog.get_extractor(selected_doc_extractor)
    if selected_entry and selected_entry.cost_per_page > 0:
        if selected_entry.provider == 'openrouter' and not os.getenv('OPENROUTER_API_KEY'):
            st.warning("⚠️ **OpenRouter API Key Required**: Set `OPENROUTER_API_KEY` in your `.env` file to use this vision extractor.")

    # Update session state and clear cache if changed
    if selected_doc_extractor != st.session_state.selected_doc_extractor:
        st.session_state.selected_doc_extractor = selected_doc_extractor
        if 'legal_events_df' in st.session_state:
            del st.session_state.legal_events_df

    return selected_doc_extractor


def create_provider_selection():
    """Create provider selection: Dynamically generated from event extractor catalog"""
    # Styled section header
    st.markdown('<div class="section-header">🤖 Event Extraction Provider</div>', unsafe_allow_html=True)

    # Import event extractor catalog
    from src.core.event_extractor_catalog import get_event_extractor_catalog

    catalog = get_event_extractor_catalog()
    enabled_providers = catalog.list_extractors(enabled=True)

    if not enabled_providers:
        st.error("❌ No event extractors available. Check catalog configuration.")
        return 'langextract', None

    # Initialize session state
    default_provider = os.getenv('EVENT_EXTRACTOR', 'openrouter').lower()
    if 'selected_provider' not in st.session_state:
        st.session_state.selected_provider = default_provider

    # Separate recommended (primary) providers from advanced providers
    # Exclude 'google' provider (internal use only - accessed via langextract model selector)
    visible_providers = [p for p in enabled_providers if p.provider_id != 'google']
    primary_providers = [p for p in visible_providers if p.recommended]
    advanced_providers = [p for p in visible_providers if not p.recommended]

    # If fewer than 3 recommended, fill with advanced providers
    if len(primary_providers) < 3:
        needed = 3 - len(primary_providers)
        primary_providers.extend(advanced_providers[:needed])
        advanced_providers = advanced_providers[needed:]

    # Dynamic provider icon mapping
    provider_icons = {
        'openrouter': '🌐',
        'langextract': '🔮',
        'openai': '🧠',
        'anthropic': '🎯',
        'deepseek': '🔍',
        'opencode_zen': '⚖️'
    }

    # Map provider_id to MODEL_CATALOG provider name for model selectors
    model_catalog_map = {
        'langextract': 'google',
        'openrouter': 'openrouter',
        'openai': 'openai',
        'anthropic': 'anthropic',
        'deepseek': 'deepseek'
    }

    # API key environment variable and URL mapping
    api_key_map = {
        'langextract': ('GEMINI_API_KEY', 'https://aistudio.google.com/app/apikey'),
        'openrouter': ('OPENROUTER_API_KEY', 'https://openrouter.ai/keys'),
        'openai': ('OPENAI_API_KEY', 'https://platform.openai.com/api-keys'),
        'anthropic': ('ANTHROPIC_API_KEY', 'https://console.anthropic.com/'),
        'deepseek': ('DEEPSEEK_API_KEY', 'https://platform.deepseek.com/'),
        'opencode_zen': ('OPENCODEZEN_API_KEY', None)
    }

    # === PRIMARY: Top recommended providers side-by-side ===
    num_primary = min(len(primary_providers), 3)
    if num_primary > 0:
        cols = st.columns(num_primary)

        for idx, provider_entry in enumerate(primary_providers):
            with cols[idx]:
                provider_id = provider_entry.provider_id
                is_configured = check_provider_status(provider_id)
                is_selected = st.session_state.selected_provider == provider_id

                # Get icon and display name from mapping
                icon = provider_icons.get(provider_id, '🤖')

                if st.button(
                    f"{icon} {provider_entry.display_name}",
                    key=f"provider_{provider_id}",
                    use_container_width=True,
                    type="primary" if is_selected else "secondary"
                ):
                    if st.session_state.selected_provider != provider_id:
                        st.session_state.selected_provider = provider_id
                        if 'legal_events_df' in st.session_state:
                            del st.session_state.legal_events_df

                # Show notes/subtitle from catalog (first sentence only for brevity)
                note_parts = provider_entry.notes.split('.')
                st.caption(note_parts[0] + '.' if note_parts else provider_entry.notes)

                # Status badge
                if is_configured:
                    st.markdown('<span class="status-pill ready">✓ Ready</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="status-pill setup">⚠ Setup Required</span>', unsafe_allow_html=True)

        # Show model selector if primary provider is selected
        selected_provider = st.session_state.selected_provider
        if selected_provider in [p.provider_id for p in primary_providers]:
            provider_entry = catalog.get_extractor(selected_provider)

            if provider_entry:
                is_configured = check_provider_status(selected_provider)

                if is_configured and provider_entry.supports_runtime_model:
                    st.markdown("")  # Spacing
                    # Use custom Gemini selector for langextract provider
                    if selected_provider == 'langextract':
                        selected_model = create_gemini_model_selector()
                        return selected_provider, selected_model
                    else:
                        catalog_provider = model_catalog_map.get(selected_provider)
                        if catalog_provider:
                            selected_model = create_unified_model_selector(catalog_provider)
                            return selected_provider, selected_model
                elif is_configured and not provider_entry.supports_runtime_model:
                    # Provider doesn't support runtime model (uses default)
                    return selected_provider, None
                elif not is_configured:
                    # Show setup instructions
                    env_key, url = api_key_map.get(selected_provider, (None, None))
                    if env_key:
                        setup_msg = f"💡 **Setup**: Add `{env_key}` to your `.env` file, then restart."
                        if url:
                            setup_msg += f"\n\nGet API key: {url}"
                        st.info(setup_msg)
                    return selected_provider, None

    # === SECONDARY: Advanced providers (in expander) ===
    if advanced_providers:
        st.divider()
        with st.expander("🔧 Direct Provider APIs (Advanced)", expanded=False):
            st.caption("Additional providers with direct API access")

            # Show in 2-column grid
            num_cols = 2
            rows_needed = (len(advanced_providers) + num_cols - 1) // num_cols

            for row_idx in range(rows_needed):
                row_cols = st.columns(num_cols)

                for col_idx in range(num_cols):
                    provider_idx = row_idx * num_cols + col_idx
                    if provider_idx < len(advanced_providers):
                        provider_entry = advanced_providers[provider_idx]
                        provider_id = provider_entry.provider_id

                        with row_cols[col_idx]:
                            is_configured = check_provider_status(provider_id)
                            is_selected = st.session_state.selected_provider == provider_id

                            # Get icon
                            icon = provider_icons.get(provider_id, '🤖')

                            if st.button(
                                f"{icon} {provider_entry.display_name}",
                                key=f"provider_adv_{provider_id}",
                                use_container_width=True,
                                type="primary" if is_selected else "secondary"
                            ):
                                if st.session_state.selected_provider != provider_id:
                                    st.session_state.selected_provider = provider_id
                                    if 'legal_events_df' in st.session_state:
                                        del st.session_state.legal_events_df

                            # Show notes (first sentence)
                            note_parts = provider_entry.notes.split('.')
                            st.caption(note_parts[0] + '.' if note_parts else provider_entry.notes)

                            # Status badge
                            badge = "✅" if is_configured else "⚠️ Setup Required"
                            st.caption(badge)

        # Show model selectors for advanced providers if selected
        selected_provider = st.session_state.selected_provider
        if selected_provider in [p.provider_id for p in advanced_providers]:
            provider_entry = catalog.get_extractor(selected_provider)

            if provider_entry and check_provider_status(selected_provider):
                if provider_entry.supports_runtime_model:
                    st.markdown("")  # Spacing
                    # Use custom Gemini selector for langextract provider
                    if selected_provider == 'langextract':
                        return selected_provider, create_gemini_model_selector()
                    else:
                        catalog_provider = model_catalog_map.get(selected_provider)
                        if catalog_provider:
                            return selected_provider, create_unified_model_selector(catalog_provider)
                else:
                    # Provider doesn't support runtime model (e.g., opencode_zen)
                    return selected_provider, None

    # No provider selected or not configured - fallback
    return st.session_state.selected_provider, None


def create_polished_metrics(legal_events_df):
    """Create polished metrics cards with gradients and icons"""
    from src.core.constants import FIVE_COLUMN_HEADERS

    # Check if classification is enabled (Document Type column exists)
    has_classification = 'Document Type' in legal_events_df.columns
    num_cols = 5 if has_classification else 4

    if has_classification:
        col1, col2, col3, col4, col5 = st.columns(num_cols)
    else:
        col1, col2, col3, col4 = st.columns(num_cols)

    with col1:
        total_events = len(legal_events_df)
        st.markdown(
            f"""
            <div class="metric-card success">
                <div class="metric-label">📊 Total Events</div>
                <div class="metric-value">{total_events}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        unique_docs = legal_events_df[FIVE_COLUMN_HEADERS[4]].nunique()
        st.markdown(
            f"""
            <div class="metric-card info">
                <div class="metric-label">📄 Documents</div>
                <div class="metric-value">{unique_docs}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        # Count events with real citations
        citations_count = len(legal_events_df[
            (~legal_events_df[FIVE_COLUMN_HEADERS[3]].str.contains("No citation available", na=False)) &
            (~legal_events_df[FIVE_COLUMN_HEADERS[3]].str.contains("processing failed", na=False))
        ])
        st.markdown(
            f"""
            <div class="metric-card warning">
                <div class="metric-label">⚖️ Citations</div>
                <div class="metric-value">{citations_count}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        avg_chars = legal_events_df[FIVE_COLUMN_HEADERS[2]].str.len().mean()
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">📝 Avg Detail</div>
                <div class="metric-value">{avg_chars:.0f}</div>
                <div class="metric-label" style="font-size: 0.63rem;">characters</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Add 5th column for Document Types (when classification is enabled)
    if has_classification:
        with col5:
            unique_types = legal_events_df['Document Type'].nunique()
            st.markdown(
                f"""
                <div class="metric-card info">
                    <div class="metric-label">🏷️ Doc Types</div>
                    <div class="metric-value">{unique_types}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


def create_file_upload_section():
    """Create the file upload section"""
    st.subheader("Document Upload")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'docx', 'txt', 'msg', 'eml', 'jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX, TXT, MSG, EML, JPG, JPEG, PNG (images require OCR)"
    )

    if uploaded_files:
        file_handler = FileHandler()

        # Create dataframe for file summary with size-based warnings
        file_data = []
        large_files = []
        very_large_files = []

        for file in uploaded_files:
            file_info = file_handler.get_file_info(file)
            size_mb = file_info['size_mb']

            # Categorize by size and support status
            if not file_info['supported']:
                status = "❌ Unsupported"
            elif size_mb > 15:
                very_large_files.append((file_info['name'], size_mb))
                status = "⚠️ Very Large"
            elif size_mb > 5:
                large_files.append((file_info['name'], size_mb))
                status = "⏱️ Large"
            else:
                status = "✅ Ready"

            file_data.append({
                "File": file_info['name'],
                "Size MB": f"{size_mb:.2f}",
                "Status": status
            })

        # Show warnings before table based on file sizes
        if very_large_files:
            file_list = ", ".join([f"**{name}** ({size:.1f}MB)" for name, size in very_large_files])

            # Check if any are images (require OCR)
            image_files = [name for name, _ in very_large_files if name.lower().endswith(('.jpg', '.jpeg', '.png'))]

            if image_files:
                st.warning(
                    f"⚠️ **Very large images detected:** {file_list}\n\n"
                    f"Image OCR may take **60-120 seconds per file**. "
                    f"Consider reducing image resolution before upload for faster processing."
                )
            else:
                st.warning(
                    f"⚠️ **Very large files detected:** {file_list}\n\n"
                    f"Processing may take **60-120 seconds per file**. Please be patient."
                )
        elif large_files:
            file_list = ", ".join([f"**{name}** ({size:.1f}MB)" for name, size in large_files])

            # Check if any are images (require OCR)
            image_files = [name for name, _ in large_files if name.lower().endswith(('.jpg', '.jpeg', '.png'))]

            if image_files:
                st.info(
                    f"⏱️ **Large images detected:** {file_list}\n\n"
                    f"Image OCR estimated time: **30-60 seconds per file**. High-quality screenshots process faster than phone photos."
                )
            else:
                st.info(
                    f"⏱️ **Large files detected:** {file_list}\n\n"
                    f"Estimated processing time: **30-60 seconds per file**."
                )

        # Collapsible file details section
        with st.expander(f"📄 File Details ({len(uploaded_files)} file{'s' if len(uploaded_files) > 1 else ''})", expanded=True):
            import pandas as pd
            summary_df = pd.DataFrame(file_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

    return uploaded_files


def main():
    """Main Streamlit application"""

    # Header
    st.markdown('<h1 class="main-header">Legal Events Extraction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-caption">Document processing with flexible extractors (Docling/Qwen-VL + AI providers)</p>', unsafe_allow_html=True)

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_files = create_file_upload_section()

    with col2:
        st.subheader("Configuration")

        # Document extractor selection
        selected_doc_extractor = create_doc_extractor_selection()

        st.divider()

        # Enhanced provider selection with runtime model support
        selected_provider, selected_model = create_provider_selection()

        st.divider()

        # === CLASSIFICATION LAYER (Layer 1.5) ===
        st.markdown('<div class="section-header">🏷️ Document Classification (Optional)</div>', unsafe_allow_html=True)

        # Auto-enable if OPENROUTER_API_KEY is configured
        classification_available = bool(os.getenv('OPENROUTER_API_KEY'))

        enable_classification = st.checkbox(
            "Enable Document Classification",
            value=classification_available,
            help="Classify documents and add Document Type as 6th column. Enables downstream filtering in Excel/database tools."
        )

        if enable_classification:
            classification_model, classification_prompt = create_classification_config()
        else:
            classification_model, classification_prompt = None, None

        st.divider()

        # Pipeline status visualization - shows the complete data flow
        st.markdown('<div class="section-header">📊 Pipeline Status</div>', unsafe_allow_html=True)
        create_pipeline_status(selected_doc_extractor, selected_provider, selected_model)

        # Pipeline configuration JSON (collapsible)
        create_pipeline_config_display(selected_doc_extractor, selected_provider, selected_model)

        st.divider()

        # Document extraction cache indicator
        from src.ui.streamlit_common import get_cache_stats, clear_document_cache
        cache_size, cached_files = get_cache_stats()
        if cache_size > 0:
            st.caption(f"💾 **Docling Cache**: {cache_size} file{'s' if cache_size != 1 else ''}")
            with st.expander("ℹ️ Cache Info", expanded=False):
                st.caption("Cached files (switching LLMs reuses extraction):")
                for idx, filename in enumerate(cached_files, 1):
                    st.caption(f"  {idx}. {filename}")
                if st.button("Clear Cache", key="clear_cache_btn"):
                    clear_document_cache()
                    st.rerun()

        st.divider()

        # Quick sample document button
        st.markdown('<div class="section-header">⚡ Quick Test</div>', unsafe_allow_html=True)
        if st.button("📄 Try Sample Document", use_container_width=True, help="Load the Famas arbitration PDF for testing"):
            from pathlib import Path
            sample_path = Path("sample_pdf/famas_dispute/Transaction_Fee_Invoice.pdf")

            if sample_path.exists():
                # Store sample file info in session state
                st.session_state['use_sample'] = True
                st.success("✅ Sample document loaded! Click **Process Files** below.")
            else:
                st.error(f"❌ Sample file not found at: {sample_path}")

        st.divider()

        # Determine which files to process (uploaded or sample)
        files_to_process = uploaded_files

        # Handle sample document if requested
        if st.session_state.get('use_sample', False) and not uploaded_files:
            from pathlib import Path
            sample_path = Path("sample_pdf/famas_dispute/Transaction_Fee_Invoice.pdf")

            if sample_path.exists():
                # Read sample file and create a file-like object
                import io
                sample_bytes = sample_path.read_bytes()
                sample_file = io.BytesIO(sample_bytes)
                sample_file.name = sample_path.name
                sample_file.seek(0)  # Reset position to start for subsequent reads
                files_to_process = [sample_file]
                st.info(f"📄 Ready to process: **{sample_path.name}** ({len(sample_bytes) / 1024:.0f} KB)")

        if files_to_process:
            # Show cost estimate WITHOUT extraction (uses page count estimation)
            from src.ui.streamlit_common import display_cost_estimates

            # Show cost estimates WITHOUT running extraction
            # display_cost_estimates handles extracted_texts=None by using page count heuristics
            display_cost_estimates(
                extracted_texts=None,  # Don't extract yet - show estimate first
                provider=selected_provider,
                runtime_model=selected_model,
                show_all_models=False,  # Show specific model estimate only
                uploaded_files=files_to_process,  # For Layer 1 cost calculation
                doc_extractor=selected_doc_extractor  # For Layer 1 cost calculation
            )

            # Optional: Calculate exact costs using tiktoken
            st.markdown("---")
            col1, col2 = st.columns([3, 1])
            with col1:
                use_tiktoken = st.checkbox(
                    "🔬 Calculate exact token counts (requires document extraction)",
                    help="Extract documents with Docling first, then use OpenAI's tiktoken for precise token counting. "
                         "More accurate (±2% variance) but requires ~3-5 seconds for extraction.",
                    key="tiktoken_exact_calc"
                )

            if use_tiktoken:
                if st.button("📊 Calculate Exact Costs", type="secondary", use_container_width=True, key="exact_calc_btn"):
                    with st.spinner("📄 Extracting documents for exact token counting..."):
                        try:
                            # Stage 1: Extract documents using Docling (free)
                            stage1_start = time.perf_counter()

                            pipeline = LegalEventsPipeline(
                                event_extractor=selected_provider,
                                runtime_model=selected_model if selected_model else None,
                                doc_extractor=selected_doc_extractor
                            )

                            # Log configuration for debugging
                            ocr_enabled = os.getenv('DOCLING_DO_OCR', 'false').lower() == 'true'
                            logger.info(
                                f"🔬 Tiktoken exact cost calculation started: "
                                f"doc_extractor={selected_doc_extractor}, "
                                f"files={len(files_to_process)}, "
                                f"OCR={'enabled' if ocr_enabled else 'disabled'}"
                            )

                            extracted_texts = []
                            extraction_stats = []  # Track per-file stats for logging

                            for file_obj in files_to_process:
                                try:
                                    # Get file metadata for better error reporting
                                    file_size_mb = len(file_obj.getbuffer()) / (1024 * 1024) if hasattr(file_obj, 'getbuffer') else 0
                                    file_ext = Path(file_obj.name).suffix.lower()

                                    logger.info(f"📄 Extracting {file_obj.name} ({file_size_mb:.2f} MB, {file_ext})")

                                    doc_result = pipeline.document_extractor.extract(file_obj)
                                    if doc_result and doc_result.plain_text:
                                        char_count = len(doc_result.plain_text)
                                        extracted_texts.append(doc_result.plain_text)
                                        extraction_stats.append({
                                            'name': file_obj.name,
                                            'chars': char_count,
                                            'status': 'success'
                                        })
                                        logger.info(f"✅ Extracted {char_count:,} chars from {file_obj.name}")
                                    else:
                                        # Empty extraction (not an error, but noteworthy)
                                        logger.warning(
                                            f"⚠️ No text extracted from {file_obj.name} "
                                            f"({file_size_mb:.2f} MB, {file_ext}, extractor: {selected_doc_extractor})"
                                        )
                                        extraction_stats.append({
                                            'name': file_obj.name,
                                            'chars': 0,
                                            'status': 'empty'
                                        })

                                except Exception as e:
                                    logger.error(
                                        f"❌ Extraction failed: {file_obj.name} "
                                        f"({file_size_mb:.2f} MB, {file_ext}, extractor: {selected_doc_extractor}) "
                                        f"Error: {str(e)}"
                                    )
                                    extraction_stats.append({
                                        'name': file_obj.name,
                                        'chars': 0,
                                        'status': 'error',
                                        'error': str(e)
                                    })

                            # Log Stage 1 completion
                            stage1_time = time.perf_counter() - stage1_start
                            successful_extractions = sum(1 for s in extraction_stats if s['status'] == 'success')
                            total_chars = sum(s['chars'] for s in extraction_stats)
                            logger.info(
                                f"⏱️  Stage 1 complete: {successful_extractions}/{len(files_to_process)} files extracted "
                                f"in {stage1_time:.2f}s ({total_chars:,} chars total)"
                            )

                            if extracted_texts:
                                # Stage 2: Calculate exact costs using tiktoken
                                stage2_start = time.perf_counter()

                                with st.spinner("🔬 Counting tokens with tiktoken..."):
                                    cost_table = estimate_all_models_with_tiktoken(
                                        extracted_texts=extracted_texts,
                                        output_ratio=0.10  # 10% output token ratio
                                    )

                                # Log Stage 2 completion
                                stage2_time = time.perf_counter() - stage2_start
                                total_time = stage1_time + stage2_time
                                logger.info(
                                    f"⏱️  Stage 2 complete: {len(cost_table or [])} models calculated "
                                    f"in {stage2_time:.2f}s | Total: {total_time:.2f}s "
                                    f"(extraction: {stage1_time:.2f}s, tiktoken: {stage2_time:.2f}s)"
                                )

                                # Stage 3: Display exact cost table
                                if cost_table:
                                    st.success("✅ Exact token counts calculated!")

                                    # Create DataFrame for display
                                    import pandas as pd
                                    df_costs = pd.DataFrame(cost_table)

                                    # Filter to show most relevant columns
                                    display_cols = ['model_id', 'input_tokens', 'output_tokens', 'input_cost', 'output_cost', 'total_cost']
                                    df_display = df_costs[[col for col in display_cols if col in df_costs.columns]]

                                    st.dataframe(
                                        df_display,
                                        use_container_width=True,
                                        hide_index=True,
                                        column_config={
                                            "model_id": st.column_config.TextColumn("Model", width="medium"),
                                            "input_tokens": st.column_config.NumberColumn("Input Tokens", format="%d"),
                                            "output_tokens": st.column_config.NumberColumn("Output Tokens", format="%d"),
                                            "input_cost": st.column_config.NumberColumn("Input Cost", format="$%.6f"),
                                            "output_cost": st.column_config.NumberColumn("Output Cost", format="$%.6f"),
                                            "total_cost": st.column_config.NumberColumn("Total Cost", format="$%.6f"),
                                        }
                                    )

                                    # Log cost calculation summary
                                    total_input_tokens = sum(model.get('input_tokens', 0) for model in cost_table)
                                    total_output_tokens = sum(model.get('output_tokens', 0) for model in cost_table)
                                    costs = [model.get('total_cost', 0) for model in cost_table]
                                    min_cost = min(costs) if costs else 0
                                    max_cost = max(costs) if costs else 0
                                    cheapest_model = next((m['model_id'] for m in cost_table if m.get('total_cost') == min_cost), 'N/A')
                                    most_expensive = next((m['model_id'] for m in cost_table if m.get('total_cost') == max_cost), 'N/A')

                                    logger.info(
                                        f"💰 Cost calculation summary: "
                                        f"{len(cost_table)} models, "
                                        f"tokens: {total_input_tokens:,} input / {total_output_tokens:,} output, "
                                        f"cost range: ${min_cost:.6f} ({cheapest_model}) "
                                        f"to ${max_cost:.6f} ({most_expensive})"
                                    )

                                    st.info(
                                        "💡 **Exact Cost Calculation**: These costs are based on actual document token counts "
                                        "using OpenAI's official tiktoken library. Accuracy: ±2% vs actual API billing."
                                    )
                                else:
                                    logger.error(
                                        f"❌ Tiktoken cost calculation returned empty results "
                                        f"(texts={len(extracted_texts)}, table_size={len(cost_table or [])})"
                                    )
                                    st.error("Failed to calculate exact costs. Please try again.")

                                    with st.expander("🔍 Debug Details"):
                                        st.write(f"**Extracted Texts:** {len(extracted_texts)}")
                                        st.write(f"**Models Attempted:** {len(cost_table or [])}")
                                        st.write("Check application logs for model-specific errors.")
                            else:
                                # Enhanced empty results error with debug information
                                error_msg = (
                                    f"❌ No text extracted from ANY files "
                                    f"({len(files_to_process)} files, extractor: {selected_doc_extractor}). "
                                    f"Possible causes: corrupted files, unsupported formats, or OCR failure."
                                )
                                logger.error(error_msg)
                                st.error("No text could be extracted from the uploaded files.")

                                # Show detailed breakdown in expander
                                with st.expander("🔍 Debug Details"):
                                    st.write(f"**Document Extractor:** {selected_doc_extractor}")
                                    st.write(f"**Files Attempted:** {len(files_to_process)}")
                                    st.write(f"**OCR Enabled:** {ocr_enabled}")
                                    st.write("**Per-File Results:**")
                                    for idx, stats in enumerate(extraction_stats, 1):
                                        status_emoji = "✅" if stats['status'] == 'success' else "❌" if stats['status'] == 'error' else "⚠️"
                                        if stats['status'] == 'success':
                                            st.write(f"  {status_emoji} {stats['name']}: {stats['chars']:,} chars")
                                        elif stats['status'] == 'empty':
                                            st.write(f"  {status_emoji} {stats['name']}: No text (possibly blank or image-based)")
                                        else:
                                            st.write(f"  {status_emoji} {stats['name']}: {stats.get('error', 'Unknown error')}")

                        except Exception as e:
                            logger.error(f"❌ Exact cost calculation failed with exception: {str(e)}", exc_info=True)
                            st.error("An unexpected error occurred during cost calculation. Check application logs for details.")

                            with st.expander("🔍 Error Details"):
                                st.write(f"**Error Type:** {type(e).__name__}")
                                st.write(f"**Error Message:** {str(e)}")
                                st.write("Check application logs for full stack trace.")

            if st.button("Process Files", type="primary", use_container_width=True):
                # Enhanced status container with context
                status_container = st.empty()

                # Get provider display name
                provider_names = {
                    'langextract': 'LangExtract (Google Gemini)',
                    'openrouter': 'OpenRouter',
                    'opencode_zen': 'OpenCode Zen',
                    'openai': 'OpenAI',
                    'anthropic': 'Anthropic',
                    'deepseek': 'DeepSeek'
                }
                provider_display = provider_names.get(
                    selected_provider,
                    selected_provider
                )

                # Add model name to display if OpenRouter
                if selected_provider == 'openrouter' and selected_model:
                    # Extract friendly model name
                    model_parts = selected_model.split('/')
                    model_short = model_parts[-1] if len(model_parts) > 1 else selected_model
                    provider_display = f"{provider_display} ({model_short})"

                # Calculate expected time based on file sizes
                file_handler = FileHandler()
                total_size = sum(file_handler.get_file_info(f)['size_mb'] for f in files_to_process)
                file_count = len(files_to_process)

                # === LAYER 1.5: CLASSIFICATION (if enabled) ===
                # Run classification on ALL documents, store as metadata for 6th column
                classification_lookup = {}  # {filename: document_type}

                if enable_classification and classification_model:
                    # Show classification status
                    with status_container:
                        st.info(f"🏷️ **Classifying {file_count} document{'s' if file_count > 1 else ''}**...")

                    try:
                        # Import classification factory
                        from src.core.classification_factory import create_classifier
                        import tempfile
                        from pathlib import Path

                        # Create classifier adapter
                        classifier = create_classifier(classification_model, classification_prompt)

                        # Get pipeline for document extraction (Layer 1)
                        pipeline = get_pipeline(
                            provider=selected_provider,
                            runtime_model=selected_model,
                            doc_extractor=selected_doc_extractor
                        )

                        if pipeline is None:
                            st.error("❌ Failed to initialize pipeline for classification")
                            status_container.empty()
                        else:
                            # Extract and classify all documents
                            classifications = []
                            with tempfile.TemporaryDirectory() as temp_dir:
                                temp_path = Path(temp_dir)

                                with st.spinner(f"📊 Extracting and classifying {file_count} document{'s' if file_count > 1 else ''}..."):
                                    for idx, file in enumerate(files_to_process, 1):
                                        try:
                                            # Reset file position
                                            if hasattr(file, 'seek'):
                                                file.seek(0)

                                            # Save and extract (uses cache)
                                            file_path = file_handler.save_uploaded_file(file, temp_path)
                                            doc_result = pipeline.document_extractor.extract(file_path)

                                            if not doc_result or not doc_result.plain_text.strip():
                                                st.warning(f"⚠️ No text extracted from {file.name} - skipping classification")
                                                classification_lookup[file.name] = "Unknown"
                                                continue

                                            # Classify document
                                            classification_result = classifier.classify(
                                                doc_result.plain_text,
                                                document_title=file.name
                                            )

                                            # Store for 6th column
                                            classification_lookup[file.name] = classification_result['primary']

                                            classifications.append({
                                                'filename': file.name,
                                                'type': classification_result['primary'],
                                                'confidence': classification_result.get('confidence', 0.0),
                                                'all_labels': classification_result.get('classes', [])
                                            })

                                            logger.info(
                                                f"✅ Classified {file.name}: {classification_result['primary']} "
                                                f"(confidence={classification_result.get('confidence', 0):.2f})"
                                            )

                                        except Exception as e:
                                            logger.error(f"❌ Classification failed for {file.name}: {e}")
                                            st.warning(f"⚠️ Classification failed for {file.name}: {str(e)}")
                                            classification_lookup[file.name] = "Classification Failed"

                            # Clear status
                            status_container.empty()

                            # Defensive logging: Check classification results
                            if len(classification_lookup) == 0:
                                logger.warning(
                                    f"⚠️ Classification completed but no files were classified. "
                                    f"Processed {file_count} files but classification_lookup is empty."
                                )
                            else:
                                logger.info(
                                    f"✅ Classification completed: {len(classification_lookup)} files classified "
                                    f"out of {file_count} processed"
                                )

                            # Show classification results (collapsible to reduce clutter)
                            if classifications:
                                st.divider()
                                with st.expander("📊 Classification Results", expanded=False):
                                    show_classification_results(classifications)
                                st.divider()

                    except Exception as e:
                        logger.error(f"❌ Classification layer failed: {e}")
                        st.error(f"🚨 Classification failed: {str(e)}\n\nProceeding without classification...")

                # === LAYER 2: EVENT EXTRACTION ===
                if files_to_process:
                    # Show processing status
                    with status_container:
                        if total_size > 15:
                            time_estimate = "⏱️ This may take 1-2 minutes..."
                        elif total_size > 5:
                            time_estimate = "⏱️ Estimated time: 30-60 seconds..."
                        else:
                            time_estimate = "⚡ Processing..."

                        st.info(
                            f"🔄 **Processing {len(files_to_process)} file{'s' if len(files_to_process) > 1 else ''}** via {provider_display}\n\n"
                            f"{time_estimate}"
                        )

                    # Process using shared utilities with provider and runtime model
                    # Pass runtime_model for ALL providers (not just OpenRouter)
                    legal_events_df = process_documents_with_spinner(
                        files_to_process,
                        show_subheader=False,
                        provider=selected_provider,
                        runtime_model=selected_model,  # Now works for all providers
                        doc_extractor=selected_doc_extractor
                    )

                    if legal_events_df is not None:
                        # === ADD CLASSIFICATION AS 6TH COLUMN (if enabled) ===
                        if enable_classification and len(classification_lookup) > 0:
                            from src.core.constants import FIVE_COLUMN_HEADERS

                            # Add Document Type column by mapping Document Reference to classification
                            legal_events_df['Document Type'] = legal_events_df[FIVE_COLUMN_HEADERS[4]].map(
                                classification_lookup
                            )

                            # Fill any missing values (shouldn't happen, but defensive)
                            legal_events_df['Document Type'] = legal_events_df['Document Type'].fillna('Unknown')

                            logger.info(f"✅ Added Document Type column with {len(classification_lookup)} classifications")
                        elif enable_classification and len(classification_lookup) == 0:
                            # Classification was enabled but no files were classified
                            logger.warning("⚠️ Classification enabled but no files were classified - column not added")
                            st.warning(
                                "⚠️ **Classification Enabled But No Results**\n\n"
                                "Classification was enabled but no documents were classified. "
                                "The Document Type column was not added.\n\n"
                                "**Possible causes**:\n"
                                "- Files may have been cleared before processing\n"
                                "- Classification pipeline initialization failed\n"
                                "- Check logs for classification errors"
                            )

                        # Store results in session state
                        st.session_state.legal_events_df = legal_events_df

                        # === SAVE METADATA JSON (for analytics ingestion) ===
                        if 'metadata' in legal_events_df.attrs:
                            try:
                                import json
                                from pathlib import Path

                                metadata_dict = legal_events_df.attrs['metadata']
                                output_dir = Path("output")
                                output_dir.mkdir(exist_ok=True)

                                metadata_file = output_dir / f"{metadata_dict['run_id']}_metadata.json"
                                with open(metadata_file, 'w') as f:
                                    json.dump(metadata_dict, f, indent=2)

                                logger.info(f"📊 Metadata saved: {metadata_file}")
                                # st.success(f"📊 Metadata saved: `{metadata_file.name}`")  # Optional: uncomment for user visibility
                            except Exception as e:
                                logger.error(f"⚠️ Failed to save metadata: {e}")

                        # Clear sample document flag after processing
                        if 'use_sample' in st.session_state:
                            del st.session_state['use_sample']

                        # Clear processing status
                        status_container.empty()
                    else:
                        # Clear processing status (error shown by shared utility)
                        status_container.empty()
                else:
                    st.warning("⚠️ No files to process.")
        else:
            st.info("👆 Upload files or try the sample document")

    # Results section - Guardrailed Five-Column Table
    if 'legal_events_df' in st.session_state:
        st.divider()

        legal_events_df = st.session_state.legal_events_df

        # Display polished metrics
        st.markdown("### 📈 Results Summary")
        create_polished_metrics(legal_events_df)

        st.markdown("")  # Spacing

        # Display table (supports optional 6th column: Document Type)
        st.markdown("### 📋 Legal Events Table")
        from src.core.constants import FIVE_COLUMN_HEADERS

        # Build column config dynamically
        column_config = {
            FIVE_COLUMN_HEADERS[0]: st.column_config.NumberColumn(FIVE_COLUMN_HEADERS[0], width="small"),
            FIVE_COLUMN_HEADERS[1]: st.column_config.TextColumn(FIVE_COLUMN_HEADERS[1], width="medium"),
            FIVE_COLUMN_HEADERS[2]: st.column_config.TextColumn(FIVE_COLUMN_HEADERS[2], width="large"),
            FIVE_COLUMN_HEADERS[3]: st.column_config.TextColumn(FIVE_COLUMN_HEADERS[3], width="medium"),
            FIVE_COLUMN_HEADERS[4]: st.column_config.TextColumn(FIVE_COLUMN_HEADERS[4], width="medium")
        }

        # Add Document Type column config if present
        if 'Document Type' in legal_events_df.columns:
            column_config['Document Type'] = st.column_config.TextColumn('Document Type', width="medium")
            st.caption("💡 Classification enabled - Document Type column shows document categories")

        st.dataframe(
            legal_events_df,
            use_container_width=True,
            hide_index=True,
            column_config=column_config
        )

        # Display performance timing metrics if available
        if "Docling_Seconds" in legal_events_df.columns and "Extractor_Seconds" in legal_events_df.columns:
            st.markdown("")  # Spacing
            st.markdown("### ⏱️ Performance Metrics")
            col1, col2, col3 = st.columns(3)

            with col1:
                avg_docling = legal_events_df["Docling_Seconds"].mean()
                st.metric("Avg Docling Time", f"{avg_docling:.3f}s")

            with col2:
                avg_extractor = legal_events_df["Extractor_Seconds"].mean()
                st.metric("Avg Extractor Time", f"{avg_extractor:.3f}s")

            with col3:
                avg_total = legal_events_df["Total_Seconds"].mean()
                st.metric("Avg Total Time", f"{avg_total:.3f}s")

        # Provide standardized downloads with provider context
        st.markdown("")  # Spacing
        provider = st.session_state.get('selected_provider', 'langextract')
        create_download_section(legal_events_df, provider=provider)

    # Footer
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            "<div style='text-align: center; color: #666; font-size: 0.85rem;'>"
            "Legal Events Extraction • Powered by Docling + AI"
            "</div>",
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()
