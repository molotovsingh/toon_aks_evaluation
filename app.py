#!/usr/bin/env python3
"""
STANDARDIZED Legal Events Extraction - Granite Guardrails Compliant
Unified Five-Column Table: Docling + LangExtract + Shared Utilities
"""

import streamlit as st
import os
import logging

# Load environment variables FIRST (before imports that need it)
from dotenv import load_dotenv
load_dotenv()

# Import refactored shared utilities for guard-railed five-column exports
from src.ui.streamlit_common import (
    get_pipeline,
    process_documents_with_spinner,
    create_download_section
)
from src.utils.file_handler import FileHandler

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
    font-size: 1.5rem;
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
    font-size: 0.85rem;
    color: #64748b;
    text-align: center;
    margin-bottom: 1rem;
    font-weight: 400;
}

/* === PROVIDER BUTTON ENHANCEMENTS === */
/* Style Streamlit buttons to look like modern cards */
div[data-testid="column"] > div > div > button {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%) !important;
    border: 2px solid #e2e8f0 !important;
    border-radius: 8px !important;
    padding: 0.75rem 0.75rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
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

/* Primary (selected) button styling */
div[data-testid="column"] > div > div > button[kind="primary"] {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    border: 2px solid #2563eb !important;
    color: white !important;
    box-shadow: 0 2px 8px rgba(59, 130, 246, 0.4) !important;
}

div[data-testid="column"] > div > div > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.5) !important;
}

/* OpenRouter specific styling (first column) */
div[data-testid="column"]:first-child button[kind="primary"] {
    background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
    border-color: #0284c7 !important;
    box-shadow: 0 2px 8px rgba(14, 165, 233, 0.4) !important;
}

div[data-testid="column"]:first-child button[kind="primary"]:hover {
    background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
    box-shadow: 0 4px 12px rgba(14, 165, 233, 0.5) !important;
}

/* Gemini specific styling (second column) */
div[data-testid="column"]:nth-child(2) button[kind="primary"] {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    border-color: #059669 !important;
    box-shadow: 0 2px 8px rgba(16, 185, 129, 0.4) !important;
}

div[data-testid="column"]:nth-child(2) button[kind="primary"]:hover {
    background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.5) !important;
}

/* === STATUS BADGES === */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.25rem 0.5rem;
    border-radius: 12px;
    font-size: 0.7rem;
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
    font-size: 0.75rem;
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
    font-size: 1.75rem;
    font-weight: 700;
    margin: 0.25rem 0;
    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.metric-label {
    font-size: 0.75rem;
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
    font-size: 0.8rem;
}
div[data-testid="stCaptionContainer"] {
    color: #64748b !important;
    font-size: 0.75rem !important;
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
    font-size: 0.75rem;
    color: #64748b;
    vertical-align: middle;
}
.pipeline-stage-header {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #475569;
    margin-bottom: 0.25rem;
}
.pipeline-stage-value {
    font-size: 0.85rem;
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
    font-size: 1.2rem;
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
        doc_extractor: 'docling' or 'gemini'
        provider: 'openrouter', 'langextract', 'openai', 'anthropic', 'deepseek', 'opencode_zen'
        model: Selected model identifier (optional)
    """
    # Stage 1: Document Processing
    doc_display_map = {
        'docling': '🔧 Docling<br><span style="font-size:0.7rem;">Local OCR</span>',
        'gemini': '🌟 Gemini 2.5<br><span style="font-size:0.7rem;">Cloud Vision</span>'
    }
    doc_display = doc_display_map.get(doc_extractor, doc_extractor)

    # Stage 2: Event Extraction Provider
    provider_display_map = {
        'langextract': '🔮 Gemini<br><span style="font-size:0.7rem;">LangExtract</span>',
        'openrouter': '🌐 OpenRouter<br><span style="font-size:0.7rem;">18+ Models</span>',
        'opencode_zen': '⚖️ OpenCode Zen<br><span style="font-size:0.7rem;">Legal AI</span>',
        'openai': '🧠 OpenAI<br><span style="font-size:0.7rem;">GPT-4o</span>',
        'anthropic': '🎯 Anthropic<br><span style="font-size:0.7rem;">Claude 3.5</span>',
        'deepseek': '🔍 DeepSeek<br><span style="font-size:0.7rem;">R1 Chat</span>'
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
            model_display = f'{model_obj.display_name}<br><span style="font-size:0.7rem;">{model_obj.cost_per_1m}</span>'
        else:
            model_display = f'{model_short}<br><span style="font-size:0.7rem;">Selected</span>'
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
        doc_extractor: 'docling' or 'gemini'
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
from dataclasses import dataclass
from typing import List

@dataclass
class ModelConfig:
    """Rich model metadata for unified selector"""
    provider: str          # 'google', 'anthropic', 'openai', 'openrouter', 'deepseek'
    model_id: str         # Backend model identifier
    display_name: str     # UI display name
    category: str         # 'Ground Truth', 'Production', 'Budget'
    cost_per_1m: str      # '$3/M' or 'Free'
    context_window: str   # '200K', '2M'
    quality_score: str = ""    # '10/10', '9/10', or ''
    badges: List[str] = None   # ['Tier 1', 'Fastest', 'Cheapest']

    def __post_init__(self):
        """Initialize default values"""
        if self.badges is None:
            self.badges = []

    def format_inline(self) -> str:
        """Format: Quality • Cost • Context • Badges"""
        parts = []
        if self.quality_score:
            parts.append(self.quality_score)
        parts.append(self.cost_per_1m)
        parts.append(self.context_window)
        if self.badges:
            parts.extend(self.badges)
        return ' • '.join(parts)


# Complete model catalog across all providers
MODEL_CATALOG = [
    # === ANTHROPIC ===
    ModelConfig("anthropic", "claude-sonnet-4-5", "Claude Sonnet 4.5",
                "Ground Truth", "$3/M", "200K", "10/10", ["Tier 1", "Recommended"]),
    ModelConfig("anthropic", "claude-opus-4", "Claude Opus 4",
                "Ground Truth", "$15/M", "200K", "10/10", ["Tier 3", "Highest Quality"]),
    ModelConfig("anthropic", "claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet",
                "Production", "$3/M", "200K", "10/10", []),
    ModelConfig("anthropic", "claude-3-haiku-20240307", "Claude 3 Haiku",
                "Production", "$0.25/M", "200K", "10/10", ["Fastest - 4.4s"]),

    # === OPENAI ===
    ModelConfig("openai", "gpt-5", "GPT-5",
                "Ground Truth", "$TBD", "128K", "", ["Tier 2", "Non-deterministic"]),
    ModelConfig("openai", "gpt-4o", "GPT-4o",
                "Production", "$2.50/M", "128K", "10/10", []),
    ModelConfig("openai", "gpt-4o-mini", "GPT-4o Mini",
                "Production", "$0.15/M", "128K", "9/10", []),

    # === GOOGLE (maps to langextract backend) ===
    ModelConfig("google", "gemini-2.5-pro", "Gemini 2.5 Pro",
                "Ground Truth", "$TBD", "2M", "", ["Tier 2", "Long Docs"]),
    ModelConfig("google", "gemini-2.0-flash", "Gemini 2.0 Flash",
                "Production", "Free", "1M", "9/10", []),

    # === OPENROUTER (18 curated models from Oct 2025 testing) ===
    ModelConfig("openrouter", "openai/gpt-4o-mini", "GPT-4o Mini",
                "Recommended", "$0.15/M", "128K", "9/10", ["Balanced"]),
    ModelConfig("openrouter", "deepseek/deepseek-r1-distill-llama-70b", "DeepSeek R1 Distill",
                "Budget", "$0.03/M", "128K", "10/10", ["Cheapest"]),
    ModelConfig("openrouter", "qwen/qwq-32b", "Qwen QwQ 32B",
                "Budget", "$0.115/M", "128K", "7/10", ["Ultra-cheap ⚠️"]),
    ModelConfig("openrouter", "anthropic/claude-3-haiku", "Claude 3 Haiku",
                "Budget", "$0.25/M", "200K", "10/10", ["4.4s ⚡"]),
    ModelConfig("openrouter", "deepseek/deepseek-chat", "DeepSeek Chat",
                "Budget", "$0.25/M", "128K", "10/10", ["Fast"]),
    ModelConfig("openrouter", "anthropic/claude-3-5-sonnet", "Claude 3.5 Sonnet",
                "Long Documents", "$3/M", "200K", "10/10", ["Max context"]),
    ModelConfig("openrouter", "openai/gpt-4o", "GPT-4o",
                "Maximum Quality", "$3/M", "128K", "10/10", ["OpenAI flagship"]),
    ModelConfig("openrouter", "meta-llama/llama-3.3-70b-instruct", "Llama 3.3 70B",
                "Open Source", "$0.60/M", "128K", "10/10", ["OSS"]),
    ModelConfig("openrouter", "mistralai/mistral-small", "Mistral Small",
                "Open Source", "$0.20/M", "128K", "10/10", ["EU compliance"]),

    # === DEEPSEEK ===
    ModelConfig("deepseek", "deepseek-chat", "DeepSeek Chat",
                "Production", "$0.25/M", "128K", "10/10", ["Fast"]),
]


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

    # Get default from env or use first model
    env_defaults = {
        'anthropic': os.getenv('ANTHROPIC_MODEL', filtered[0].model_id),
        'openai': os.getenv('OPENAI_MODEL', filtered[0].model_id),
        'google': os.getenv('GEMINI_MODEL_ID', filtered[0].model_id),
        'openrouter': os.getenv('OPENROUTER_MODEL', filtered[0].model_id),
        'deepseek': os.getenv('DEEPSEEK_MODEL', filtered[0].model_id)
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


def create_anthropic_model_selector():
    """Create Anthropic model selector with search/filter"""

    models = {
        "🏆 Ground Truth Models": [
            ("claude-sonnet-4-5", "Claude Sonnet 4.5", "$3/M • 200K", "Tier 1 • Best balance"),
            ("claude-opus-4", "Claude Opus 4", "$15/M • 200K", "Tier 3 • Highest quality"),
        ],
        "📊 Production Models": [
            ("claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet", "$3/M • 200K", "Quality baseline"),
            ("claude-3-haiku-20240307", "Claude 3 Haiku", "$0.25/M • 200K", "Budget option"),
        ],
    }

    # Flatten into searchable format with normalized search text
    all_models = []
    for category, model_list in models.items():
        category_emoji = category.split()[0]
        for model_id, display_name, cost, badge in model_list:
            search_text = f"{display_name} {model_id} {cost} {badge} {category}"
            all_models.append({
                "id": model_id,
                "display": f"{category_emoji} {display_name} • {cost} • {badge}",
                "search_text": search_text.lower(),
                "search_text_normalized": normalize_search_text(search_text)
            })

    # Search box
    search_query = st.text_input(
        "🔍 Search Models",
        placeholder="Type to filter (e.g., 'haiku', 'opus', 'ground truth', 'tier 1')...",
        key="anthropic_model_search"
    )

    # Filter models using normalized search
    if search_query:
        normalized_query = normalize_search_text(search_query)
        filtered_models = [m for m in all_models if normalized_query in m["search_text_normalized"]]
        st.caption(f"Found {len(filtered_models)} model(s)")
    else:
        filtered_models = all_models

    # Handle no results
    if not filtered_models:
        st.warning("No models match your search. Try different keywords.")
        return 'claude-3-haiku-20240307'

    # Create selectbox options
    options = [m["id"] for m in filtered_models]
    display_map = {m["id"]: m["display"] for m in filtered_models}

    # Session state management
    default_model = os.getenv('ANTHROPIC_MODEL', 'claude-3-haiku-20240307')
    if 'anthropic_model' not in st.session_state:
        st.session_state.anthropic_model = default_model

    if st.session_state.anthropic_model not in options:
        st.session_state.anthropic_model = options[0]

    selected_model = st.selectbox(
        "Select Model",
        options=options,
        index=options.index(st.session_state.anthropic_model),
        format_func=lambda x: display_map.get(x, x),
        help="Ground truth models for reference dataset creation. Production models for cost-effective daily use.",
        key="anthropic_model_selector"
    )

    if selected_model != st.session_state.anthropic_model:
        st.session_state.anthropic_model = selected_model
        if 'legal_events_df' in st.session_state:
            del st.session_state.legal_events_df

    return selected_model


def create_openai_model_selector():
    """Create OpenAI model selector with search/filter"""

    models = {
        "🏆 Ground Truth Models": [
            ("gpt-5", "GPT-5", "$TBD • 128K", "Tier 2 • Pending price"),
        ],
        "📊 Production Models": [
            ("gpt-4o", "GPT-4o", "$2.50/M • 128K", "Flagship model"),
            ("gpt-4o-mini", "GPT-4o Mini", "$0.15/M • 128K", "Budget option"),
        ],
    }

    # Flatten into searchable format with normalized search text
    all_models = []
    for category, model_list in models.items():
        category_emoji = category.split()[0]
        for model_id, display_name, cost, badge in model_list:
            search_text = f"{display_name} {model_id} {cost} {badge} {category}"
            all_models.append({
                "id": model_id,
                "display": f"{category_emoji} {display_name} • {cost} • {badge}",
                "search_text": search_text.lower(),
                "search_text_normalized": normalize_search_text(search_text)
            })

    # Search box
    search_query = st.text_input(
        "🔍 Search Models",
        placeholder="Type to filter (e.g., 'gpt 5', 'gpt-5', 'mini', 'ground truth')...",
        key="openai_model_search"
    )

    # Filter models using normalized search
    if search_query:
        normalized_query = normalize_search_text(search_query)
        filtered_models = [m for m in all_models if normalized_query in m["search_text_normalized"]]
        st.caption(f"Found {len(filtered_models)} model(s)")
    else:
        filtered_models = all_models

    # Handle no results
    if not filtered_models:
        st.warning("No models match your search. Try different keywords.")
        return 'gpt-4o-mini'

    # Create selectbox options
    options = [m["id"] for m in filtered_models]
    display_map = {m["id"]: m["display"] for m in filtered_models}

    # Session state management
    default_model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
    if 'openai_model' not in st.session_state:
        st.session_state.openai_model = default_model

    if st.session_state.openai_model not in options:
        st.session_state.openai_model = options[0]

    selected_model = st.selectbox(
        "Select Model",
        options=options,
        index=options.index(st.session_state.openai_model),
        format_func=lambda x: display_map.get(x, x),
        help="GPT-5 for ground truth creation (pricing TBD). ⚠️ Note: GPT-5 uses temperature=1.0 (non-deterministic) - outputs vary between runs. GPT-4o/mini for production use.",
        key="openai_model_selector"
    )

    if selected_model != st.session_state.openai_model:
        st.session_state.openai_model = selected_model
        if 'legal_events_df' in st.session_state:
            del st.session_state.legal_events_df

    return selected_model


def create_langextract_model_selector():
    """Create LangExtract/Gemini model selector with search/filter"""

    models = {
        "🏆 Ground Truth Models": [
            ("gemini-2.5-pro", "Gemini 2.5 Pro", "$TBD • 2M", "Tier 2 • Long docs"),
        ],
        "📊 Production Models": [
            ("gemini-2.0-flash", "Gemini 2.0 Flash", "Free • 1M", "Default model"),
        ],
    }

    # Flatten into searchable format with normalized search text
    all_models = []
    for category, model_list in models.items():
        category_emoji = category.split()[0]
        for model_id, display_name, cost, badge in model_list:
            search_text = f"{display_name} {model_id} {cost} {badge} {category}"
            all_models.append({
                "id": model_id,
                "display": f"{category_emoji} {display_name} • {cost} • {badge}",
                "search_text": search_text.lower(),
                "search_text_normalized": normalize_search_text(search_text)
            })

    # Search box
    search_query = st.text_input(
        "🔍 Search Models",
        placeholder="Type to filter (e.g., 'flash', 'pro', '2m context', 'gemini 2')...",
        key="langextract_model_search"
    )

    # Filter models using normalized search
    if search_query:
        normalized_query = normalize_search_text(search_query)
        filtered_models = [m for m in all_models if normalized_query in m["search_text_normalized"]]
        st.caption(f"Found {len(filtered_models)} model(s)")
    else:
        filtered_models = all_models

    # Handle no results
    if not filtered_models:
        st.warning("No models match your search. Try different keywords.")
        return 'gemini-2.0-flash'

    # Create selectbox options
    options = [m["id"] for m in filtered_models]
    display_map = {m["id"]: m["display"] for m in filtered_models}

    # Session state management
    default_model = os.getenv('GEMINI_MODEL_ID', 'gemini-2.0-flash')
    if 'langextract_model' not in st.session_state:
        st.session_state.langextract_model = default_model

    if st.session_state.langextract_model not in options:
        st.session_state.langextract_model = options[0]

    selected_model = st.selectbox(
        "Select Model",
        options=options,
        index=options.index(st.session_state.langextract_model),
        format_func=lambda x: display_map.get(x, x),
        help="Gemini 2.5 Pro for ground truth (2M context for very long documents). Gemini 2.0 Flash for daily use.",
        key="langextract_model_selector"
    )

    if selected_model != st.session_state.langextract_model:
        st.session_state.langextract_model = selected_model
        if 'legal_events_df' in st.session_state:
            del st.session_state.legal_events_df

    return selected_model


def create_openrouter_model_selector():
    """Create OpenRouter model selector with search/filter (scales to 100+ models)"""

    # Curated models from Oct 2025 testing (scripts/test_fallback_models.py + real doc benchmarks)
    # Only includes models that passed JSON mode tests (10/10 or 9/10 quality)
    # Organized by use case rather than price for better UX
    models = {
        "🎯 Recommended Starting Point": [
            ("openai/gpt-4o-mini", "GPT-4o Mini", "$0.15/M • 128K", "⭐ 9/10 • Balanced"),
        ],
        "💰 Budget Conscious": [
            ("deepseek/deepseek-r1-distill-llama-70b", "DeepSeek R1 Distill", "$0.03/M • 128K", "⭐ 10/10 • CHEAPEST"),
            ("qwen/qwq-32b", "Qwen QwQ 32B", "$0.115/M • 128K", "7/10 • Ultra-cheap ⚠️"),
            ("anthropic/claude-3-haiku", "Claude 3 Haiku", "$0.25/M • 200K", "10/10 • 4.4s ⚡"),
            ("deepseek/deepseek-chat", "DeepSeek Chat", "$0.25/M • 128K", "10/10 • Fast"),
        ],
        "📄 Long Documents (50+ pages)": [
            ("anthropic/claude-3-5-sonnet", "Claude 3.5 Sonnet", "$3/M • 200K", "⭐ 10/10 • Max context"),
            ("anthropic/claude-3-haiku", "Claude 3 Haiku", "$0.25/M • 200K", "10/10 • Budget + 200K"),
        ],
        "🏆 Maximum Quality": [
            ("anthropic/claude-3-5-sonnet", "Claude 3.5 Sonnet", "$3/M • 200K", "⭐ 10/10 • Premium"),
            ("openai/gpt-4o", "GPT-4o", "$3/M • 128K", "10/10 • OpenAI flagship"),
        ],
        "🌍 Open Source / EU Hosting": [
            ("meta-llama/llama-3.3-70b-instruct", "Llama 3.3 70B", "$0.60/M • 128K", "10/10 • OSS"),
            ("mistralai/mistral-small", "Mistral Small", "$0.20/M • 128K", "10/10 • EU compliance"),
        ]
    }

    # Flatten into searchable format with pre-computed normalized search text
    all_models = []
    for category, model_list in models.items():
        category_emoji = category.split()[0]
        for model_id, display_name, cost, badge in model_list:
            search_text = f"{display_name} {model_id} {cost} {badge} {category}"
            all_models.append({
                "id": model_id,
                "name": display_name,
                "cost": cost,
                "badge": badge,
                "category": category,
                "display": f"{category_emoji} {display_name} • {cost} • {badge}",
                "search_text": search_text.lower(),
                "search_text_normalized": normalize_search_text(search_text)
            })

    # Search box
    search_query = st.text_input(
        "🔍 Search Models",
        placeholder="Type to filter (e.g., 'claude 3', 'gpt 4o', '200k', 'budget')...",
        key="openrouter_model_search"
    )

    # Filter models using normalized search
    if search_query:
        normalized_query = normalize_search_text(search_query)
        filtered_models = [m for m in all_models if normalized_query in m["search_text_normalized"]]
        st.caption(f"Found {len(filtered_models)} model(s)")
    else:
        filtered_models = all_models

    # Handle no results
    if not filtered_models:
        st.warning("No models match your search. Try different keywords.")
        return os.getenv('OPENROUTER_MODEL', 'openai/gpt-4o-mini')

    # Create selectbox options
    options = [m["id"] for m in filtered_models]
    display_map = {m["id"]: m["display"] for m in filtered_models}

    # Get default from env or session state
    default_model = os.getenv('OPENROUTER_MODEL', 'openai/gpt-4o-mini')
    if 'openrouter_model' not in st.session_state:
        st.session_state.openrouter_model = default_model

    # Ensure default is in filtered options (fallback to first if not)
    if st.session_state.openrouter_model not in options:
        st.session_state.openrouter_model = options[0]

    # Model selector
    selected_model = st.selectbox(
        "Select Model",
        options=options,
        index=options.index(st.session_state.openrouter_model),
        format_func=lambda x: display_map.get(x, x),
        help="Battle-tested models (Oct 2025). Quality scores from JSON mode + legal extraction tests.",
        key="openrouter_model_selector"
    )

    # Update session state if changed
    if selected_model != st.session_state.openrouter_model:
        st.session_state.openrouter_model = selected_model
        # Clear previous results when model changes
        if 'legal_events_df' in st.session_state:
            del st.session_state.legal_events_df

    # Model selection guidance
    with st.expander("💡 Quick Selection Guide", expanded=False):
        st.markdown("""
        **Not sure which model to choose?**

        - **New user?** → Start with **GPT-4o Mini** (proven on 15-page contracts, best balance)
        - **Processing 1000+ documents?** → Use **DeepSeek R1 Distill** (50x cheaper, same 10/10 quality)
        - **Extreme budget?** → Try **Qwen QwQ 32B** ($0.115/M, but ⚠️ may miss some events on complex docs)
        - **50+ page contracts?** → Use **Claude 3.5 Sonnet** or **Claude 3 Haiku** (200K context window)
        - **Need results in 5 seconds?** → Use **Claude 3 Haiku** (4.4s extraction time)
        - **EU data compliance?** → Use **Mistral Small** (EU-hosted model)

        **Key Metrics**:
        - **Quality scores** (10/10, 9/10, 7/10): From Oct 2025 JSON mode + legal extraction tests
        - **⚠️ Warning symbol**: Model may extract fewer events on complex documents (test before production)
        - **Context window** (128K, 200K): Capacity for long documents (~100-200 pages)
        - **Cost** (/M = per million tokens): $0.03 to $3.00 range
        - **Speed**: Claude 3 Haiku extracts in 4.4s vs others at 14-36s

        ℹ️ Most models scored 10/10 or 9/10. GPT-OSS/Gemini/Cohere/Perplexity failed and were excluded.
        """)

    return selected_model


def create_doc_extractor_selection():
    """Create document extractor selection UI"""
    st.markdown('<div class="section-header">📄 Document Processing</div>', unsafe_allow_html=True)

    # Initialize session state for document extractor
    default_doc_extractor = os.getenv('DOC_EXTRACTOR', 'docling').lower()
    if 'selected_doc_extractor' not in st.session_state:
        st.session_state.selected_doc_extractor = default_doc_extractor

    # Check if Gemini API key is configured
    gemini_configured = bool(os.getenv('GEMINI_API_KEY'))

    # Document extractor options with cost/speed info
    doc_options = {
        'docling': {
            'name': '🔧 Docling (Local Processing)',
            'subtitle': 'FREE • Fast • OCR-Ready',
            'description': '✅ Production-ready, Tesseract OCR, 2-22s per doc'
        },
        'gemini': {
            'name': '🌟 Gemini 2.5 (Cloud Vision)',
            'subtitle': 'Premium • Multimodal • 4.5x cost',
            'description': '🧪 Experimental, Native PDF vision, ~$0.0014 per doc' if gemini_configured else '⚠️ Requires GEMINI_API_KEY'
        }
    }

    # Radio button for document extractor selection
    selected_doc_extractor = st.radio(
        "Choose document processor:",
        options=['docling', 'gemini'],
        format_func=lambda x: doc_options[x]['name'],
        index=0 if st.session_state.selected_doc_extractor == 'docling' else 1,
        help="Docling: Local OCR processing (FREE). Gemini: Cloud multimodal vision (premium quality, higher cost)"
    )

    # Show subtitle and description for selected option
    option_info = doc_options[selected_doc_extractor]
    st.caption(f"{option_info['subtitle']}")
    st.caption(f"ℹ️ {option_info['description']}")

    # Warn if Gemini selected but not configured
    if selected_doc_extractor == 'gemini' and not gemini_configured:
        st.warning("⚠️ **Gemini API Key Required**: Set `GEMINI_API_KEY` in your `.env` file to use Gemini document extraction.")

    # Update session state and clear cache if changed
    if selected_doc_extractor != st.session_state.selected_doc_extractor:
        st.session_state.selected_doc_extractor = selected_doc_extractor
        if 'legal_events_df' in st.session_state:
            del st.session_state.legal_events_df

    return selected_doc_extractor


def create_provider_selection():
    """Create provider selection: Two primary providers (OpenRouter + LangExtract) with advanced options"""
    # Styled section header
    st.markdown('<div class="section-header">🤖 Event Extraction Provider</div>', unsafe_allow_html=True)

    # Initialize session state
    default_provider = os.getenv('EVENT_EXTRACTOR', 'openrouter').lower()
    if 'selected_provider' not in st.session_state:
        st.session_state.selected_provider = default_provider

    # === PRIMARY: Two main providers side-by-side ===
    col1, col2 = st.columns(2)

    # Column 1: OpenRouter
    with col1:
        openrouter_configured = check_provider_status('openrouter')
        is_openrouter_selected = st.session_state.selected_provider == 'openrouter'

        if st.button(
            "🌐 OpenRouter",
            key="provider_openrouter",
            use_container_width=True,
            type="primary" if is_openrouter_selected else "secondary"
        ):
            if st.session_state.selected_provider != 'openrouter':
                st.session_state.selected_provider = 'openrouter'
                if 'legal_events_df' in st.session_state:
                    del st.session_state.legal_events_df
                st.rerun()

        st.caption("One API • 18+ Models")
        if openrouter_configured:
            st.markdown('<span class="status-pill ready">✓ Ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-pill setup">⚠ Setup Required</span>', unsafe_allow_html=True)

    # Column 2: Gemini
    with col2:
        langextract_configured = check_provider_status('langextract')
        is_langextract_selected = st.session_state.selected_provider == 'langextract'

        if st.button(
            "🔮 Gemini",
            key="provider_langextract",
            use_container_width=True,
            type="primary" if is_langextract_selected else "secondary"
        ):
            if st.session_state.selected_provider != 'langextract':
                st.session_state.selected_provider = 'langextract'
                if 'legal_events_df' in st.session_state:
                    del st.session_state.legal_events_df
                st.rerun()

        st.caption("Google • 2M Context")
        if langextract_configured:
            st.markdown('<span class="status-pill ready">✓ Ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-pill setup">⚠ Setup Required</span>', unsafe_allow_html=True)

    # Show model selector if OpenRouter is selected
    if is_openrouter_selected:
        if openrouter_configured:
            st.markdown("")  # Spacing
            selected_model = create_unified_model_selector('openrouter')
            return 'openrouter', selected_model
        else:
            st.info("💡 **Setup**: Add `OPENROUTER_API_KEY` to your `.env` file, then restart.\n\nGet free API key: https://openrouter.ai/keys")
            return 'openrouter', None

    # Show model selector if Gemini is selected
    if is_langextract_selected:
        if langextract_configured:
            st.markdown("")  # Spacing
            selected_model = create_unified_model_selector('google')  # 'google' in MODEL_CATALOG, backend uses 'langextract'
            return 'langextract', selected_model
        else:
            st.info("💡 **Setup**: Add `GEMINI_API_KEY` to your `.env` file, then restart.\n\nGet free API key: https://aistudio.google.com/app/apikey")
            return 'langextract', None

    st.divider()

    # === SECONDARY: Direct Provider APIs (Advanced) ===
    with st.expander("🔧 Direct Provider APIs (Advanced)", expanded=False):
        st.caption("Enterprise-grade providers with direct API access")

        # Direct provider options (without LangExtract - now in primary section)
        direct_providers = {
            'openai': ('🧠 OpenAI', 'GPT-4o / 4o-mini', '$0.15-$3/M', 'OPENAI_API_KEY'),
            'anthropic': ('🎯 Anthropic', 'Claude 3.5', 'Premium', 'ANTHROPIC_API_KEY'),
            'deepseek': ('🔍 DeepSeek', 'DeepSeek R1', 'Budget', 'DEEPSEEK_API_KEY'),
            'opencode_zen': ('⚖️ OpenCode Zen', 'Legal AI', 'Premium', 'OPENCODEZEN_API_KEY'),
        }

        cols = st.columns(2)
        for idx, (key, (icon_name, subtitle, tier, env_key)) in enumerate(direct_providers.items()):
            col = cols[idx % 2]

            with col:
                is_configured = check_provider_status(key)
                is_selected = st.session_state.selected_provider == key

                if st.button(
                    f"{icon_name}",
                    key=f"provider_{key}",
                    use_container_width=True,
                    type="primary" if is_selected else "secondary"
                ):
                    if st.session_state.selected_provider != key:
                        st.session_state.selected_provider = key
                        if 'legal_events_df' in st.session_state:
                            del st.session_state.legal_events_df
                        st.rerun()

                st.caption(f"{subtitle} • {tier}")
                badge = "✅" if is_configured else f"⚠️ {env_key}"
                st.caption(badge)

    # Show model selectors for direct provider APIs if they're selected
    selected_provider = st.session_state.selected_provider
    if selected_provider == 'anthropic' and check_provider_status('anthropic'):
        st.markdown("")  # Spacing
        return 'anthropic', create_unified_model_selector('anthropic')
    elif selected_provider == 'openai' and check_provider_status('openai'):
        st.markdown("")  # Spacing
        return 'openai', create_unified_model_selector('openai')
    elif selected_provider == 'deepseek' and check_provider_status('deepseek'):
        st.markdown("")  # Spacing
        return 'deepseek', create_unified_model_selector('deepseek')
    elif selected_provider == 'opencode_zen':
        # OpenCodeZen doesn't have models in catalog yet - use env defaults
        return selected_provider, None

    # No provider selected or not configured
    return st.session_state.selected_provider, None


def create_polished_metrics(legal_events_df):
    """Create polished metrics cards with gradients and icons"""
    from src.core.constants import FIVE_COLUMN_HEADERS

    col1, col2, col3, col4 = st.columns(4)

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
                <div class="metric-label" style="font-size: 0.7rem;">characters</div>
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
    st.markdown('<p class="main-caption">Document processing with flexible extractors (Docling/Gemini + AI providers)</p>', unsafe_allow_html=True)

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
            sample_path = Path("sample_pdf/famas_dispute/Answer to Request for Arbitration.pdf")

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
            sample_path = Path("sample_pdf/famas_dispute/Answer to Request for Arbitration.pdf")

            if sample_path.exists():
                # Read sample file and create a file-like object
                import io
                sample_bytes = sample_path.read_bytes()
                sample_file = io.BytesIO(sample_bytes)
                sample_file.name = sample_path.name
                files_to_process = [sample_file]
                st.info(f"📄 Ready to process: **{sample_path.name}** (930 KB)")

        if files_to_process:
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

                # Show processing status with context
                with status_container:
                    if total_size > 15:
                        time_estimate = "⏱️ This may take 1-2 minutes..."
                    elif total_size > 5:
                        time_estimate = "⏱️ Estimated time: 30-60 seconds..."
                    else:
                        time_estimate = "⚡ Processing..."

                    st.info(
                        f"🔄 **Processing {file_count} file{'s' if file_count > 1 else ''}** via {provider_display}\n\n"
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
                    # Store results in session state
                    st.session_state.legal_events_df = legal_events_df

                    # Clear sample document flag after processing
                    if 'use_sample' in st.session_state:
                        del st.session_state['use_sample']

                    # Clear processing status
                    status_container.empty()
                else:
                    # Clear processing status (error shown by shared utility)
                    status_container.empty()
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

        # Display table
        st.markdown("### 📋 Legal Events Table")
        from src.core.constants import FIVE_COLUMN_HEADERS
        st.dataframe(
            legal_events_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                FIVE_COLUMN_HEADERS[0]: st.column_config.NumberColumn(FIVE_COLUMN_HEADERS[0], width="small"),
                FIVE_COLUMN_HEADERS[1]: st.column_config.TextColumn(FIVE_COLUMN_HEADERS[1], width="medium"),
                FIVE_COLUMN_HEADERS[2]: st.column_config.TextColumn(FIVE_COLUMN_HEADERS[2], width="large"),
                FIVE_COLUMN_HEADERS[3]: st.column_config.TextColumn(FIVE_COLUMN_HEADERS[3], width="medium"),
                FIVE_COLUMN_HEADERS[4]: st.column_config.TextColumn(FIVE_COLUMN_HEADERS[4], width="medium")
            }
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
