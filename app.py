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

# Custom CSS - Enhanced styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

.main-header {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 1.75rem;
    font-weight: 500;
    color: #222;
    text-align: center;
    margin-bottom: 1rem;
}
.main-caption {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 0.95rem;
    color: #555;
    text-align: center;
    margin-bottom: 2rem;
}
div[data-testid="stSidebar"] {
    font-size: 0.75rem;
}
.stMarkdown {
    font-size: 0.875rem;
}

/* Provider card styling */
.provider-card {
    padding: 1rem;
    border-radius: 8px;
    border: 2px solid #e0e0e0;
    margin-bottom: 0.5rem;
    background: white;
    transition: all 0.2s ease;
}
.provider-card:hover {
    border-color: #4CAF50;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
.provider-card.selected {
    border-color: #4CAF50;
    background: #f1f8f4;
}
.provider-status-badge {
    display: inline-block;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
}
.status-configured {
    background: #d4edda;
    color: #155724;
}
.status-missing {
    background: #f8d7da;
    color: #721c24;
}

/* Metrics card styling */
.metric-card {
    padding: 1.25rem;
    border-radius: 8px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    text-align: center;
}
.metric-card.success {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
}
.metric-card.info {
    background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
}
.metric-card.warning {
    background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
}
.metric-value {
    font-size: 2rem;
    font-weight: 600;
    margin: 0.5rem 0;
}
.metric-label {
    font-size: 0.875rem;
    opacity: 0.9;
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


def create_openrouter_model_selector():
    """Create OpenRouter model dropdown with categorized models from test results"""

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

    # Flatten for dropdown (no headers, just clean list)
    options = []
    format_map = {}

    for category, model_list in models.items():
        # Add models with category prefix
        for model_id, display_name, cost, badge in model_list:
            options.append(model_id)
            # Show category in label for context
            category_emoji = category.split()[0]  # Get emoji from category name
            format_map[model_id] = f"{category_emoji} {display_name} • {cost} • {badge}"

    # Get default from env or session state
    default_model = os.getenv('OPENROUTER_MODEL', 'openai/gpt-4o-mini')
    if 'openrouter_model' not in st.session_state:
        st.session_state.openrouter_model = default_model

    # Ensure default is in options list
    if st.session_state.openrouter_model not in options:
        st.session_state.openrouter_model = 'openai/gpt-4o-mini'

    # Model selector
    selected_model = st.selectbox(
        "Model Selection",
        options=options,
        index=options.index(st.session_state.openrouter_model),
        format_func=lambda x: format_map.get(x, x),
        help="Battle-tested models (Oct 2025). Quality scores from JSON mode + legal extraction tests. Context windows shown for long documents.",
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
    st.markdown("**Document Processing**")

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
    """Create restructured provider selection: OpenRouter gateway vs direct APIs"""
    st.markdown("**Event Extraction Provider**")

    # Initialize session state
    default_provider = os.getenv('EVENT_EXTRACTOR', 'openrouter').lower()
    if 'selected_provider' not in st.session_state:
        st.session_state.selected_provider = default_provider

    # === PRIMARY: OpenRouter Gateway (Recommended) ===
    st.markdown("### 🌐 Unified Gateway (Recommended)")

    openrouter_configured = check_provider_status('openrouter')
    is_openrouter_selected = st.session_state.selected_provider == 'openrouter'

    # OpenRouter selection button
    if st.button(
        "🌐 OpenRouter • One API Key • 18+ Models",
        key="provider_openrouter",
        use_container_width=True,
        type="primary" if is_openrouter_selected else "secondary"
    ):
        if st.session_state.selected_provider != 'openrouter':
            st.session_state.selected_provider = 'openrouter'
            if 'legal_events_df' in st.session_state:
                del st.session_state.legal_events_df
            st.rerun()

    # Status badge
    status_badge = "✅ Ready" if openrouter_configured else "⚠️ Setup Required"
    status_class = "configured" if openrouter_configured else "missing"
    st.markdown(f'<span class="provider-status-badge status-{status_class}">{status_badge}</span>', unsafe_allow_html=True)

    # Show model selector if OpenRouter is selected
    if is_openrouter_selected:
        if openrouter_configured:
            st.markdown("")  # Spacing
            selected_model = create_openrouter_model_selector()
            return 'openrouter', selected_model
        else:
            st.info("💡 **Setup**: Add `OPENROUTER_API_KEY` to your `.env` file, then restart.\n\nGet free API key: https://openrouter.ai/keys")
            return 'openrouter', None

    st.divider()

    # === SECONDARY: Direct Provider APIs (Advanced) ===
    with st.expander("🔧 Direct Provider APIs (Advanced)", expanded=False):
        st.caption("Use when you need provider-specific features or enterprise accounts")

        # Direct provider options
        direct_providers = {
            'langextract': ('🤖 LangExtract', 'Google Gemini 2.0', 'Free', 'GEMINI_API_KEY'),
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

    # Return selected provider (no model for direct APIs yet)
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
        type=['pdf', 'docx', 'txt', 'msg'],
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX, TXT, MSG"
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
            st.warning(
                f"⚠️ **Very large files detected:** {file_list}\n\n"
                f"Processing may take **60-120 seconds per file**. Please be patient."
            )
        elif large_files:
            file_list = ", ".join([f"**{name}** ({size:.1f}MB)" for name, size in large_files])
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

        # Quick sample document button
        st.markdown("**Quick Test**")
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
                legal_events_df = process_documents_with_spinner(
                    files_to_process,
                    show_subheader=False,
                    provider=selected_provider,
                    runtime_model=selected_model if selected_provider == 'openrouter' else None,
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