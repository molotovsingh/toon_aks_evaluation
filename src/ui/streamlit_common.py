"""
Shared Streamlit Utilities - Common functions for all legal events apps
Ensures consistent pipeline caching, processing, and UI across endpoints
"""

import streamlit as st
import pandas as pd
import logging
import re
from pathlib import Path
from typing import Optional, List

from ..core.legal_pipeline_refactored import LegalEventsPipeline
from ..core.constants import FIVE_COLUMN_HEADERS

logger = logging.getLogger(__name__)



def get_pipeline(provider: Optional[str] = None, runtime_model: Optional[str] = None, doc_extractor: Optional[str] = None) -> Optional[LegalEventsPipeline]:
    """
    Get or create pipeline instance with session state caching
    Ensures environment validation runs once and same instance is reused

    Args:
        provider: Event extractor provider ('langextract', 'openrouter', 'opencode_zen', 'openai', 'anthropic')
                  If None, uses environment default (EVENT_EXTRACTOR env var)
        runtime_model: Runtime model override (for OpenRouter multi-model selection)
                       If None, uses environment default (OPENROUTER_MODEL env var)
        doc_extractor: Document extractor provider ('docling', 'gemini')
                       If None, uses environment default (DOC_EXTRACTOR env var)

    Returns:
        LegalEventsPipeline instance configured for the specified provider, or None if initialization fails
    """
    from ..core.extractor_factory import ExtractorConfigurationError

    # Track current provider + model + doc extractor combination for cache invalidation
    current_provider = provider if provider else 'default'
    current_doc = doc_extractor if doc_extractor else 'default'
    current_key = f"{current_doc}:{current_provider}:{runtime_model or 'default'}"

    # Invalidate cache if provider or model changed
    if 'pipeline_key' in st.session_state and st.session_state['pipeline_key'] != current_key:
        old_key = st.session_state['pipeline_key']
        logger.info(f"🔄 Pipeline config changed from {old_key} to {current_key} - clearing cache")
        if 'pipeline' in st.session_state:
            del st.session_state['pipeline']
        # Clear any previous provider errors
        if 'provider_error' in st.session_state:
            del st.session_state['provider_error']

    # Create new pipeline if not cached
    if 'pipeline' not in st.session_state:
        try:
            st.session_state['pipeline'] = LegalEventsPipeline(
                event_extractor=provider,
                runtime_model=runtime_model,
                doc_extractor=doc_extractor
            )
            st.session_state['pipeline_key'] = current_key

            # Clear any previous errors on successful initialization
            if 'provider_error' in st.session_state:
                del st.session_state['provider_error']

            provider_display = provider if provider else "environment default"
            doc_display = doc_extractor if doc_extractor else "environment default"
            logger.info(f"✅ Pipeline initialized with doc extractor: {doc_display}, event provider: {provider_display}")

        except ValueError as e:
            # Handle pipeline-level validation errors (provider-specific credential checks)
            provider_name = provider if provider else "langextract"
            logger.error(f"❌ Pipeline validation error for {provider_name}: {e}")

            # Store error in session state for display
            st.session_state['provider_error'] = {
                'provider': provider_name,
                'message': str(e),
                'type': 'validation'
            }

            # Display user-friendly error with guidance
            error_msg = f"**Provider Validation Error: {provider_name}**\n\n{str(e)}"

            # Add specific guidance based on provider
            if 'openrouter' in provider_name.lower():
                error_msg += "\n\n**Required**: Set `OPENROUTER_API_KEY` in your `.env` file"
            elif 'opencode' in provider_name.lower() or 'zen' in provider_name.lower():
                error_msg += "\n\n**Required**: Set `OPENCODEZEN_API_KEY` in your `.env` file"
            elif 'openai' in provider_name.lower():
                error_msg += "\n\n**Required**: Set `OPENAI_API_KEY` in your `.env` file"
            elif 'anthropic' in provider_name.lower():
                error_msg += "\n\n**Required**: Set `ANTHROPIC_API_KEY` in your `.env` file"
            elif 'langextract' in provider_name.lower() or 'default' in provider_name.lower():
                error_msg += "\n\n**Required**: Set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) in your `.env` file"

            error_msg += "\n\nℹ️ **Tip**: Configure the required API key in your `.env` file, then restart the app."

            st.error(error_msg)
            return None

        except ExtractorConfigurationError as e:
            # Handle adapter-level configuration errors (secondary validation)
            provider_name = provider if provider else "default"
            logger.error(f"❌ Provider configuration error for {provider_name}: {e}")

            # Store error in session state for display
            st.session_state['provider_error'] = {
                'provider': provider_name,
                'message': str(e),
                'type': 'configuration'
            }

            # Display user-friendly error with guidance
            error_msg = f"**Provider Configuration Error: {provider_name}**\n\n{str(e)}"

            # Add specific guidance based on provider
            if 'openrouter' in provider_name.lower():
                error_msg += "\n\n**Required**: Set `OPENROUTER_API_KEY` in your `.env` file"
            elif 'opencode' in provider_name.lower() or 'zen' in provider_name.lower():
                error_msg += "\n\n**Required**: Set `OPENCODEZEN_API_KEY` in your `.env` file"
            elif 'openai' in provider_name.lower():
                error_msg += "\n\n**Required**: Set `OPENAI_API_KEY` in your `.env` file"
            elif 'anthropic' in provider_name.lower():
                error_msg += "\n\n**Required**: Set `ANTHROPIC_API_KEY` in your `.env` file"
            elif 'langextract' in provider_name.lower():
                error_msg += "\n\n**Required**: Set `GEMINI_API_KEY` in your `.env` file"

            error_msg += "\n\nℹ️ **Tip**: Switch to a different provider or configure the required API key, then restart the app."

            st.error(error_msg)
            return None

        except Exception as e:
            # Handle unexpected errors
            provider_name = provider if provider else "default"
            logger.error(f"❌ Unexpected error initializing pipeline for {provider_name}: {e}")

            st.session_state['provider_error'] = {
                'provider': provider_name,
                'message': str(e),
                'type': 'unexpected'
            }

            st.error(f"🚨 **Unexpected Error**: Failed to initialize pipeline with provider `{provider_name}`\n\n{str(e)}\n\nPlease check logs for details.")
            return None

    return st.session_state.get('pipeline')


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing or replacing problematic characters

    Args:
        filename: Original filename

    Returns:
        Sanitized filename safe for file systems
    """
    # Remove extension
    name = Path(filename).stem

    # Replace spaces and special chars with underscores
    name = re.sub(r'[^\w\-.]', '_', name)

    # Remove multiple consecutive underscores
    name = re.sub(r'_+', '_', name)

    # Remove leading/trailing underscores
    name = name.strip('_')

    return name or "document"


def save_results_to_project(
    df: pd.DataFrame,
    provider: str,
    uploaded_files: List,
    pipeline: Optional[LegalEventsPipeline] = None
) -> None:
    """
    Auto-save results with parser-extractor pair identifier + metadata sidecar

    Saves to: output/docling-{provider}/{doc_name}_{timestamp}.{xlsx,csv,json,_metadata.json}

    Args:
        df: DataFrame with legal events to save
        provider: Event extractor provider name
        uploaded_files: List of uploaded file objects (for naming)
        pipeline: LegalEventsPipeline instance for export functions
    """
    if df is None or df.empty:
        return

    if not uploaded_files:
        logger.warning("⚠️ Auto-save skipped: No uploaded files info")
        return

    if pipeline is None:
        logger.warning("⚠️ Auto-save skipped: Pipeline not available")
        return

    try:
        # Create parser-extractor directory structure
        parser_name = "docling"  # Current parser (future: make configurable)
        output_dir = Path("output") / f"{parser_name}-{provider}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize document name from first uploaded file
        doc_name = sanitize_filename(uploaded_files[0].name)

        # Create timestamp for uniqueness
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"{doc_name}_{timestamp}"

        # Save all three formats for different use cases
        saved_files = []
        for fmt in ['xlsx', 'csv', 'json']:
            try:
                # Use pipeline's export function (already tested)
                data = pipeline.export_legal_events_table(df, fmt)
                file_path = output_dir / f"{base_name}.{fmt}"
                file_path.write_bytes(data)
                saved_files.append(f"{fmt}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to save {fmt}: {e}")

        # Save metadata sidecar JSON (for manual inspection and future DB import)
        try:
            metadata_dict = df.attrs.get('metadata')
            if metadata_dict:
                import json
                metadata_path = output_dir / f"{base_name}_metadata.json"
                metadata_path.write_text(json.dumps(metadata_dict, indent=2, default=str), encoding='utf-8')
                saved_files.append('metadata.json')
                logger.info(f"💾 Saved metadata sidecar: {metadata_path.name}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to save metadata sidecar: {e}")

        if saved_files:
            logger.info(
                f"✅ Auto-saved to {output_dir.name}/{base_name}.[{','.join(saved_files)}]"
            )

    except Exception as e:
        # Non-blocking: Log error but don't fail the pipeline
        logger.warning(f"⚠️ Auto-save failed: {e}")


def process_documents_with_spinner(uploaded_files, show_subheader: bool = True, provider: Optional[str] = None, runtime_model: Optional[str] = None, doc_extractor: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Shared processing helper with spinner and status handling
    Reusable across all Streamlit entry points to avoid duplication

    Args:
        uploaded_files: List of uploaded file objects
        show_subheader: Whether to show processing subheader
        provider: Event extractor provider override (optional)
        runtime_model: Runtime model override for OpenRouter multi-model selection (optional)
        doc_extractor: Document extractor provider override (optional)

    Returns:
        DataFrame with legal events or None if failed
    """
    if not uploaded_files:
        return None

    if show_subheader:
        st.subheader("🔄 Legal Events Processing")

    # Determine provider display names for spinner
    doc_extractor_name = doc_extractor.title() if doc_extractor else "Docling"
    provider_name = provider if provider else "default provider"
    model_suffix = f" ({runtime_model})" if runtime_model else ""
    spinner_text = f"Processing through pipeline: {doc_extractor_name} → {provider_name.title()}{model_suffix} → Five-Column Table..."

    with st.spinner(spinner_text):
        try:
            # Get cached pipeline instance with provider and model override
            pipeline = get_pipeline(provider=provider, runtime_model=runtime_model, doc_extractor=doc_extractor)

            # Check if pipeline initialization failed
            if pipeline is None:
                st.warning("⚠️ Cannot process documents - provider initialization failed. Please check the error message above.")
                return None

            # Process documents through standardized sequence
            legal_events_df, warning_message = pipeline.process_documents_for_legal_events(uploaded_files)

            # Display warnings if any
            if warning_message:
                st.warning(f"⚠️ {warning_message}")

            # Validate result
            if legal_events_df is None or legal_events_df.empty:
                st.error("🚨 No legal events extracted from any documents")
                return None

            # Validate format
            if not pipeline.validate_five_column_format(legal_events_df):
                st.error("🚨 DataFrame format validation failed")
                return None

            st.success(f"✅ Successfully extracted {len(legal_events_df)} legal events")

            # Store in session state for tab access
            st.session_state['legal_events_df'] = legal_events_df

            # Auto-save results with parser-extractor pair identifier
            provider_key = provider if provider else "langextract"
            save_results_to_project(legal_events_df, provider_key, uploaded_files, pipeline)

            return legal_events_df

        except Exception as e:
            st.error(f"🚨 CRITICAL PIPELINE FAILURE: {str(e)}")
            return None


def display_legal_events_table(legal_events_df: pd.DataFrame) -> None:
    """
    Standard legal events table display using shared column constants
    Ensures consistent formatting across all apps
    """
    if legal_events_df is None or legal_events_df.empty:
        st.info("No legal events to display")
        return

    st.header("Legal Events Table")
    st.caption("Legal events extracted from uploaded documents")

    # Display table with standardized column configuration
    st.dataframe(
        legal_events_df,
        width='stretch',
        hide_index=True,
        column_config={
            FIVE_COLUMN_HEADERS[0]: st.column_config.NumberColumn(FIVE_COLUMN_HEADERS[0], width="small"),
            FIVE_COLUMN_HEADERS[1]: st.column_config.TextColumn(FIVE_COLUMN_HEADERS[1], width="medium"),  # Date
            FIVE_COLUMN_HEADERS[2]: st.column_config.TextColumn(FIVE_COLUMN_HEADERS[2], width="large"),   # Event Particulars
            FIVE_COLUMN_HEADERS[3]: st.column_config.TextColumn(FIVE_COLUMN_HEADERS[3], width="medium"),  # Citation
            FIVE_COLUMN_HEADERS[4]: st.column_config.TextColumn(FIVE_COLUMN_HEADERS[4], width="medium")   # Document Reference
        }
    )

    # Display summary statistics using shared constants
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Events", len(legal_events_df))
    with col2:
        unique_docs = legal_events_df[FIVE_COLUMN_HEADERS[4]].nunique()  # Document Reference
        st.metric("Documents Processed", unique_docs)
    with col3:
        # Count events with real citations (not defaults)
        citations_count = len(legal_events_df[
            (~legal_events_df[FIVE_COLUMN_HEADERS[3]].str.contains("No citation available", na=False)) &  # Citation
            (~legal_events_df[FIVE_COLUMN_HEADERS[3]].str.contains("processing failed", na=False))
        ])
        st.metric("Events with Citations", citations_count)
    with col4:
        avg_chars = legal_events_df[FIVE_COLUMN_HEADERS[2]].str.len().mean()  # Event Particulars
        st.metric("Avg Event Detail Length", f"{avg_chars:.0f} chars")

    # Display performance timing metrics if available
    if "Docling_Seconds" in legal_events_df.columns and "Extractor_Seconds" in legal_events_df.columns:
        st.subheader("⏱️  Performance Metrics")
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


def create_download_section(legal_events_df: pd.DataFrame, provider: Optional[str] = None) -> None:
    """
    Standard download section for legal events tables
    Uses shared pipeline instance for consistent export formats

    Args:
        legal_events_df: DataFrame to export
        provider: Event extractor provider (for pipeline caching)
    """
    if legal_events_df is None or legal_events_df.empty:
        return

    st.header("💾 Download Legal Events")

    pipeline = get_pipeline(provider=provider)

    # Check if pipeline is available
    if pipeline is None:
        st.info("⚠️ Pipeline not available. Downloads disabled until provider is properly configured.")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        # Excel download
        try:
            excel_data = pipeline.export_legal_events_table(legal_events_df, "xlsx")
            st.download_button(
                label="📊 Download Excel (.xlsx)",
                data=excel_data,
                file_name=f"legal_events_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.error(f"Excel export failed: {e}")

    with col2:
        # CSV download
        try:
            csv_data = pipeline.export_legal_events_table(legal_events_df, "csv")
            st.download_button(
                label="📄 Download CSV (.csv)",
                data=csv_data,
                file_name=f"legal_events_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"CSV export failed: {e}")

    with col3:
        # JSON download
        try:
            json_data = pipeline.export_legal_events_table(legal_events_df, "json")
            st.download_button(
                label="🔧 Download JSON (.json)",
                data=json_data,
                file_name=f"legal_events_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        except Exception as e:
            st.error(f"JSON export failed: {e}")


def show_sample_table_format() -> None:
    """Display sample table format using shared constants"""
    st.subheader("📋 Sample Five-Column Format")
    st.caption("This is the guaranteed output format:")

    from ..core.table_formatter import TableFormatter
    sample_df = TableFormatter.create_fallback_dataframe("Sample - no files uploaded yet")
    st.dataframe(sample_df, width='stretch', hide_index=True)