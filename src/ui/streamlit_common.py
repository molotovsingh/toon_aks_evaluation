"""
Shared Streamlit Utilities - Common functions for all legal events apps
Ensures consistent pipeline caching, processing, and UI across endpoints
Includes document extraction cache management utilities
"""

import streamlit as st
import pandas as pd
import logging
import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple

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


def extract_all_documents_for_estimation(
    uploaded_files: List,
    provider: Optional[str] = None,
    runtime_model: Optional[str] = None,
    doc_extractor: Optional[str] = None
) -> Optional[List[str]]:
    """
    Extract text from all uploaded documents once for cost estimation.

    This function runs Docling extraction on all files and caches the results
    in session state to avoid re-extraction during processing.

    Args:
        uploaded_files: List of uploaded file objects
        provider: Event extractor provider (for pipeline initialization)
        runtime_model: Runtime model override (optional)
        doc_extractor: Document extractor provider ('docling', 'gemini', 'qwen_vl')

    Returns:
        List of extracted plain text strings, or None if extraction failed
    """
    if not uploaded_files:
        return None

    try:
        # Get pipeline for document extraction
        pipeline = get_pipeline(provider=provider, runtime_model=runtime_model, doc_extractor=doc_extractor)

        if pipeline is None:
            return None  # Pipeline initialization failed

        # Import tempfile and file handler for saving BytesIO files
        import tempfile
        from ..utils.file_handler import FileHandler
        file_handler = FileHandler()

        # Extract text from all documents
        extracted_texts = []

        # Create temporary directory for BytesIO files (same as processing path)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with st.spinner(f"📊 Extracting text from {len(uploaded_files)} file{'s' if len(uploaded_files) > 1 else ''}..."):
                for idx, file in enumerate(uploaded_files, 1):
                    try:
                        # Reset file position if it's a file-like object (BytesIO, uploaded file, etc.)
                        if hasattr(file, 'seek'):
                            file.seek(0)

                        # Save file to disk first (required for Docling)
                        # This matches the processing path behavior
                        file_path = file_handler.save_uploaded_file(file, temp_path)

                        # Extract from saved file path (not BytesIO)
                        extracted_doc = pipeline.document_extractor.extract(file_path)
                        plain_text = extracted_doc.plain_text if extracted_doc else ""

                        if not plain_text:
                            logger.warning(f"⚠️ No text extracted from file {idx}/{len(uploaded_files)}: {file.name}")
                            extracted_texts.append("")  # Add empty string to maintain index alignment
                        else:
                            extracted_texts.append(plain_text)
                            logger.info(f"✅ Extracted {len(plain_text)} chars from file {idx}/{len(uploaded_files)}: {file.name}")

                    except Exception as e:
                        logger.warning(f"⚠️ Extraction failed for file {idx}/{len(uploaded_files)}: {file.name} - {e}")
                        extracted_texts.append("")  # Add empty string to maintain index alignment

        # Cache extracted texts in session state (keyed by file names + doc_extractor for cache invalidation)
        file_key = ":".join(f.name for f in uploaded_files)
        cache_key = f"{file_key}:{doc_extractor or 'docling'}"
        if 'extracted_texts_cache' not in st.session_state:
            st.session_state['extracted_texts_cache'] = {}

        st.session_state['extracted_texts_cache'][cache_key] = extracted_texts

        # Filter out empty texts for validation
        non_empty_texts = [t for t in extracted_texts if t.strip()]
        if not non_empty_texts:
            st.warning("⚠️ No text could be extracted from any uploaded files")
            return None

        return extracted_texts

    except Exception as e:
        logger.error(f"❌ Document extraction failed: {e}")
        st.error(f"🚨 Text extraction failed: {str(e)}")
        return None


def display_cost_estimates(
    extracted_texts: Optional[List[str]],
    provider: Optional[str] = None,
    runtime_model: Optional[str] = None,
    show_all_models: bool = False,
    uploaded_files: Optional[List] = None,
    doc_extractor: Optional[str] = None
) -> None:
    """
    Display cost estimates for document extraction before processing.

    Shows TWO-LAYER cost breakdown: document extraction (Layer 1) + event extraction (Layer 2).

    Args:
        extracted_texts: List of pre-extracted plain text strings from documents (or None for pre-extraction estimate)
        provider: Event extractor provider (for filtering models)
        runtime_model: Specific model to estimate for (optional)
        show_all_models: Show estimates for all recommended models (default: False)
        uploaded_files: List of uploaded file objects (for Layer 1 cost estimation)
        doc_extractor: Document extractor ID ('docling', 'qwen_vl', 'gemini') for Layer 1
    """
    from .cost_estimator import (
        estimate_cost,
        estimate_all_models,
        estimate_tokens,
        estimate_cost_two_layer
    )

    try:
        # Create expandable section for cost estimates
        with st.expander("💰 **Cost Estimates** (click to expand)", expanded=False):
            st.caption("⚠️ **Estimates only — actual billing may vary by ±20-30%**")

            # === TWO-LAYER COST BREAKDOWN (if doc_extractor provided) ===
            if uploaded_files and doc_extractor and runtime_model:
                # Calculate two-layer estimate (works with or without extracted_texts)
                two_layer_result = estimate_cost_two_layer(
                    uploaded_files=uploaded_files,
                    doc_extractor=doc_extractor,
                    event_model=runtime_model,
                    extracted_texts=extracted_texts  # Can be None - uses page count heuristic
                )

                # Display layered breakdown
                st.markdown("### 📊 Cost Breakdown")

                # Show disclaimer if using page count estimation (no extraction run yet)
                if not extracted_texts:
                    st.info("ℹ️ **Pre-extraction estimate** based on page count (no document extraction run yet). Click \"Process Files\" to run extraction.")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "📄 Document Extraction",
                        two_layer_result["document_cost_display"],
                        help=f"{two_layer_result['document_extractor']} processing {two_layer_result['page_count']} pages"
                    )
                    from .document_page_estimator import get_confidence_message
                    confidence_msg = get_confidence_message(
                        two_layer_result['page_confidence'],
                        two_layer_result['page_count']
                    )
                    st.caption(confidence_msg)

                with col2:
                    st.metric(
                        "🤖 Event Extraction",
                        two_layer_result["event_cost_display"],
                        help=f"{two_layer_result['event_model']} processing {two_layer_result['tokens_total']:,} tokens"
                    )
                    if not extracted_texts:
                        st.caption(f"~{two_layer_result['tokens_total']:,} tokens (estimated from page count)")
                    else:
                        st.caption(f"~{two_layer_result['tokens_total']:,} tokens")

                with col3:
                    st.metric(
                        "💰 Total Estimated",
                        two_layer_result["total_cost_display"],
                        help="Combined document + event extraction cost"
                    )
                    st.caption(two_layer_result['note'])

            # === SINGLE-LAYER (Legacy: event extraction only, requires extracted_texts) ===
            elif extracted_texts:
                # Combine all extracted texts for total cost estimate
                combined_text = "\n\n".join(extracted_texts)

                if not combined_text.strip():
                    st.info("⚠️ No text available for cost estimation")
                    return

                # Calculate token estimate across ALL documents
                total_tokens = estimate_tokens(combined_text)

                # Display token estimate
                st.metric("Estimated Tokens", f"{total_tokens:,}", help="Based on 4 chars = 1 token heuristic")
            else:
                # No estimate available (need either two-layer params or extracted_texts)
                st.info("ℹ️ **No cost estimate available** - Upload files and select provider/model to see costs")
                return

            # Show model comparison or specific model estimate
            if (show_all_models or not runtime_model):
                # Show comparison table across recommended models
                st.subheader("Model Cost Comparison")

                # Use two-layer estimation if we have uploaded_files + doc_extractor
                # Otherwise fall back to single-layer (extracted_texts only)
                if uploaded_files and doc_extractor:
                    # Two-layer comparison: shows doc cost + event cost + total for each model
                    from .cost_estimator import estimate_all_models_two_layer  # noqa: E402

                    estimates = estimate_all_models_two_layer(
                        uploaded_files=uploaded_files,
                        doc_extractor=doc_extractor,
                        provider=provider,
                        recommended_only=True,
                        extracted_texts=extracted_texts  # Can be None
                    )

                    if estimates:
                        # Build two-layer comparison DataFrame
                        rows = []
                        for model_id, est in estimates.items():
                            if est.get("pricing_available"):
                                rows.append({
                                    "Model": est["display_name"],
                                    "Doc Cost": f"${est['document_cost']:.4f}",
                                    "Event Cost": f"${est['event_cost']:.4f}",
                                    "Total Cost": f"${est['total_cost']:.4f}",
                                    "Tokens": f"{est['tokens_total']:,}",
                                    "Provider": model_id.split("/")[0] if "/" in model_id else provider or "N/A"
                                })

                        if rows:
                            df = pd.DataFrame(rows)
                            # Sort by total cost
                            df["_sort_cost"] = [float(row["Total Cost"].replace("$", "")) for row in rows]
                            df = df.sort_values("_sort_cost").drop(columns=["_sort_cost"]).reset_index(drop=True)

                            st.dataframe(df, use_container_width=True, hide_index=True)

                            # Add note about doc cost being same across models
                            if rows:
                                doc_cost = rows[0]["Doc Cost"]
                                st.caption(f"ℹ️ Document extraction cost ({doc_cost}) is same for all models (same doc extractor: {doc_extractor})")

                            # Show cost range summary
                            costs = [float(row["Total Cost"].replace("$", "")) for row in rows]
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Cheapest", f"${min(costs):.4f}")
                            with col2:
                                st.metric("Most Expensive", f"${max(costs):.4f}")
                        else:
                            st.info("💡 Pricing unavailable for selected models")
                    else:
                        st.info("💡 No models found for cost estimation")

                elif extracted_texts:
                    # Single-layer comparison (legacy): event extraction only
                    combined_text = "\n\n".join(extracted_texts)

                    estimates = estimate_all_models(combined_text, provider=provider, recommended_only=True)

                    if estimates:
                        # Build comparison DataFrame
                        rows = []
                        for model_id, est in estimates.items():
                            if est.get("pricing_available"):
                                rows.append({
                                    "Model": est["display_name"],
                                    "Cost (USD)": f"${est['cost_usd']:.4f}",
                                    "Tokens": f"{est['tokens_total']:,}",
                                    "Provider": est["model_id"].split("/")[0] if "/" in est["model_id"] else provider or "N/A"
                                })

                        if rows:
                            df = pd.DataFrame(rows)
                            # Sort by cost
                            df["_sort_cost"] = [float(row["Cost (USD)"].replace("$", "")) for row in rows]
                            df = df.sort_values("_sort_cost").drop(columns=["_sort_cost"]).reset_index(drop=True)

                            st.dataframe(df, use_container_width=True, hide_index=True)

                            # Show cost range summary
                            costs = [float(row["Cost (USD)"].replace("$", "")) for row in rows]
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Cheapest", f"${min(costs):.4f}")
                            with col2:
                                st.metric("Most Expensive", f"${max(costs):.4f}")
                        else:
                            st.info("💡 Pricing unavailable for selected models")
                    else:
                        st.info("💡 No models found for cost estimation")

            elif extracted_texts and runtime_model:
                # Need combined_text for specific model estimate
                combined_text = "\n\n".join(extracted_texts)

                # Show estimate for specific model only
                estimate = estimate_cost(combined_text, runtime_model)

                if estimate.get("pricing_available"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Model", estimate["display_name"])

                    with col2:
                        st.metric("Estimated Cost", f"${estimate['cost_usd']:.4f}")

                    with col3:
                        st.metric("Tokens", f"{estimate['tokens_total']:,}")

                    # Show token breakdown
                    with st.expander("Token Breakdown", expanded=False):
                        st.write(f"**Input tokens**: {estimate['tokens_input']:,} (90%)")
                        st.write(f"**Output tokens**: {estimate['tokens_output']:,} (10%)")
                        st.caption("Split assumes extraction tasks with large input, small structured output")
                else:
                    st.info(f"💡 Pricing unavailable for {runtime_model}")

            st.caption("💡 **Tip**: These estimates help you compare costs before running expensive extractions")

    except Exception as e:
        logger.error(f"Cost estimation failed: {e}")
        # Non-blocking: Don't show error to user, just skip estimates


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
        use_container_width=True,
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
    st.dataframe(sample_df, use_container_width=True, hide_index=True)


# ====================
# Document Extraction Cache Management
# ====================

def get_cache_stats() -> Tuple[int, List[str]]:
    """
    Get document extraction cache statistics

    Returns:
        Tuple of (cache_size, list_of_cached_filenames)
    """
    cache = st.session_state.get('doc_extraction_cache', {})
    cache_size = len(cache)

    # Extract filenames from cache keys (format: filename:size:extractor)
    cached_files = []
    for cache_key in cache.keys():
        # Split cache key and get filename part
        parts = cache_key.split(':', 2)
        if parts:
            cached_files.append(parts[0])

    return cache_size, cached_files


def clear_document_cache() -> None:
    """
    Clear all document extraction cache entries
    """
    if 'doc_extraction_cache' in st.session_state:
        cache_size = len(st.session_state['doc_extraction_cache'])
        st.session_state['doc_extraction_cache'] = {}
        logger.info(f"💾 Cleared {cache_size} cache entries")


def display_cache_info(location: str = "sidebar") -> None:
    """
    Display cache information and clear button

    Args:
        location: Where to display ('sidebar' or 'main')
    """
    cache_size, cached_files = get_cache_stats()

    if location == "sidebar":
        if cache_size > 0:
            st.sidebar.caption(f"💾 **Document Cache**: {cache_size} file{'s' if cache_size != 1 else ''}")

            with st.sidebar.expander("Cached Files", expanded=False):
                for idx, filename in enumerate(cached_files, 1):
                    st.caption(f"{idx}. {filename}")

            if st.sidebar.button("🗑️ Clear Cache", help="Clear all cached document extractions"):
                clear_document_cache()
                st.rerun()
        else:
            st.sidebar.caption("💾 Cache: Empty")
    else:
        # Main area display
        if cache_size > 0:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"💾 **{cache_size} file{'s' if cache_size != 1 else ''} cached** - Switching LLMs will reuse extracted text")
            with col2:
                if st.button("Clear Cache"):
                    clear_document_cache()
                    st.rerun()


# ============================================================================
# TWO-STAGE COST-AWARE MODEL SELECTION (Post-Docling)
# ============================================================================

def show_cost_aware_model_selection(
    uploaded_files: List,
    provider: Optional[str] = None,
    runtime_model: Optional[str] = None,
    doc_extractor: Optional[str] = None
) -> Optional[str]:
    """
    Two-stage model selection with tiktoken-based cost estimation.

    Stage 1 (FREE): Extract text with Docling
    Stage 2 (INFORMED): Show exact costs for all models using tiktoken,
                        let user pick model based on cost/quality tradeoff

    This enables users to make informed decisions about cost vs accuracy
    BEFORE making expensive API calls.

    Args:
        uploaded_files: List of uploaded file objects
        provider: Default provider (optional, for pipeline init)
        runtime_model: Default model (optional)
        doc_extractor: Document extractor ID ('docling', 'qwen_vl', 'gemini')

    Returns:
        Selected model_id string, or None if no selection made

    Example:
        >>> uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)
        >>> if uploaded_files:
        ...     selected_model = show_cost_aware_model_selection(
        ...         uploaded_files,
        ...         doc_extractor='docling'
        ...     )
        ...     if selected_model:
        ...         st.success(f"Processing with {selected_model}")
    """
    if not uploaded_files:
        return None

    # ========================================================================
    # STAGE 1: Extract Text (FREE)
    # ========================================================================
    st.subheader("📄 Stage 1: Extract Text (FREE)")

    if st.button(
        "🔽 Extract Text with Docling",
        help="Extract and parse text from documents (no API calls, completely free)",
        use_container_width=True
    ):
        # Extract using existing function
        extracted_texts = extract_all_documents_for_estimation(
            uploaded_files=uploaded_files,
            provider=provider,
            runtime_model=runtime_model,
            doc_extractor=doc_extractor
        )

        if extracted_texts:
            # Store in session state for reuse
            st.session_state['extracted_texts_for_cost'] = extracted_texts
            st.success(f"✅ Extracted {len(extracted_texts)} document(s)")
            st.rerun()  # Refresh to show cost table
        else:
            st.error("❌ Text extraction failed")
            return None

    # ========================================================================
    # STAGE 2: Cost-Aware Model Selection
    # ========================================================================
    if 'extracted_texts_for_cost' in st.session_state:
        extracted_texts = st.session_state['extracted_texts_for_cost']

        st.divider()
        st.subheader("💰 Stage 2: Select Model by Cost & Quality")

        # Calculate costs with tiktoken (exact token counts)
        from .cost_estimator import estimate_all_models_with_tiktoken
        from .cost_comparison import show_cost_comparison_selector

        with st.spinner("📊 Calculating costs with tiktoken..."):
            cost_table = estimate_all_models_with_tiktoken(
                extracted_texts=extracted_texts,
                output_ratio=0.10
            )

        if cost_table:
            st.info(
                f"📋 Based on {len(extracted_texts)} document(s) with "
                f"{sum(len(t) for t in extracted_texts):,} characters"
            )

            # Show cost-aware selector
            selected_model = show_cost_comparison_selector(
                cost_table=cost_table,
                default_model="claude-3-haiku-20240307",  # Recommended: fastest production model
                show_summary=True
            )

            # Store selected model in session state
            if selected_model:
                st.session_state['selected_model_for_processing'] = selected_model
                return selected_model
        else:
            st.warning("⚠️ Could not calculate model costs")

    return None