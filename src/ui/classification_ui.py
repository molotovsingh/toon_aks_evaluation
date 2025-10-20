"""
Classification UI Components (Layer 1.5)

Provides Streamlit UI components for document classification configuration.
Mirrors pattern from app.py provider selection (Layer 2) and document extractor
selection (Layer 1) for architectural consistency.

Usage:
    from src.ui.classification_ui import (
        create_classification_config,
        show_classification_results,
        filter_documents_by_type
    )

    # In app.py:
    model_id, prompt_variant = create_classification_config()
    selected_types = show_classification_results(classifications)
"""

import streamlit as st
import os
import pandas as pd
from typing import List, Dict, Any, Set, Tuple

from src.core.classification_catalog import get_classification_catalog
from src.core.prompt_registry import get_prompt_variant, list_prompt_variants

import logging
logger = logging.getLogger(__name__)


def create_classification_config() -> Tuple[str, str]:
    """
    Create classification configuration UI.

    Mirrors provider selection pattern from app.py:655-839.
    Displays:
    - Model selection buttons (side-by-side)
    - Model metadata (cost, speed, labels/doc)
    - Recommended prompt variant
    - Advanced prompt override (optional)

    Returns:
        tuple: (model_id, prompt_variant_name)

    Example:
        >>> model_id, prompt = create_classification_config()
        >>> # Returns: ("anthropic/claude-3-haiku", "comprehensive")
    """
    st.markdown(
        '<div class="section-header">🏷️ Document Classification</div>',
        unsafe_allow_html=True
    )

    catalog = get_classification_catalog()
    enabled_models = catalog.list_models(enabled=True, recommended_only=True)

    if not enabled_models:
        st.error("❌ No classification models available. Check catalog configuration.")
        return None, None

    # Initialize session state
    if 'selected_classification_model' not in st.session_state:
        st.session_state.selected_classification_model = enabled_models[0].model_id

    # Model selection (side-by-side buttons - mirrors app.py:714-748)
    num_models = len(enabled_models)
    cols = st.columns(num_models)

    for idx, model_entry in enumerate(enabled_models):
        with cols[idx]:
            is_selected = st.session_state.selected_classification_model == model_entry.model_id

            if st.button(
                f"{model_entry.display_name}",
                key=f"clf_model_{model_entry.model_id}",
                type="primary" if is_selected else "secondary",
                use_container_width=True
            ):
                if st.session_state.selected_classification_model != model_entry.model_id:
                    st.session_state.selected_classification_model = model_entry.model_id
                    st.rerun()

            # Show metadata (mirrors provider button captions)
            st.caption(f"💰 ${model_entry.cost_per_1m}/M • ⚡ {model_entry.speed}")
            st.caption(f"📊 {model_entry.typical_labels_per_doc} labels/doc")

            # Show recommended prompt variant
            prompt_variant = get_prompt_variant(model_entry.recommended_prompt)
            st.caption(f"🎯 {prompt_variant.name}")

            # Show use case badge
            use_case_display = {
                "search_discovery": "🔍 Search",
                "routing_triage": "🎯 Routing",
                "comprehensive_analysis": "📊 Analysis"
            }
            st.caption(use_case_display.get(model_entry.primary_use_case, ""))

    # Get selected model
    selected_model_entry = catalog.get_model(
        st.session_state.selected_classification_model
    )

    # Advanced: Override prompt (optional, collapsed by default)
    with st.expander("🔧 Advanced: Prompt Override", expanded=False):
        st.caption("Override the recommended prompt variant for this model")

        custom_prompt = st.selectbox(
            "Prompt Variant",
            options=list_prompt_variants(),
            format_func=lambda x: get_prompt_variant(x).name,
            index=list_prompt_variants().index(selected_model_entry.recommended_prompt),
            key="classification_prompt_override"
        )

        use_custom = st.checkbox(
            "Use custom prompt",
            key="use_custom_classification_prompt",
            help="Enable to override the model's recommended prompt"
        )

        if use_custom:
            selected_prompt = custom_prompt
            st.info(f"Using custom prompt: **{get_prompt_variant(custom_prompt).name}**")
        else:
            selected_prompt = selected_model_entry.recommended_prompt

    # Use recommended prompt if advanced section not used
    if not use_custom:
        selected_prompt = selected_model_entry.recommended_prompt

    return st.session_state.selected_classification_model, selected_prompt


def show_classification_results(
    classifications: List[Dict[str, Any]]
) -> None:
    """
    Display classification results as metadata summary.

    Classifications will be added as 6th column to final events table,
    enabling downstream filtering and analysis in Excel/database tools.

    Args:
        classifications: List of classification dicts with keys:
                         - filename: str
                         - type: str (primary classification)
                         - confidence: float
                         - all_labels: List[str] (optional)

    Example:
        >>> classifications = [
        ...     {"filename": "doc1.pdf", "type": "Court Order/Judgment", "confidence": 0.92},
        ...     {"filename": "doc2.pdf", "type": "Correspondence", "confidence": 0.85}
        ... ]
        >>> show_classification_results(classifications)
    """
    if not classifications:
        st.warning("No classification results to display")
        return

    st.markdown("### 📊 Classification Results")
    st.caption("Document types will be added as metadata column in final events table")

    # Create summary dataframe
    df = pd.DataFrame(classifications)

    # Add confidence percentage column for display
    if 'confidence' in df.columns:
        df['Confidence %'] = (df['confidence'] * 100).round(1).astype(str) + '%'

    # Display results table
    display_cols = ['filename', 'type']
    if 'Confidence %' in df.columns:
        display_cols.append('Confidence %')
    if 'all_labels' in df.columns:
        display_cols.append('all_labels')

    st.dataframe(
        df[display_cols],
        width="stretch",
        hide_index=True,
        column_config={
            'filename': st.column_config.TextColumn('Document', width="large"),
            'type': st.column_config.TextColumn('Primary Type', width="medium"),
            'Confidence %': st.column_config.TextColumn('Confidence', width="small"),
            'all_labels': st.column_config.ListColumn('All Labels', width="medium")
        }
    )

    # Show type distribution statistics
    type_counts = df['type'].value_counts()
    total_docs = len(df)

    st.markdown("### 📈 Document Type Distribution")

    # Create metrics for top 3 types
    top_types = type_counts.head(3)
    cols = st.columns(len(top_types))

    for idx, (doc_type, count) in enumerate(top_types.items()):
        with cols[idx]:
            percentage = (count / total_docs * 100)
            st.metric(
                label=doc_type,
                value=f"{count} docs",
                delta=f"{percentage:.0f}%"
            )

    st.info(
        "💡 **Tip**: Document types will appear as a new column in the exported table. "
        "Use Excel pivot tables or database queries to filter/group events by type."
    )


def filter_documents_by_type(
    documents: List[Dict[str, Any]],
    classifications: List[Dict[str, Any]],
    selected_types: Set[str]
) -> List[Dict[str, Any]]:
    """
    Filter documents based on selected classification types.

    Args:
        documents: List of document dicts (must have 'filename' key)
        classifications: List of classification dicts (filename + type)
        selected_types: Set of document types to keep

    Returns:
        List of filtered document dicts

    Example:
        >>> docs = [{"filename": "doc1.pdf", "text": "..."}]
        >>> classifications = [{"filename": "doc1.pdf", "type": "Court Order/Judgment"}]
        >>> selected = {"Court Order/Judgment"}
        >>> filtered = filter_documents_by_type(docs, classifications, selected)
    """
    # Create filename -> type mapping
    type_map = {clf['filename']: clf['type'] for clf in classifications}

    # Filter documents
    filtered = [
        doc for doc in documents
        if type_map.get(doc['filename']) in selected_types
    ]

    logger.info(
        f"Filtered {len(documents)} documents → {len(filtered)} documents "
        f"(selected types: {selected_types})"
    )

    return filtered


def show_classification_cost_estimate(
    num_documents: int,
    avg_chars_per_doc: int,
    model_id: str
) -> None:
    """
    Show cost estimate for classification layer.

    Args:
        num_documents: Number of documents to classify
        avg_chars_per_doc: Average characters per document
        model_id: Classification model ID

    Example:
        >>> show_classification_cost_estimate(50, 2000, "anthropic/claude-3-haiku")
    """
    from src.core.classification_catalog import get_classification_catalog

    catalog = get_classification_catalog()
    model_entry = catalog.get_model(model_id)

    if not model_entry:
        return

    # Estimate tokens (4 chars ≈ 1 token)
    total_chars = num_documents * avg_chars_per_doc
    estimated_tokens = total_chars / 4

    # Calculate cost
    cost_usd = (estimated_tokens / 1_000_000) * model_entry.cost_per_1m

    st.markdown("### 💰 Classification Cost Estimate")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Documents", num_documents)

    with col2:
        st.metric("Estimated Tokens", f"{estimated_tokens:,.0f}")

    with col3:
        st.metric("Estimated Cost", f"${cost_usd:.4f}")

    st.caption(
        f"Using **{model_entry.display_name}** "
        f"(${model_entry.cost_per_1m}/M tokens). "
        f"Estimate assumes ~{avg_chars_per_doc} chars/doc, 4 chars/token."
    )
