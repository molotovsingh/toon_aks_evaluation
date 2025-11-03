"""
Cost Comparison UI Component - Interactive model selection by cost and quality

This module provides Streamlit UI components for cost-aware model selection.
After Docling extraction, users see exact costs for ALL models and can choose
based on their budget/accuracy preferences.

Features:
- Cost table organized by category (Budget/Production/Ground Truth)
- Quality score and speed metrics alongside cost
- Token count breakdown (input/output)
- Easy model selection with visual indicators
- Sortable/filterable results

Usage:
    from src.ui.cost_comparison import show_cost_comparison_selector

    cost_table = estimate_all_models_with_tiktoken(extracted_texts)
    selected_model = show_cost_comparison_selector(cost_table)

    if selected_model:
        # Process with selected_model
        events = extractor.extract_events(text, selected_model)
"""

import streamlit as st
from typing import List, Dict, Optional
import pandas as pd


def show_cost_comparison_selector(
    cost_table: List[Dict],
    default_model: Optional[str] = None,
    show_summary: bool = True
) -> Optional[str]:
    """
    Display interactive cost comparison table with model selection.

    Shows all models organized by category (Budget/Production/Ground Truth)
    with costs, quality scores, and speed metrics.

    Args:
        cost_table: List of cost dicts from estimate_all_models_with_tiktoken()
        default_model: Model ID to highlight as recommended (e.g., 'claude-3-haiku-20240307')
        show_summary: Whether to show cost summary stats (min/max/avg)

    Returns:
        Selected model_id string, or None if no selection made

    Example:
        >>> from src.ui.cost_estimator import estimate_all_models_with_tiktoken
        >>> cost_table = estimate_all_models_with_tiktoken(extracted_texts)
        >>> selected = show_cost_comparison_selector(
        ...     cost_table,
        ...     default_model="claude-3-haiku-20240307"
        ... )
        >>> if selected:
        ...     st.success(f"Processing with {selected}")
    """
    if not cost_table:
        st.warning("No models available for selection")
        return None

    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================
    if show_summary:
        st.subheader("💰 Cost Summary")

        # Calculate stats
        costs = [m["total_cost"] for m in cost_table]
        cheapest_cost = min(costs)
        most_expensive_cost = max(costs)
        avg_cost = sum(costs) / len(costs)

        cheapest_model = next(m for m in cost_table if m["total_cost"] == cheapest_cost)
        most_expensive_model = next(
            m for m in cost_table if m["total_cost"] == most_expensive_cost
        )

        # Display in columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "💵 Cheapest",
                f"${cheapest_cost:.4f}",
                f"{cheapest_model['display_name']}",
            )

        with col2:
            st.metric(
                "🏆 Most Expensive",
                f"${most_expensive_cost:.4f}",
                f"{most_expensive_model['display_name']}",
            )

        with col3:
            st.metric("📊 Average Cost", f"${avg_cost:.4f}")

        with col4:
            cost_ratio = most_expensive_cost / cheapest_cost if cheapest_cost > 0 else 0
            st.metric("📈 Cost Range", f"{cost_ratio:.1f}x")

        st.divider()

    # ========================================================================
    # CATEGORIZED MODEL SELECTION
    # ========================================================================
    st.subheader("🔍 Select Model by Cost & Quality")

    # Category definitions with display order
    categories = {
        "budget": {
            "display_name": "💵 Budget Tier (<$0.01)",
            "filter_fn": lambda m: m["total_cost"] < 0.01,
            "expanded": False,
        },
        "production": {
            "display_name": "🏭 Production Tier ($0.01-$0.05)",
            "filter_fn": lambda m: 0.01 <= m["total_cost"] < 0.05,
            "expanded": True,  # Open by default
        },
        "ground_truth": {
            "display_name": "🎯 Ground Truth Tier (>$0.05)",
            "filter_fn": lambda m: m["total_cost"] >= 0.05,
            "expanded": False,
        },
    }

    selected_model = None

    # Show models grouped by category
    for category_id, category_config in categories.items():
        category_models = [
            m for m in cost_table if category_config["filter_fn"](m)
        ]

        if not category_models:
            continue  # Skip empty categories

        with st.expander(
            category_config["display_name"],
            expanded=category_config["expanded"],
        ):
            # Create table data
            table_data = []
            for model in category_models:
                # Format cost with visual indicator
                cost_display = f"${model['total_cost']:.4f}"

                # Quality badge
                quality = model.get("quality_score", "N/A")

                # Speed indicator
                speed = ""
                if model.get("speed_seconds"):
                    speed = f"{model['speed_seconds']:.1f}s"

                table_data.append({
                    "Model": model["display_name"],
                    "Provider": model["provider"],
                    "Cost": cost_display,
                    "Quality": quality,
                    "Speed": speed,
                    "Input (K)": f"{model['input_tokens'] // 1000}",
                    "Output": f"{model['output_tokens']}",
                    "Select": "▶️",  # Interactive column
                })

            # Display as table with selection buttons
            for i, row_data in enumerate(table_data):
                model_obj = category_models[i]
                model_id = model_obj["model_id"]

                # Create three columns: info, metrics, select button
                col1, col2, col3, col4 = st.columns([3, 2, 1, 1])

                with col1:
                    # Highlight default model
                    if model_id == default_model:
                        st.markdown(
                            f"**✨ {row_data['Model']}** ({row_data['Provider']})"
                        )
                    else:
                        st.markdown(f"**{row_data['Model']}** ({row_data['Provider']})")

                with col2:
                    st.text(f"${model_obj['total_cost']:.4f} | {row_data['Quality']}")

                with col3:
                    if row_data["Speed"]:
                        st.caption(row_data["Speed"])

                with col4:
                    # Selection button
                    if st.button(f"Select", key=f"select_{model_id}", use_container_width=True):
                        selected_model = model_id
                        st.session_state.selected_model = model_id

                # Optional: Show token breakdown on expand
                with st.expander("Details", key=f"details_{model_id}"):
                    detail_cols = st.columns(3)
                    with detail_cols[0]:
                        st.metric(
                            "Input Tokens",
                            f"{model_obj['input_tokens']:,}",
                        )
                    with detail_cols[1]:
                        st.metric(
                            "Output Tokens",
                            f"{model_obj['output_tokens']:,}",
                        )
                    with detail_cols[2]:
                        st.metric(
                            "Total Tokens",
                            f"{model_obj['input_tokens'] + model_obj['output_tokens']:,}",
                        )

                    # Cost breakdown
                    st.caption("**Cost Breakdown:**")
                    breakdown_cols = st.columns(2)
                    with breakdown_cols[0]:
                        st.text(f"Input cost: ${model_obj['input_cost']:.6f}")
                    with breakdown_cols[1]:
                        st.text(f"Output cost: ${model_obj['output_cost']:.6f}")

                st.divider()

    # ========================================================================
    # SELECTION CONFIRMATION
    # ========================================================================
    if selected_model or "selected_model" in st.session_state:
        final_selection = selected_model or st.session_state.get("selected_model")
        selected_info = next(
            (m for m in cost_table if m["model_id"] == final_selection), None
        )

        if selected_info:
            quality = selected_info.get("quality_score", "N/A")
            st.success(
                f"✅ Selected: **{selected_info['display_name']}** "
                f"(${selected_info['total_cost']:.4f}, {quality})"
            )
            return final_selection

    return None


def show_cost_comparison_table(cost_table: List[Dict]) -> None:
    """
    Display all models as a sortable/filterable dataframe table.

    Alternative to categorized selector - shows all models in one table
    with sorting and filtering capabilities.

    Args:
        cost_table: List of cost dicts from estimate_all_models_with_tiktoken()

    Example:
        >>> from src.ui.cost_estimator import estimate_all_models_with_tiktoken
        >>> cost_table = estimate_all_models_with_tiktoken(extracted_texts)
        >>> show_cost_comparison_table(cost_table)
    """
    if not cost_table:
        st.warning("No models available")
        return

    # Convert to dataframe for better display
    df_data = []
    for model in cost_table:
        df_data.append({
            "Model": model["display_name"],
            "Provider": model["provider"],
            "Category": model["category"],
            "Cost (USD)": f"${model['total_cost']:.4f}",
            "Input Tokens": f"{model['input_tokens']:,}",
            "Output Tokens": f"{model['output_tokens']:,}",
            "Quality": model.get("quality_score", "N/A"),
            "Speed (s)": model.get("speed_seconds") or "N/A",
            "Recommended": "✓" if model.get("recommended") else "-",
        })

    df = pd.DataFrame(df_data)

    # Display as interactive table
    st.subheader("📊 All Models Comparison")
    st.dataframe(
        df,
        use_container_width=True,
        height=400,
    )


def show_cost_breakdown(cost_table: List[Dict], selected_model_id: str) -> None:
    """
    Show detailed cost breakdown for a selected model.

    Displays input/output tokens and cost split for the selected model
    compared to other models in same category.

    Args:
        cost_table: List of cost dicts
        selected_model_id: Selected model ID for detailed view
    """
    selected = next(
        (m for m in cost_table if m["model_id"] == selected_model_id), None
    )

    if not selected:
        st.warning("Model not found")
        return

    # ========================================================================
    # DETAILED BREAKDOWN
    # ========================================================================
    st.subheader(f"💰 Cost Breakdown: {selected['display_name']}")

    # Main metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Cost", f"${selected['total_cost']:.6f}")

    with col2:
        st.metric("Input Cost", f"${selected['input_cost']:.6f}")

    with col3:
        st.metric("Output Cost", f"${selected['output_cost']:.6f}")

    with col4:
        ratio = (
            selected["output_cost"] / selected["input_cost"]
            if selected["input_cost"] > 0
            else 0
        )
        st.metric("Output/Input Ratio", f"{ratio:.1%}")

    st.divider()

    # Token breakdown
    st.subheader("Token Count")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Input Tokens", f"{selected['input_tokens']:,}")

    with col2:
        st.metric("Output Tokens", f"{selected['output_tokens']:,}")

    with col3:
        total = selected["input_tokens"] + selected["output_tokens"]
        st.metric("Total Tokens", f"{total:,}")

    # Comparison with cheapest/most expensive
    st.divider()
    st.subheader("📊 Comparison")

    # Create local sorted copy to avoid assuming cost_table is pre-sorted
    sorted_by_cost = sorted(cost_table, key=lambda m: m['total_cost'])
    cheapest = sorted_by_cost[0]
    most_expensive = sorted_by_cost[-1]

    col1, col2, col3 = st.columns(3)

    with col1:
        ratio = selected["total_cost"] / cheapest["total_cost"] if cheapest["total_cost"] > 0 else 0
        st.metric(
            "vs Cheapest",
            f"{ratio:.1f}x",
            f"{cheapest['display_name']}",
        )

    with col2:
        ratio = most_expensive["total_cost"] / selected["total_cost"] if selected["total_cost"] > 0 else 0
        st.metric(
            "vs Most Expensive",
            f"{ratio:.1f}x",
            f"{most_expensive['display_name']}",
        )

    with col3:
        savings = most_expensive["total_cost"] - selected["total_cost"]
        st.metric(
            "Savings vs Most Expensive",
            f"${savings:.6f}",
        )
