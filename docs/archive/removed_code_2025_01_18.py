"""
ARCHIVED CODE - Removed January 18, 2025

This file preserves code removed during the dynamic catalog refactoring cleanup.

## Context

These functions were replaced by a unified, dynamic system that reads from the
event extractor catalog instead of using hardcoded provider-specific selectors.

## Replacement

All 4 provider-specific model selectors were replaced by:
- `create_unified_model_selector(provider, container)` in app.py:453

The unified selector dynamically loads models from MODEL_CATALOG (which itself
is built from src/core/model_catalog.py) and supports:
- Search with normalization (e.g., "gpt 5" matches "gpt-5")
- Advanced filters (category, badges, cost)
- Consistent UX across all providers
- Single source of truth for model metadata

## Removal Reason

These functions created maintenance burden:
- 4 separate implementations of nearly identical logic (~350 lines total)
- Hardcoded model lists that became stale
- Duplication of search/filter logic
- Changes required updates to all 4 functions

## Removed From

File: /Users/aks/docling_langextract_testing/app.py
Lines: 584-930, 1196-1207
Date: January 18, 2025
Removed By: redundancy-scanner agent analysis

## Verification

No references found via `grep` or tests. These functions were completely unused
after the unified selector was introduced.

---
"""

# ============================================================================
# REMOVED FUNCTION 1: create_anthropic_model_selector()
# Lines: 584-657 (74 lines)
# ============================================================================

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


# ============================================================================
# REMOVED FUNCTION 2: create_openai_model_selector()
# Lines: 660-732 (73 lines)
# ============================================================================

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


# ============================================================================
# REMOVED FUNCTION 3: create_langextract_model_selector()
# Lines: 735-806 (72 lines)
# ============================================================================

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


# ============================================================================
# REMOVED FUNCTION 4: create_openrouter_model_selector()
# Lines: 809-930 (122 lines)
# ============================================================================

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
            ("openai/gpt-oss-120b", "GPT-OSS 120B", "$0.31/M • 128K", "10/10 • Self-hostable"),
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
        - **Privacy/sovereignty concerns?** → Use **GPT-OSS 120B** (Apache 2.0, can self-host if needed)

        **Key Metrics**:
        - **Quality scores** (10/10, 9/10, 7/10): From Oct 2025 JSON mode + legal extraction tests
        - **⚠️ Warning symbol**: Model may extract fewer events on complex documents (test before production)
        - **Context window** (128K, 200K): Capacity for long documents (~100-200 pages)
        - **Cost** (/M = per million tokens): $0.03 to $3.00 range
        - **Speed**: Claude 3 Haiku extracts in 4.4s vs others at 14-36s

        ℹ️ Most models scored 10/10 or 9/10. GPT-OSS/Gemini/Cohere/Perplexity failed and were excluded.
        """)

    return selected_model


# ============================================================================
# REMOVED FUNCTION 5: _legacy_hardcoded_provider_ui_marker()
# Lines: 1196-1207 (12 lines)
# ============================================================================

def _legacy_hardcoded_provider_ui_marker():
    """
    MARKER: Lines 1038-1187 previously contained hardcoded provider buttons.
    This has been replaced with dynamic catalog-driven UI starting at line 1004.

    Benefits of new approach:
    - Toggle enabled=False → provider disappears (no code edits)
    - Add new provider → auto-appears in UI (just add catalog entry)
    - Recommended flag controls primary vs advanced placement
    - All metadata (icons, notes, status) from catalog
    """
    pass


# ============================================================================
# MIGRATION NOTES
# ============================================================================

"""
## How to Use the Unified Selector

Instead of calling provider-specific selectors:

OLD CODE:
```python
if provider == 'anthropic':
    selected_model = create_anthropic_model_selector()
elif provider == 'openai':
    selected_model = create_openai_model_selector()
elif provider == 'langextract':
    selected_model = create_langextract_model_selector()
elif provider == 'openrouter':
    selected_model = create_openrouter_model_selector()
```

NEW CODE:
```python
# Map event provider to MODEL_CATALOG provider name
model_catalog_map = {
    'langextract': 'google',
    'openrouter': 'openrouter',
    'openai': 'openai',
    'anthropic': 'anthropic',
    'deepseek': 'deepseek'
}

catalog_provider = model_catalog_map.get(provider)
if catalog_provider:
    selected_model = create_unified_model_selector(catalog_provider)
```

## Model Metadata Migration

All model metadata is now centralized in:
- `/Users/aks/docling_langextract_testing/src/core/model_catalog.py`

To add new models, update the catalog (single source of truth) rather than
editing individual selector functions.

## Testing After Migration

All tests continue to pass without modification - the unified selector
maintains API compatibility with the old selectors.

Run:
```bash
uv run python -m tests.test_event_extractor_registry
```

Expected: All 26 tests passing
"""
