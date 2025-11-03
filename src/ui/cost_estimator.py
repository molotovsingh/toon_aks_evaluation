"""
Cost Estimator Module - Pre-extraction cost estimation for legal event extraction

Provides rough order-of-magnitude cost estimates before running expensive LLM extractions.
Calculates both Layer 1 (document extraction) and Layer 2 (event extraction) costs.

Usage:
    from src.ui.cost_estimator import estimate_cost_two_layer, estimate_cost

    # Two-layer estimate (document + event extraction)
    result = estimate_cost_two_layer(
        uploaded_files=[file1, file2],
        doc_extractor='qwen_vl',
        event_model='gpt-4o-mini'
    )
    print(f"Document extraction: ${result['document_cost']:.4f}")
    print(f"Event extraction: ${result['event_cost']:.4f}")
    print(f"Total: ${result['total_cost']:.4f}")

    # Legacy single-layer estimate (event extraction only)
    result = estimate_cost(extracted_text, "gpt-4o-mini")
    print(f"Estimated cost: ${result['cost_usd']:.4f}")
"""

from typing import Dict, Optional, List, Union, BinaryIO
from pathlib import Path
import logging

from ..core.model_catalog import get_model_catalog, ModelEntry
from ..core.document_extractor_catalog import get_doc_extractor_catalog
from .document_page_estimator import estimate_document_pages

logger = logging.getLogger(__name__)

# ============================================================================
# TOKEN ESTIMATION HEURISTICS
# ============================================================================

# Character-to-token ratio (conservative estimate: 4 chars = 1 token)
# Based on OpenAI tokenizer analysis: English text typically ~3.5-4.5 chars/token
CHARS_PER_TOKEN = 4.0

# Input/output token split assumption (90% input, 10% output for extraction tasks)
# Legal event extraction is mostly reading (input) with structured output (small)
INPUT_TOKEN_RATIO = 0.9
OUTPUT_TOKEN_RATIO = 0.1


def estimate_tokens(text: str) -> int:
    """
    Estimate token count from text using character-based heuristic.

    Uses conservative 4 chars = 1 token rule. Real tokenizers may vary by ±20%,
    but this provides reasonable order-of-magnitude guidance.

    Args:
        text: Input text to estimate

    Returns:
        Estimated token count (integer)

    Example:
        >>> estimate_tokens("Hello world")
        3  # 11 chars / 4 = 2.75 → 3 tokens
    """
    if not text:
        return 0

    char_count = len(text)
    token_estimate = int(char_count / CHARS_PER_TOKEN)

    return max(token_estimate, 1)  # Minimum 1 token


def split_input_output_tokens(total_tokens: int) -> tuple[int, int]:
    """
    Split total tokens into input/output using extraction task assumptions.

    For legal event extraction:
    - Input: Full document text (90%)
    - Output: Structured JSON events (10%)

    Args:
        total_tokens: Total estimated token usage

    Returns:
        Tuple of (input_tokens, output_tokens)

    Example:
        >>> split_input_output_tokens(1000)
        (900, 100)
    """
    input_tokens = int(total_tokens * INPUT_TOKEN_RATIO)
    output_tokens = int(total_tokens * OUTPUT_TOKEN_RATIO)

    return input_tokens, output_tokens


# ============================================================================
# COST ESTIMATION
# ============================================================================

def estimate_cost(
    text: str,
    model_id: str,
    input_output_split: Optional[tuple[int, int]] = None
) -> Dict[str, any]:
    """
    Estimate cost for processing text with a specific model.

    Args:
        text: Input text to process
        model_id: Model identifier (e.g., 'gpt-4o-mini', 'claude-sonnet-4-5')
        input_output_split: Override default (input_tokens, output_tokens) split

    Returns:
        Dict with estimation results:
        {
            "model_id": str,
            "tokens_total": int,
            "tokens_input": int,
            "tokens_output": int,
            "cost_usd": float,
            "cost_display": str,
            "pricing_available": bool,
            "note": str  # Caveat about estimate accuracy
        }

    Example:
        >>> result = estimate_cost("Long legal document...", "gpt-4o-mini")
        >>> print(f"Estimated: ${result['cost_usd']:.4f} ({result['tokens_total']} tokens)")
    """
    catalog = get_model_catalog()

    # Get model metadata
    model = catalog.get_model(model_id)
    if not model:
        logger.warning(f"Model {model_id} not found in catalog")
        return {
            "model_id": model_id,
            "display_name": model_id,  # Use model_id as fallback
            "tokens_total": 0,
            "tokens_input": 0,
            "tokens_output": 0,
            "cost_usd": 0.0,
            "cost_display": "Unknown model",
            "pricing_available": False,
            "note": f"Model '{model_id}' not found in catalog"
        }

    # Estimate tokens
    if input_output_split:
        # Use custom split (tokens_total is sum of custom values)
        input_tokens, output_tokens = input_output_split
        total_tokens = input_tokens + output_tokens
    else:
        # Use default estimation
        total_tokens = estimate_tokens(text)
        input_tokens, output_tokens = split_input_output_tokens(total_tokens)

    # Get pricing from catalog
    pricing = catalog.get_pricing(model_id)

    if not pricing:
        return {
            "model_id": model_id,
            "display_name": model.display_name,
            "tokens_total": total_tokens,
            "tokens_input": input_tokens,
            "tokens_output": output_tokens,
            "cost_usd": 0.0,
            "cost_display": model.cost_display or "Pricing unavailable",
            "pricing_available": False,
            "note": "Pricing data not available in catalog"
        }

    # Calculate cost (pricing is per 1M tokens)
    cost_input = (input_tokens / 1_000_000) * pricing["cost_input_per_1m"]
    cost_output = (output_tokens / 1_000_000) * pricing["cost_output_per_1m"]
    total_cost = cost_input + cost_output

    return {
        "model_id": model_id,
        "display_name": model.display_name,
        "tokens_total": total_tokens,
        "tokens_input": input_tokens,
        "tokens_output": output_tokens,
        "cost_usd": total_cost,
        "cost_display": f"${total_cost:.4f}" if total_cost > 0 else model.cost_display,
        "pricing_available": True,
        "note": "Estimates only — actual billing may vary by ±20%"
    }


def estimate_all_models(
    text: str,
    provider: Optional[str] = None,
    category: Optional[str] = None,
    recommended_only: bool = False
) -> Dict[str, Dict]:
    """
    Estimate costs across multiple models for comparison.

    Useful for showing users cost differences before selecting a provider/model.

    Args:
        text: Input text to process
        provider: Filter by provider ('anthropic', 'openai', 'google', 'openrouter', 'deepseek')
        category: Filter by category ('Ground Truth', 'Production', 'Budget')
        recommended_only: Only include models marked as recommended

    Returns:
        Dict mapping model_id to cost estimate dict
        {
            "gpt-4o-mini": {"cost_usd": 0.0012, "tokens_total": 8000, ...},
            "claude-3-haiku-20240307": {"cost_usd": 0.0020, "tokens_total": 8000, ...}
        }

    Example:
        >>> estimates = estimate_all_models(text, provider="openai")
        >>> sorted_by_cost = sorted(estimates.items(), key=lambda x: x[1]['cost_usd'])
        >>> cheapest = sorted_by_cost[0]
        >>> print(f"Cheapest: {cheapest[0]} at ${cheapest[1]['cost_usd']:.4f}")
    """
    catalog = get_model_catalog()

    # Query models with filters
    models = catalog.list_models(
        provider=provider,
        category=category,
        recommended_only=recommended_only
    )

    if not models:
        logger.warning(f"No models found for filters: provider={provider}, category={category}")
        return {}

    # Estimate cost for each model
    estimates = {}
    for model in models:
        estimate = estimate_cost(text, model.model_id)
        estimates[model.model_id] = estimate

    return estimates


def get_cost_summary(
    text: str,
    provider: Optional[str] = None
) -> Dict[str, any]:
    """
    Get summary statistics for cost estimation across models.

    Args:
        text: Input text to process
        provider: Filter by provider (optional)

    Returns:
        Dict with summary stats:
        {
            "token_estimate": int,
            "models_analyzed": int,
            "cost_min": float,
            "cost_max": float,
            "cost_range_display": str,
            "cheapest_model": str,
            "most_expensive_model": str
        }
    """
    estimates = estimate_all_models(text, provider=provider, recommended_only=True)

    if not estimates:
        return {
            "token_estimate": estimate_tokens(text),
            "models_analyzed": 0,
            "cost_min": 0.0,
            "cost_max": 0.0,
            "cost_range_display": "No pricing available",
            "cheapest_model": None,
            "most_expensive_model": None
        }

    # Filter only models with pricing
    priced_estimates = {
        model_id: est for model_id, est in estimates.items()
        if est["pricing_available"]
    }

    if not priced_estimates:
        return {
            "token_estimate": estimate_tokens(text),
            "models_analyzed": len(estimates),
            "cost_min": 0.0,
            "cost_max": 0.0,
            "cost_range_display": "Pricing unavailable",
            "cheapest_model": None,
            "most_expensive_model": None
        }

    # Calculate stats
    costs = [(est["cost_usd"], model_id) for model_id, est in priced_estimates.items()]
    costs.sort()

    min_cost, cheapest_model = costs[0]
    max_cost, most_expensive_model = costs[-1]

    # Format display string
    if min_cost == max_cost:
        cost_range_display = f"${min_cost:.4f}"
    else:
        cost_range_display = f"${min_cost:.4f} - ${max_cost:.4f}"

    return {
        "token_estimate": estimate_tokens(text),
        "models_analyzed": len(priced_estimates),
        "cost_min": min_cost,
        "cost_max": max_cost,
        "cost_range_display": cost_range_display,
        "cheapest_model": cheapest_model,
        "most_expensive_model": most_expensive_model
    }


# ============================================================================
# TWO-LAYER COST ESTIMATION (Document + Event Extraction)
# ============================================================================

def estimate_cost_two_layer(
    uploaded_files: List[Union[Path, str, BinaryIO]],
    doc_extractor: str,
    event_model: str,
    extracted_texts: Optional[List[str]] = None
) -> Dict[str, any]:
    """
    Estimate total cost including both document extraction (Layer 1) and event extraction (Layer 2).

    Args:
        uploaded_files: List of file paths or file-like objects
        doc_extractor: Document extractor ID ('docling', 'qwen_vl', 'gemini')
        event_model: Event extraction model ID (from model catalog)
        extracted_texts: Pre-extracted texts (if available, skips Layer 1 estimate)

    Returns:
        Dict with layered cost breakdown:
        {
            "document_cost": float,          # Layer 1 cost
            "document_cost_display": str,    # Layer 1 display
            "event_cost": float,             # Layer 2 cost
            "event_cost_display": str,       # Layer 2 display
            "total_cost": float,             # Combined cost
            "total_cost_display": str,       # Combined display
            "page_count": int,               # Total estimated pages
            "page_confidence": str,          # 'high', 'medium', 'low'
            "tokens_total": int,             # Event extraction tokens
            "pricing_available": bool,       # Both layers have pricing
            "note": str                      # Accuracy caveat
        }
    """
    doc_catalog = get_doc_extractor_catalog()

    # === Layer 1: Document Extraction Cost ===
    total_pages = 0
    page_confidences = []

    for file in uploaded_files:
        page_count, confidence = estimate_document_pages(file)
        total_pages += page_count
        page_confidences.append(confidence)

    # Aggregate confidence: worst confidence wins
    if page_confidences:
        confidence_priority = {'low': 3, 'medium': 2, 'high': 1}
        worst_confidence = max(page_confidences, key=lambda c: confidence_priority.get(c, 3))
    else:
        worst_confidence = "low"  # Default for empty file list

    doc_cost_result = doc_catalog.estimate_cost(doc_extractor, total_pages)

    # === Layer 2: Event Extraction Cost ===
    if extracted_texts:
        # Use pre-extracted texts
        combined_text = "\n\n".join(extracted_texts)
    else:
        # Estimate text length from page count (rough: 2000 chars/page)
        estimated_chars = total_pages * 2000
        combined_text = "x" * estimated_chars  # Placeholder for token estimation

    event_cost_result = estimate_cost(combined_text, event_model)

    # === Combined Result ===
    total_cost = doc_cost_result["cost_usd"] + event_cost_result["cost_usd"]

    # Determine pricing availability
    pricing_available = (
        doc_cost_result["pricing_available"] and
        event_cost_result["pricing_available"]
    )

    # Build comprehensive result
    return {
        # Layer 1 (Document Extraction)
        "document_cost": doc_cost_result["cost_usd"],
        "document_cost_display": doc_cost_result["cost_display"],
        "document_extractor": doc_cost_result["display_name"],

        # Layer 2 (Event Extraction)
        "event_cost": event_cost_result["cost_usd"],
        "event_cost_display": event_cost_result["cost_display"],
        "event_model": event_cost_result["display_name"],
        "tokens_total": event_cost_result["tokens_total"],
        "tokens_input": event_cost_result["tokens_input"],
        "tokens_output": event_cost_result["tokens_output"],

        # Combined
        "total_cost": total_cost,
        "total_cost_display": f"${total_cost:.4f}" if total_cost > 0 else "FREE",

        # Metadata
        "page_count": total_pages,
        "page_confidence": worst_confidence,
        "file_count": len(uploaded_files),
        "pricing_available": pricing_available,

        # Notes
        "note": (
            "Two-layer estimate (document + event extraction). "
            "Actual costs may vary ±20% for events, ±30% for document processing."
        )
    }


def estimate_all_models_two_layer(
    uploaded_files: List[Union[Path, str, BinaryIO]],
    doc_extractor: str,
    provider: Optional[str] = None,
    recommended_only: bool = False,
    extracted_texts: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    Estimate costs across multiple event extraction models (all with same doc extractor).

    Args:
        uploaded_files: List of file paths or file-like objects
        doc_extractor: Document extractor ID
        provider: Filter event models by provider (optional)
        recommended_only: Only include recommended models
        extracted_texts: Pre-extracted texts (if available)

    Returns:
        Dict mapping model_id to two-layer cost estimate dict
    """
    model_catalog = get_model_catalog()
    models = model_catalog.list_models(
        provider=provider,
        recommended_only=recommended_only
    )

    estimates = {}
    for model in models:
        try:
            estimate = estimate_cost_two_layer(
                uploaded_files=uploaded_files,
                doc_extractor=doc_extractor,
                event_model=model.model_id,
                extracted_texts=extracted_texts
            )
            estimates[model.model_id] = estimate
        except Exception as e:
            logger.warning(f"Failed to estimate cost for {model.model_id}: {e}")

    return estimates


# ============================================================================
# TIKTOKEN-BASED EXACT TOKEN COUNTING (Post-Docling)
# ============================================================================

def estimate_all_models_with_tiktoken(
    extracted_texts: List[str],
    output_ratio: float = 0.10
) -> List[Dict]:
    """
    Calculate costs for ALL models using EXACT token counts from tiktoken.

    This function should be called AFTER Docling extraction is complete.
    It uses tiktoken (OpenAI's official library) for precise token counting,
    then calculates costs across all available models.

    Provides a sortable cost table for cost-aware model selection UI.

    Args:
        extracted_texts: List of extracted document texts (from Docling)
        output_ratio: Estimated output/input ratio for cost calculation
                     (default 0.10 = 10% of input for event extraction)

    Returns:
        List of dicts, sorted by total cost (cheapest first):
        [
            {
                'model_id': 'deepseek-chat',
                'display_name': 'DeepSeek Chat',
                'provider': 'deepseek',
                'category': 'Production',
                'tier': 'production',
                'input_tokens': 8234,
                'output_tokens': 823,
                'input_cost': 0.0012,
                'output_cost': 0.0002,
                'total_cost': 0.0014,
                'cost_display': '$0.0014',
                'quality_score': '9/10',
                'speed_seconds': 5.2,
                'supports_json': True
            },
            ...
        ]

    Example:
        >>> from src.core.legal_pipeline_refactored import DoclingAdapter
        >>> doc_extractor = DoclingAdapter()
        >>> extracted = [doc_extractor.extract(f).markdown for f in files]
        >>> cost_table = estimate_all_models_with_tiktoken(extracted)
        >>> cheapest = cost_table[0]
        >>> print(f"Cheapest: {cheapest['display_name']} at ${cheapest['total_cost']:.4f}")
    """
    try:
        from ..utils.token_counter import count_tokens_batch, estimate_output_tokens
    except ImportError:
        logger.warning("tiktoken not available, falling back to character heuristic")
        return estimate_all_models_with_heuristic(extracted_texts, output_ratio)

    catalog = get_model_catalog()
    results = []

    for model in catalog.list_models():  # All models, no filters
        model_id = model.model_id

        try:
            # Get exact input token count using tiktoken
            input_tokens = count_tokens_batch(extracted_texts, model_id)

            # Estimate output tokens based on task type
            output_tokens = estimate_output_tokens(input_tokens, "event_extraction")

            # Get pricing from catalog
            pricing = catalog.get_pricing(model_id)

            if not pricing:
                logger.debug(f"Skipping {model_id}: no pricing data")
                continue

            # Calculate costs
            input_cost = (input_tokens / 1_000_000) * pricing.get("cost_input_per_1m", 0)
            output_cost = (output_tokens / 1_000_000) * pricing.get("cost_output_per_1m", 0)
            total_cost = input_cost + output_cost

            # Build result entry
            results.append({
                'model_id': model_id,
                'display_name': model.display_name,
                'provider': model.provider,
                'category': model.category,
                'tier': str(model.tier.value),
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'input_cost': input_cost,
                'output_cost': output_cost,
                'total_cost': total_cost,
                'cost_display': f"${total_cost:.4f}" if total_cost > 0 else model.cost_display,
                'quality_score': model.quality_score or "N/A",
                'speed_seconds': model.extraction_speed_seconds,
                'supports_json': model.supports_json_mode,
                'recommended': model.recommended,
            })

        except Exception as e:
            logger.debug(f"Error calculating cost for {model_id}: {str(e)}")
            continue

    # Sort by total cost (cheapest first)
    results.sort(key=lambda x: x['total_cost'])

    return results


def estimate_all_models_with_heuristic(
    extracted_texts: List[str],
    output_ratio: float = 0.10
) -> List[Dict]:
    """
    Fallback: Calculate costs using character-based heuristic (no tiktoken).

    Uses when tiktoken is not available or for quick estimates.
    Less accurate than tiktoken but requires no additional dependencies.

    Args:
        extracted_texts: List of extracted document texts
        output_ratio: Estimated output/input ratio

    Returns:
        List of cost estimates (same format as estimate_all_models_with_tiktoken)
    """
    catalog = get_model_catalog()
    results = []

    # Estimate total tokens using character heuristic
    total_text = "".join(extracted_texts)
    total_tokens = estimate_tokens(total_text)
    input_tokens = int(total_tokens * (1.0 - output_ratio))
    output_tokens = int(total_tokens * output_ratio)

    for model in catalog.list_models():
        model_id = model.model_id
        pricing = catalog.get_pricing(model_id)

        if not pricing:
            continue

        # Calculate costs
        input_cost = (input_tokens / 1_000_000) * pricing.get("cost_input_per_1m", 0)
        output_cost = (output_tokens / 1_000_000) * pricing.get("cost_output_per_1m", 0)
        total_cost = input_cost + output_cost

        results.append({
            'model_id': model_id,
            'display_name': model.display_name,
            'provider': model.provider,
            'category': model.category,
            'tier': str(model.tier.value),
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': total_cost,
            'cost_display': f"${total_cost:.4f}" if total_cost > 0 else model.cost_display,
            'quality_score': model.quality_score or "N/A",
            'speed_seconds': model.extraction_speed_seconds,
            'supports_json': model.supports_json_mode,
            'recommended': model.recommended,
        })

    results.sort(key=lambda x: x['total_cost'])
    return results


# ============================================================================
# CALIBRATION HELPERS (for future refinement)
# ============================================================================

def calculate_accuracy(estimated_tokens: int, actual_tokens: int) -> float:
    """
    Calculate estimation accuracy as percentage error.

    Use this with actual token counts from metadata to calibrate the estimator.

    Args:
        estimated_tokens: Token count from estimate_tokens()
        actual_tokens: Actual token count from LLM API response

    Returns:
        Accuracy as float (1.0 = perfect, 0.8 = 20% error)

    Example:
        >>> accuracy = calculate_accuracy(estimated=1000, actual=950)
        >>> print(f"Estimation accuracy: {accuracy:.1%}")  # "95.0%"
    """
    if actual_tokens == 0:
        return 0.0

    error = abs(estimated_tokens - actual_tokens) / actual_tokens
    accuracy = max(0.0, 1.0 - error)

    return accuracy


def suggest_calibration_factor(
    estimated_tokens_list: List[int],
    actual_tokens_list: List[int]
) -> float:
    """
    Suggest calibration factor based on historical estimates vs actuals.

    Args:
        estimated_tokens_list: List of estimated token counts
        actual_tokens_list: List of actual token counts (same length)

    Returns:
        Suggested multiplier for CHARS_PER_TOKEN (e.g., 0.9 means reduce by 10%)

    Example:
        >>> factor = suggest_calibration_factor([1000, 2000], [1100, 2200])
        >>> print(f"Recommended adjustment: {factor:.2f}x")
    """
    if not estimated_tokens_list or not actual_tokens_list:
        return 1.0

    if len(estimated_tokens_list) != len(actual_tokens_list):
        logger.error("Calibration lists must have same length")
        return 1.0

    # Calculate average ratio (actual / estimated)
    ratios = []
    for est, act in zip(estimated_tokens_list, actual_tokens_list):
        if est > 0:
            ratios.append(act / est)

    if not ratios:
        return 1.0

    avg_ratio = sum(ratios) / len(ratios)

    return avg_ratio
