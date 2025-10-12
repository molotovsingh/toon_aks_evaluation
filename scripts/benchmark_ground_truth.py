#!/usr/bin/env python3
"""
Ground Truth Benchmark Script

Tests ground truth models (Claude Sonnet 4.5, GPT-5, Gemini 2.5 Pro) against
production models to validate quality and cost trade-offs.

Usage:
    # Benchmark Anthropic models (Sonnet 4.5 vs Haiku)
    TEST_PROVIDER=anthropic uv run python scripts/benchmark_ground_truth.py

    # Benchmark OpenAI models (GPT-5 vs GPT-4o-mini)
    TEST_PROVIDER=openai uv run python scripts/benchmark_ground_truth.py

    # Benchmark LangExtract models (Gemini 2.5 Pro vs 2.0 Flash)
    TEST_PROVIDER=langextract uv run python scripts/benchmark_ground_truth.py

    # Run all providers
    uv run python scripts/benchmark_ground_truth.py --all

Requirements:
    - Relevant API keys set in .env (ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY)
    - Test document in sample_pdf/famas_dispute/ directory

Output:
    - Console report with quality metrics
    - JSON comparison file: output/ground_truth_benchmark_<provider>_<timestamp>.json
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.config import load_provider_config
from core.extractor_factory import create_extractors
from core.interfaces import EventRecord


@dataclass
class BenchmarkResult:
    """Results from a single extraction run"""
    model: str
    provider: str
    event_count: int
    events_with_dates: int
    events_with_citations: int
    extraction_time: float
    estimated_cost: float
    events: List[Dict[str, Any]]


@dataclass
class ComparisonMetrics:
    """Comparison metrics between ground truth and production"""
    ground_truth_model: str
    production_model: str
    ground_truth_events: int
    production_events: int
    event_recall: float  # production_events / ground_truth_events
    date_match_count: int
    date_accuracy: float  # matched_dates / ground_truth_dates
    citation_match_count: int
    citation_recall: float  # production_citations / ground_truth_citations
    avg_jaccard_similarity: float
    pass_fail: str  # "PASS", "CONDITIONAL_PASS", or "FAIL"


def load_test_document() -> Tuple[str, Dict[str, Any]]:
    """
    Load test document for benchmarking

    Returns:
        Tuple of (text_content, metadata)
    """
    # Use Famas arbitration PDF as standard test document
    test_file = Path("sample_pdf/famas_dispute/Answer to Request for Arbitration.pdf")

    if not test_file.exists():
        raise FileNotFoundError(
            f"Test document not found: {test_file}\n"
            f"Please ensure sample_pdf/famas_dispute/ directory contains test files."
        )

    # Extract text using Docling
    from core.docling_adapter import DoclingAdapter
    from core.config import DoclingConfig

    print(f"📄 Loading test document: {test_file.name}")
    docling_config = DoclingConfig()
    doc_extractor = DoclingAdapter(docling_config)

    with open(test_file, 'rb') as f:
        extracted_doc = doc_extractor.extract_text(f.read(), {"file_path": str(test_file)})

    print(f"✅ Extracted {len(extracted_doc.plain_text)} characters")

    return extracted_doc.plain_text, extracted_doc.metadata


def run_extraction(
    provider: str,
    model: str,
    text: str,
    metadata: Dict[str, Any]
) -> BenchmarkResult:
    """
    Run extraction with specified provider and model

    Args:
        provider: Provider name (anthropic, openai, langextract)
        model: Model identifier
        text: Document text
        metadata: Document metadata

    Returns:
        BenchmarkResult with extraction results
    """
    print(f"\n🔄 Running extraction: {provider} / {model}")

    # Load config with runtime model override
    docling_config, event_config, extractor_config = load_provider_config(
        provider=provider,
        runtime_model=model
    )

    # Create extractors
    doc_extractor, event_extractor = create_extractors(
        docling_config,
        event_config,
        extractor_config
    )

    # Check if extractor is available
    if not event_extractor.is_available():
        raise RuntimeError(
            f"❌ {provider.title()} event extractor not available. "
            f"Check API key configuration in .env"
        )

    # Run extraction with timing
    start_time = time.perf_counter()
    events = event_extractor.extract_events(text, metadata)
    extraction_time = time.perf_counter() - start_time

    # Analyze results
    events_with_dates = sum(1 for e in events if e.date and e.date != "Date not available")
    events_with_citations = sum(1 for e in events if e.citation and e.citation != "No citation available")

    # Estimate cost (rough approximation)
    char_count = len(text)
    estimated_tokens = char_count / 4  # Rough estimate: 4 chars per token

    # Provider-specific cost estimation
    if provider == "anthropic":
        if "sonnet-4-5" in model:
            estimated_cost = (estimated_tokens / 1_000_000) * 3.0  # $3/M input
        elif "opus-4" in model:
            estimated_cost = (estimated_tokens / 1_000_000) * 15.0  # $15/M input
        elif "haiku" in model:
            estimated_cost = (estimated_tokens / 1_000_000) * 0.25  # $0.25/M input
        else:
            estimated_cost = (estimated_tokens / 1_000_000) * 3.0  # Default to Sonnet pricing
    elif provider == "openai":
        if "gpt-5" in model:
            estimated_cost = (estimated_tokens / 1_000_000) * 5.0  # $5/M estimated
        elif "gpt-4o-mini" in model:
            estimated_cost = (estimated_tokens / 1_000_000) * 0.15  # $0.15/M input
        else:
            estimated_cost = (estimated_tokens / 1_000_000) * 2.5  # Default to GPT-4o pricing
    elif provider == "langextract":
        estimated_cost = 0.01  # Gemini is free/very cheap
    else:
        estimated_cost = 0.0

    print(f"✅ Extracted {len(events)} events in {extraction_time:.2f}s (est. ${estimated_cost:.4f})")

    return BenchmarkResult(
        model=model,
        provider=provider,
        event_count=len(events),
        events_with_dates=events_with_dates,
        events_with_citations=events_with_citations,
        extraction_time=extraction_time,
        estimated_cost=estimated_cost,
        events=[{
            "number": e.number,
            "date": e.date,
            "event_particulars": e.event_particulars,
            "citation": e.citation,
            "document_reference": e.document_reference
        } for e in events]
    )


def compute_jaccard_similarity(text1: str, text2: str) -> float:
    """
    Compute Jaccard similarity between two texts

    Args:
        text1: First text
        text2: Second text

    Returns:
        Jaccard similarity (0.0 to 1.0)
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    intersection = words1 & words2
    union = words1 | words2

    if not union:
        return 0.0

    return len(intersection) / len(union)


def compare_results(ground_truth: BenchmarkResult, production: BenchmarkResult) -> ComparisonMetrics:
    """
    Compare ground truth and production results

    Args:
        ground_truth: Ground truth extraction results
        production: Production model extraction results

    Returns:
        ComparisonMetrics with comparison analysis
    """
    print(f"\n📊 Comparing {ground_truth.model} (ground truth) vs {production.model} (production)")

    # Event recall
    event_recall = production.event_count / ground_truth.event_count if ground_truth.event_count > 0 else 0.0

    # Date accuracy (simple matching - could be more sophisticated)
    ground_truth_dates = {e["date"] for e in ground_truth.events if e["date"] != "Date not available"}
    production_dates = {e["date"] for e in production.events if e["date"] != "Date not available"}
    date_match_count = len(ground_truth_dates & production_dates)
    date_accuracy = date_match_count / len(ground_truth_dates) if ground_truth_dates else 0.0

    # Citation recall
    ground_truth_citations = sum(1 for e in ground_truth.events if e["citation"] != "No citation available")
    production_citations = sum(1 for e in production.events if e["citation"] != "No citation available")
    citation_match_count = production_citations  # Simplified: count if present
    citation_recall = production_citations / ground_truth_citations if ground_truth_citations > 0 else 0.0

    # Jaccard similarity (average across all event pairs)
    jaccard_similarities = []
    for gt_event in ground_truth.events:
        best_similarity = 0.0
        for prod_event in production.events:
            similarity = compute_jaccard_similarity(
                gt_event["event_particulars"],
                prod_event["event_particulars"]
            )
            best_similarity = max(best_similarity, similarity)
        jaccard_similarities.append(best_similarity)

    avg_jaccard = sum(jaccard_similarities) / len(jaccard_similarities) if jaccard_similarities else 0.0

    # Pass/Fail determination
    if event_recall >= 0.90 and date_accuracy >= 0.95 and citation_recall >= 0.80:
        pass_fail = "PASS"
    elif event_recall >= 0.80 and date_accuracy >= 0.95 and citation_recall >= 0.70:
        pass_fail = "CONDITIONAL_PASS"
    else:
        pass_fail = "FAIL"

    return ComparisonMetrics(
        ground_truth_model=ground_truth.model,
        production_model=production.model,
        ground_truth_events=ground_truth.event_count,
        production_events=production.event_count,
        event_recall=event_recall,
        date_match_count=date_match_count,
        date_accuracy=date_accuracy,
        citation_match_count=citation_match_count,
        citation_recall=citation_recall,
        avg_jaccard_similarity=avg_jaccard,
        pass_fail=pass_fail
    )


def print_comparison_report(metrics: ComparisonMetrics, ground_truth: BenchmarkResult, production: BenchmarkResult):
    """Print formatted comparison report"""
    print("\n" + "=" * 80)
    print("GROUND TRUTH BENCHMARK REPORT")
    print("=" * 80)

    print(f"\n📊 **Model Comparison**")
    print(f"   Ground Truth: {metrics.ground_truth_model}")
    print(f"   Production:   {metrics.production_model}")

    print(f"\n📈 **Event Extraction**")
    print(f"   Ground Truth Events: {metrics.ground_truth_events}")
    print(f"   Production Events:   {metrics.production_events}")
    print(f"   Event Recall:        {metrics.event_recall:.1%} {'✅' if metrics.event_recall >= 0.90 else '⚠️' if metrics.event_recall >= 0.80 else '❌'} (target: ≥90%)")

    print(f"\n📅 **Date Extraction**")
    print(f"   Date Matches:        {metrics.date_match_count}")
    print(f"   Date Accuracy:       {metrics.date_accuracy:.1%} {'✅' if metrics.date_accuracy >= 0.95 else '❌'} (target: ≥95%)")

    print(f"\n📖 **Citation Extraction**")
    print(f"   Citation Matches:    {metrics.citation_match_count}")
    print(f"   Citation Recall:     {metrics.citation_recall:.1%} {'✅' if metrics.citation_recall >= 0.80 else '⚠️' if metrics.citation_recall >= 0.70 else '❌'} (target: ≥80%)")

    print(f"\n📝 **Text Similarity**")
    print(f"   Avg Jaccard:         {metrics.avg_jaccard_similarity:.1%} {'✅' if metrics.avg_jaccard_similarity >= 0.60 else '⚠️'} (target: ≥60%)")

    print(f"\n💰 **Cost Analysis**")
    print(f"   Ground Truth Cost:   ${ground_truth.estimated_cost:.4f}")
    print(f"   Production Cost:     ${production.estimated_cost:.4f}")
    cost_ratio = ground_truth.estimated_cost / production.estimated_cost if production.estimated_cost > 0 else 0.0
    print(f"   Cost Reduction:      {cost_ratio:.0f}x cheaper 💸")

    print(f"\n⏱️  **Performance**")
    print(f"   Ground Truth Time:   {ground_truth.extraction_time:.2f}s")
    print(f"   Production Time:     {production.extraction_time:.2f}s")

    print(f"\n🏆 **Final Result**")
    if metrics.pass_fail == "PASS":
        print(f"   Status: ✅ **PASS** - Production model meets quality targets")
        print(f"   Recommendation: Deploy production model ({production.model})")
    elif metrics.pass_fail == "CONDITIONAL_PASS":
        print(f"   Status: ⚠️  **CONDITIONAL PASS** - Acceptable with caveats")
        print(f"   Recommendation: Review missed events/citations, consider deploying with monitoring")
    else:
        print(f"   Status: ❌ **FAIL** - Production model below quality targets")
        print(f"   Recommendation: Try different production model or improve prompt")

    print("\n" + "=" * 80)


def benchmark_provider(provider: str) -> Dict[str, Any]:
    """
    Run benchmark for a specific provider

    Args:
        provider: Provider name (anthropic, openai, langextract)

    Returns:
        Benchmark results dictionary
    """
    # Define ground truth and production models for each provider
    model_pairs = {
        "anthropic": {
            "ground_truth": "claude-sonnet-4-5",
            "production": "claude-3-haiku-20240307"
        },
        "openai": {
            "ground_truth": "gpt-5",
            "production": "gpt-4o-mini"
        },
        "langextract": {
            "ground_truth": "gemini-2.5-pro",
            "production": "gemini-2.0-flash"
        }
    }

    if provider not in model_pairs:
        raise ValueError(f"Unknown provider: {provider}. Supported: {list(model_pairs.keys())}")

    print(f"\n🚀 Starting benchmark for {provider.upper()}")

    # Load test document
    text, metadata = load_test_document()

    # Run ground truth extraction
    ground_truth_result = run_extraction(
        provider=provider,
        model=model_pairs[provider]["ground_truth"],
        text=text,
        metadata=metadata
    )

    # Run production extraction
    production_result = run_extraction(
        provider=provider,
        model=model_pairs[provider]["production"],
        text=text,
        metadata=metadata
    )

    # Compare results
    metrics = compare_results(ground_truth_result, production_result)

    # Print report
    print_comparison_report(metrics, ground_truth_result, production_result)

    # Return structured results
    return {
        "provider": provider,
        "timestamp": datetime.now().isoformat(),
        "test_document": str(metadata.get("file_path", "Unknown")),
        "ground_truth": asdict(ground_truth_result),
        "production": asdict(production_result),
        "metrics": asdict(metrics)
    }


def main():
    """Main benchmark execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Ground Truth Model Benchmark")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run benchmarks for all providers"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/benchmarks",
        help="Output directory for benchmark results (default: output/benchmarks)"
    )

    args = parser.parse_args()

    # Determine which provider to test
    test_provider = os.getenv("TEST_PROVIDER", "anthropic").lower()

    if args.all:
        providers = ["anthropic", "openai", "langextract"]
        print("🔬 Running benchmarks for all providers")
    else:
        providers = [test_provider]
        print(f"🔬 Running benchmark for {test_provider}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run benchmarks
    results = []
    for provider in providers:
        try:
            result = benchmark_provider(provider)
            results.append(result)

            # Save individual result
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"ground_truth_benchmark_{provider}_{timestamp}.json"

            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)

            print(f"\n💾 Saved results to: {output_file}")

        except Exception as e:
            print(f"\n❌ Benchmark failed for {provider}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    if len(results) > 1:
        print("\n" + "=" * 80)
        print("SUMMARY - ALL PROVIDERS")
        print("=" * 80)

        for result in results:
            metrics = result["metrics"]
            print(f"\n{result['provider'].upper()}: {metrics['pass_fail']}")
            print(f"   Event Recall:     {metrics['event_recall']:.1%}")
            print(f"   Date Accuracy:    {metrics['date_accuracy']:.1%}")
            print(f"   Citation Recall:  {metrics['citation_recall']:.1%}")
            print(f"   Cost Reduction:   {result['ground_truth']['estimated_cost'] / result['production']['estimated_cost']:.0f}x")


if __name__ == "__main__":
    main()
