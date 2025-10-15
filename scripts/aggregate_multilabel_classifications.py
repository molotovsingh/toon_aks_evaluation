#!/usr/bin/env python3
"""
Aggregate multi-label classification results and generate analysis report.

This script analyzes multi-label classification results where models can return
multiple classes per document. Key metrics:
- Primary Accuracy: Does "primary" field match GPT-5 single-label ground truth?
- Any-Label Recall: Is GPT-5 label anywhere in "classes" array?
- Label Count: Average labels per document (hedging behavior)
- Multi-Label Frequency: Which documents consistently get multiple labels?
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# Multi-label classification directory
OUTPUT_DIR = Path("output/classification_multilabel")

# Single-label ground truth directory
GROUND_TRUTH_DIR = Path("output/classification")

# Model names for cleaner display
MODEL_DISPLAY_NAMES = {
    "anthropic_claude-3-haiku": "Claude 3 Haiku",
    "openai_gpt-4o-mini": "GPT-4o-mini",
    "openai_gpt-oss-120b": "GPT-OSS-120B",
    "meta-llama_llama-3.3-70b-instruct": "Llama 3.3 70B",
    "mistralai_mistral-large-2411": "Mistral Large 2411",
    "gpt-5": "GPT-5 (Ground Truth)",
}


def extract_document_name(filename: str) -> str:
    """Extract document base name from classification JSON filename."""
    # Remove model suffix
    for model_key in MODEL_DISPLAY_NAMES.keys():
        if filename.endswith(f"__{model_key}.json"):
            return filename.replace(f"__{model_key}.json", "")
    return filename.replace(".json", "")


def load_classification(file_path: Path) -> Dict:
    """Load classification JSON and extract key fields."""
    with open(file_path, "r") as f:
        data = json.load(f)

    response = data.get("response", {})
    return {
        "primary": response.get("primary", "UNKNOWN"),
        "classes": response.get("classes", []),
        "confidence": response.get("confidence", 0.0),
        "rationale": response.get("rationale", ""),
        "document": data.get("document", ""),
        "model": data.get("model", ""),
    }


def load_ground_truth() -> Dict[str, str]:
    """Load GPT-5 single-label ground truth from output/classification/."""
    ground_truth = {}

    for json_file in GROUND_TRUTH_DIR.glob("*__gpt-5.json"):
        doc_name = extract_document_name(json_file.name)
        classification = load_classification(json_file)
        ground_truth[doc_name] = classification["primary"]

    return ground_truth


def aggregate_results() -> Tuple[Dict, List[str], List[str]]:
    """
    Aggregate all multi-label classification results.

    Returns:
        - results: Dict[document_name][model_name] = classification_dict
        - all_documents: List of unique document names
        - all_models: List of model names in order
    """
    results = defaultdict(dict)
    all_models_set = set()

    for json_file in sorted(OUTPUT_DIR.glob("*.json")):
        filename = json_file.name
        doc_name = extract_document_name(filename)

        # Determine model
        model_key = None
        for key in MODEL_DISPLAY_NAMES.keys():
            if f"__{key}.json" in filename:
                model_key = MODEL_DISPLAY_NAMES[key]
                all_models_set.add(model_key)
                break

        if model_key:
            classification = load_classification(json_file)
            results[doc_name][model_key] = classification

    all_documents = sorted(results.keys())
    all_models = ["Claude 3 Haiku", "GPT-4o-mini", "GPT-OSS-120B", "Llama 3.3 70B", "Mistral Large 2411"]  # Production models only

    return results, all_documents, all_models


def calculate_primary_accuracy(results: Dict, documents: List[str], models: List[str], ground_truth: Dict) -> Dict:
    """Calculate primary field accuracy against GPT-5 ground truth."""
    accuracy_stats = {}

    for model in models:
        exact_matches = 0
        total_compared = 0

        for doc in documents:
            # Only compare if both model has classification and ground truth exists
            if model in results[doc] and doc in ground_truth:
                gt_class = ground_truth[doc]
                model_class = results[doc][model].get("primary", "N/A")

                if gt_class != "N/A" and model_class != "N/A":
                    total_compared += 1
                    if gt_class == model_class:
                        exact_matches += 1

        if total_compared > 0:
            accuracy_stats[model] = {
                "exact_matches": exact_matches,
                "total": total_compared,
                "accuracy": exact_matches / total_compared,
            }

    return accuracy_stats


def calculate_anylabel_recall(results: Dict, documents: List[str], models: List[str], ground_truth: Dict) -> Dict:
    """Calculate any-label recall: Is GPT-5 label anywhere in classes array?"""
    recall_stats = {}

    for model in models:
        hits = 0
        total_compared = 0

        for doc in documents:
            if model in results[doc] and doc in ground_truth:
                gt_class = ground_truth[doc]
                model_classes = results[doc][model].get("classes", [])

                if gt_class != "N/A" and len(model_classes) > 0:
                    total_compared += 1
                    if gt_class in model_classes:
                        hits += 1

        if total_compared > 0:
            recall_stats[model] = {
                "hits": hits,
                "total": total_compared,
                "recall": hits / total_compared,
            }

    return recall_stats


def calculate_label_count_stats(results: Dict, documents: List[str], models: List[str]) -> Dict:
    """Calculate average label count per document per model."""
    label_stats = {}

    for model in models:
        label_counts = []
        single_label_docs = 0
        multi_label_docs = 0

        for doc in documents:
            if model in results[doc]:
                classes = results[doc][model].get("classes", [])
                count = len(classes)
                label_counts.append(count)

                if count == 1:
                    single_label_docs += 1
                elif count > 1:
                    multi_label_docs += 1

        if label_counts:
            label_stats[model] = {
                "avg_labels": sum(label_counts) / len(label_counts),
                "min_labels": min(label_counts),
                "max_labels": max(label_counts),
                "single_label_docs": single_label_docs,
                "multi_label_docs": multi_label_docs,
                "multi_label_rate": multi_label_docs / len(label_counts) if len(label_counts) > 0 else 0,
            }

    return label_stats


def identify_multilabel_documents(results: Dict, documents: List[str], models: List[str]) -> List[Dict]:
    """Identify documents that consistently receive multiple labels across models."""
    multilabel_docs = []

    for doc in documents:
        model_label_counts = {}
        for model in models:
            if model in results[doc]:
                classes = results[doc][model].get("classes", [])
                model_label_counts[model] = len(classes)

        # Document is "consistently multi-label" if 3+ models give it 2+ labels
        multi_count = sum(1 for count in model_label_counts.values() if count >= 2)
        if multi_count >= 3:
            multilabel_docs.append({
                "document": doc,
                "models_with_multi": multi_count,
                "label_counts": model_label_counts,
            })

    return multilabel_docs


def generate_markdown_report(
    results: Dict,
    documents: List[str],
    models: List[str],
    ground_truth: Dict,
    primary_accuracy: Dict,
    anylabel_recall: Dict,
    label_counts: Dict,
    multilabel_docs: List[Dict],
) -> str:
    """Generate comprehensive multi-label analysis report."""
    report = []

    # Header
    report.append("# Multi-Label Classification Analysis Report")
    report.append("")
    report.append("**Date**: 2025-10-13")
    report.append("**Analysis**: Multi-label vs single-label classification comparison")
    report.append("**Models Tested**: 5 production models with multi-label prompt")
    report.append("")
    report.append("---")
    report.append("")

    # Executive Summary
    report.append("## Executive Summary")
    report.append("")
    report.append(f"Tested **multi-label classification** on {len(documents)} documents across 5 models:")
    report.append("")
    for model in models:
        if model in label_counts:
            avg = label_counts[model]["avg_labels"]
            multi_rate = label_counts[model]["multi_label_rate"] * 100
            report.append(f"- **{model}**: Avg {avg:.2f} labels/doc ({multi_rate:.1f}% multi-label)")
    report.append("")
    report.append(f"**Key Finding**: {len(multilabel_docs)} documents consistently receive multiple labels (3+ models agree)")
    report.append("")
    report.append("---")
    report.append("")

    # Accuracy Comparison
    report.append("## Accuracy vs Single-Label Ground Truth")
    report.append("")
    report.append("Comparing against GPT-5 single-label classifications:")
    report.append("")
    report.append("| Model | Primary Accuracy | Any-Label Recall | Gain |")
    report.append("|-------|------------------|------------------|------|")

    for model in models:
        if model in primary_accuracy and model in anylabel_recall:
            primary = primary_accuracy[model]["accuracy"] * 100
            recall = anylabel_recall[model]["recall"] * 100
            gain = recall - primary
            report.append(f"| **{model}** | {primary:.1f}% | {recall:.1f}% | +{gain:.1f}% |")

    report.append("")
    report.append("**Interpretation**:")
    report.append("- **Primary Accuracy**: Does the `primary` field match GPT-5's single-label classification?")
    report.append("- **Any-Label Recall**: Is GPT-5's label anywhere in the `classes` array?")
    report.append("- **Gain**: How much better is any-label recall vs primary-only accuracy?")
    report.append("")
    report.append("---")
    report.append("")

    # Label Count Analysis
    report.append("## Label Count Analysis")
    report.append("")
    report.append("Average labels per document (hedging behavior):")
    report.append("")
    report.append("| Model | Avg Labels | Min | Max | Single-Label Docs | Multi-Label Docs | Multi-Label Rate |")
    report.append("|-------|------------|-----|-----|-------------------|------------------|------------------|")

    for model in models:
        if model in label_counts:
            stats = label_counts[model]
            rate = stats["multi_label_rate"] * 100
            report.append(
                f"| **{model}** | {stats['avg_labels']:.2f} | {stats['min_labels']} | {stats['max_labels']} | "
                f"{stats['single_label_docs']} | {stats['multi_label_docs']} | {rate:.1f}% |"
            )

    report.append("")
    report.append("---")
    report.append("")

    # Multi-Label Documents
    report.append("## Consistently Multi-Label Documents")
    report.append("")
    report.append(f"Documents where 3+ models provided multiple labels (total: {len(multilabel_docs)}):")
    report.append("")

    if multilabel_docs:
        for item in multilabel_docs:
            doc = item["document"]
            models_with_multi = item["models_with_multi"]
            counts = item["label_counts"]

            report.append(f"**{doc}** ({models_with_multi}/5 models multi-label):")
            for model, count in counts.items():
                if count >= 2:
                    report.append(f"- {model}: {count} labels")
            report.append("")
    else:
        report.append("No documents consistently received multiple labels.")
        report.append("")

    report.append("---")
    report.append("")

    # Key Insights
    report.append("## Key Insights")
    report.append("")
    report.append("### 1. Multi-Label vs Single-Label Performance")

    # Find model with biggest gain
    max_gain_model = None
    max_gain = 0
    for model in models:
        if model in primary_accuracy and model in anylabel_recall:
            gain = (anylabel_recall[model]["recall"] - primary_accuracy[model]["accuracy"]) * 100
            if gain > max_gain:
                max_gain = gain
                max_gain_model = model

    if max_gain_model:
        report.append(f"- **{max_gain_model}** showed the biggest gain (+{max_gain:.1f}%) in any-label recall vs primary accuracy")
        report.append(f"- This suggests the model understands document complexity but may prioritize differently than GPT-5")

    report.append("")
    report.append("### 2. Label Count Behavior")

    # Find model with highest multi-label rate
    max_multi_model = None
    max_multi_rate = 0
    for model in models:
        if model in label_counts:
            rate = label_counts[model]["multi_label_rate"]
            if rate > max_multi_rate:
                max_multi_rate = rate
                max_multi_model = model

    if max_multi_model:
        report.append(f"- **{max_multi_model}** uses multi-labels most frequently ({max_multi_rate * 100:.1f}% of documents)")
        report.append(f"- This could indicate either comprehensive understanding or hedging behavior")

    report.append("")
    report.append("### 3. Document Complexity Signals")
    if len(multilabel_docs) > 0:
        report.append(f"- {len(multilabel_docs)} documents are genuinely multi-purpose (multiple models agree)")
        report.append(f"- These documents likely serve multiple legal functions simultaneously")
    else:
        report.append("- Most documents have a clear single primary purpose")

    report.append("")
    report.append("---")
    report.append("")

    # Recommendations
    report.append("## Recommendations")
    report.append("")
    report.append("### Production Use Cases")
    report.append("")
    report.append("**Use Multi-Label When:**")
    report.append("- Document serves multiple functions (e.g., email containing contract)")
    report.append("- Need to capture all aspects for comprehensive tagging")
    report.append("- Building search/filter systems that benefit from multiple tags")
    report.append("")
    report.append("**Use Single-Label When:**")
    report.append("- Need one clear primary classification")
    report.append("- Building routing/triage systems")
    report.append("- Simplicity is preferred over comprehensiveness")
    report.append("")
    report.append("### Model Selection for Multi-Label")

    # Recommend best model based on any-label recall
    best_recall_model = None
    best_recall = 0
    for model in models:
        if model in anylabel_recall:
            recall = anylabel_recall[model]["recall"]
            if recall > best_recall:
                best_recall = recall
                best_recall_model = model

    if best_recall_model:
        report.append(f"1. **Best Any-Label Recall**: {best_recall_model} ({best_recall * 100:.1f}%)")

    # Recommend model with best balance
    report.append("2. **Balanced Approach**: Use primary accuracy as main metric, any-label recall as validation")
    report.append("")
    report.append("---")
    report.append("")

    # Reproducibility
    report.append("## Reproducibility")
    report.append("")
    report.append("### Commands Used")
    report.append("```bash")
    report.append("# Multi-label classification with Claude 3 Haiku")
    report.append("uv run python scripts/classify_documents.py --execute --multi-label \\")
    report.append("  --model anthropic/claude-3-haiku --glob 'sample_pdf/**/*' \\")
    report.append("  --max-chars 1600 --temperature 0.0")
    report.append("")
    report.append("# Aggregate multi-label results")
    report.append("uv run python scripts/aggregate_multilabel_classifications.py")
    report.append("```")
    report.append("")
    report.append("### Data Directories")
    report.append(f"- **Multi-label results**: `{OUTPUT_DIR}/` ({len(list(OUTPUT_DIR.glob('*.json')))} JSON files)")
    report.append(f"- **Ground truth**: `{GROUND_TRUTH_DIR}/` (GPT-5 single-label classifications)")
    report.append("")
    report.append("---")
    report.append("")
    report.append("*End of Report*")

    return "\n".join(report)


def main():
    print("Aggregating multi-label classification results...")

    # Load ground truth
    ground_truth = load_ground_truth()
    print(f"Loaded {len(ground_truth)} ground truth classifications from GPT-5")

    # Load multi-label results
    results, documents, models = aggregate_results()
    print(f"Found {len(documents)} documents")
    print(f"Models: {', '.join(models)}")

    # Calculate metrics
    primary_accuracy = calculate_primary_accuracy(results, documents, models, ground_truth)
    anylabel_recall = calculate_anylabel_recall(results, documents, models, ground_truth)
    label_counts = calculate_label_count_stats(results, documents, models)
    multilabel_docs = identify_multilabel_documents(results, documents, models)

    # Generate report
    report = generate_markdown_report(
        results,
        documents,
        models,
        ground_truth,
        primary_accuracy,
        anylabel_recall,
        label_counts,
        multilabel_docs,
    )

    output_path = Path("docs/reports/classification-multilabel-analysis.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(report)

    print(f"\n✅ Multi-label analysis report generated: {output_path}")

    # Print quick summary
    print("\n📊 Quick Summary:")
    for model in models:
        if model in primary_accuracy:
            primary = primary_accuracy[model]["accuracy"] * 100
            recall = anylabel_recall[model]["recall"] * 100 if model in anylabel_recall else 0
            gain = recall - primary
            print(f"  {model}: {primary:.1f}% primary → {recall:.1f}% any-label (+{gain:.1f}%)")


if __name__ == "__main__":
    main()
