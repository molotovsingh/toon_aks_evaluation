#!/usr/bin/env python3
"""
Compare V1 vs V2 multi-label classification results.

Generates comprehensive report analyzing:
- Multi-label rate changes (hedging reduction)
- Primary accuracy maintenance/improvement
- Any-label recall maintenance
- Cross-model consistency improvements
- Document-level behavior changes
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def load_ground_truth(gt_dir: Path) -> Dict[str, str]:
    """Load GPT-5 single-label ground truth classifications."""
    ground_truth = {}
    for json_file in gt_dir.glob("*__openai_gpt-5.json"):
        data = json.loads(json_file.read_text())
        doc_name = Path(data["document"]).stem
        ground_truth[doc_name] = data["response"]["primary"]
    return ground_truth


def load_results(results_dir: Path, prompt_version: str) -> Dict[str, Dict[str, dict]]:
    """Load classification results by model and document."""
    results = defaultdict(dict)
    for json_file in results_dir.glob("*.json"):
        data = json.loads(json_file.read_text())

        # Only process files matching the prompt version
        if data["prompt_version"] != prompt_version:
            continue

        doc_name = Path(data["document"]).stem
        model = data["model"]
        results[model][doc_name] = data["response"]
    return results


def calculate_metrics(
    results: Dict[str, dict], ground_truth: Dict[str, str]
) -> Tuple[float, float, float, float, int, int]:
    """Calculate primary accuracy, any-label recall, avg labels, multi-label rate."""
    primary_correct = 0
    any_label_correct = 0
    total_labels = 0
    multi_label_count = 0
    total = 0

    for doc_name, response in results.items():
        if doc_name not in ground_truth:
            continue

        total += 1
        gt_label = ground_truth[doc_name]
        classes = response.get("classes", [])
        primary = response.get("primary", "")

        # Primary accuracy
        if primary == gt_label:
            primary_correct += 1

        # Any-label recall
        if gt_label in classes:
            any_label_correct += 1

        # Label count stats
        total_labels += len(classes)
        if len(classes) > 1:
            multi_label_count += 1

    if total == 0:
        return 0.0, 0.0, 0.0, 0.0, 0, 0

    primary_acc = (primary_correct / total) * 100
    any_recall = (any_label_correct / total) * 100
    avg_labels = total_labels / total
    multi_rate = (multi_label_count / total) * 100

    return primary_acc, any_recall, avg_labels, multi_rate, total, multi_label_count


def analyze_document_changes(
    v1_results: Dict[str, dict],
    v2_results: Dict[str, dict],
) -> List[Tuple[str, int, int, List[str], List[str]]]:
    """Identify documents where label count changed between V1 and V2."""
    changes = []

    for doc_name in set(v1_results.keys()) & set(v2_results.keys()):
        v1_classes = v1_results[doc_name].get("classes", [])
        v2_classes = v2_results[doc_name].get("classes", [])

        if len(v1_classes) != len(v2_classes):
            changes.append((doc_name, len(v1_classes), len(v2_classes), v1_classes, v2_classes))

    return sorted(changes, key=lambda x: abs(x[1] - x[2]), reverse=True)


def main():
    # Directories
    gt_dir = Path("output/classification")
    v1_dir = Path("output/classification_multilabel")
    v2_dir = Path("output/classification_multilabel_v2")

    # Load ground truth
    print("Loading GPT-5 ground truth...")
    ground_truth = load_ground_truth(gt_dir)
    print(f"Loaded {len(ground_truth)} ground truth documents")

    # Load V1 and V2 results
    print("\nLoading V1 results...")
    v1_results = load_results(v1_dir, "multilabel-v1")
    print(f"Loaded {len(v1_results)} models")

    print("\nLoading V2 results...")
    v2_results = load_results(v2_dir, "multilabel-v2")
    print(f"Loaded {len(v2_results)} models")

    # Calculate metrics per model
    print("\n" + "=" * 80)
    print("V1 vs V2 COMPARISON")
    print("=" * 80)

    models = sorted(set(v1_results.keys()) & set(v2_results.keys()))

    for model in models:
        print(f"\n### {model}")
        print("-" * 80)

        # V1 metrics
        v1_primary, v1_any, v1_avg_labels, v1_multi_rate, v1_total, v1_multi_count = calculate_metrics(
            v1_results[model], ground_truth
        )

        # V2 metrics
        v2_primary, v2_any, v2_avg_labels, v2_multi_rate, v2_total, v2_multi_count = calculate_metrics(
            v2_results[model], ground_truth
        )

        print(f"Documents: V1={v1_total}, V2={v2_total}")
        print(f"\nPrimary Accuracy: V1={v1_primary:.1f}% → V2={v2_primary:.1f}% ({v2_primary - v1_primary:+.1f}%)")
        print(f"Any-Label Recall: V1={v1_any:.1f}% → V2={v2_any:.1f}% ({v2_any - v1_any:+.1f}%)")
        print(f"Avg Labels/Doc:   V1={v1_avg_labels:.2f} → V2={v2_avg_labels:.2f} ({v2_avg_labels - v1_avg_labels:+.2f})")
        print(f"Multi-Label Rate: V1={v1_multi_rate:.1f}% ({v1_multi_count}/{v1_total}) → V2={v2_multi_rate:.1f}% ({v2_multi_count}/{v2_total}) ({v2_multi_rate - v1_multi_rate:+.1f}%)")

        # Document-level changes
        changes = analyze_document_changes(v1_results[model], v2_results[model])
        if changes:
            print(f"\nDocument Changes (label count): {len(changes)} documents")
            for doc, v1_count, v2_count, v1_classes, v2_classes in changes[:5]:
                print(f"  {doc}: {v1_count} → {v2_count} labels")
                print(f"    V1: {v1_classes}")
                print(f"    V2: {v2_classes}")

    # Cross-model consistency analysis
    print("\n" + "=" * 80)
    print("CROSS-MODEL CONSISTENCY")
    print("=" * 80)

    # Calculate variance in multi-label rates
    v1_rates = []
    v2_rates = []
    for model in models:
        _, _, _, v1_rate, _, _ = calculate_metrics(v1_results[model], ground_truth)
        _, _, _, v2_rate, _, _ = calculate_metrics(v2_results[model], ground_truth)
        v1_rates.append(v1_rate)
        v2_rates.append(v2_rate)

    if v1_rates and v2_rates:
        v1_variance = max(v1_rates) - min(v1_rates)
        v2_variance = max(v2_rates) - min(v2_rates)
        print(f"\nMulti-Label Rate Variance:")
        print(f"  V1: {v1_variance:.1f}% (range: {min(v1_rates):.1f}% - {max(v1_rates):.1f}%)")
        print(f"  V2: {v2_variance:.1f}% (range: {min(v2_rates):.1f}% - {max(v2_rates):.1f}%)")
        print(f"  Improvement: {v1_variance - v2_variance:+.1f}% (lower is better)")


if __name__ == "__main__":
    main()
