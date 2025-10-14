# Classification Prompt Registry Guide

**Purpose**: Manage multiple classification prompt strategies with documented characteristics, authorship, and versioning metadata.

**Date**: 2025-10-14
**Status**: Production-Ready

---

## Overview

The **Classification Prompt Registry** provides a centralized system for managing document classification prompts with full metadata tracking. The registry enables:

- **Multiple prompt strategies** with documented trade-offs (recall vs precision, comprehensive vs decisive)
- **Authorship tracking** for accountability and collaboration
- **Version control** with ISO-formatted dates and changelogs
- **Model recommendations** based on empirical testing
- **Use-case guidance** for selecting the right prompt

## Quick Start

### List Available Prompts

```bash
# View all prompt variants with descriptions
uv run python scripts/classify_documents.py --list-prompts
```

**Output**:
```
================================================================================
Available Classification Prompt Variants
================================================================================

📋 comprehensive: Comprehensive Multi-Label (V1)
   Captures all possible document functions. Use when you need high recall ...
   Use cases: search, tagging, comprehensive_analysis
   Labels/doc: 1.5-2.0, Multi-label rate: 35-70%

📋 decisive: Decisive Single-Label Default (V2)
   Commits to single primary label by default. Use when you need decisive ...
   Use cases: routing, triage, primary_classification
   Labels/doc: 1.0-1.5, Multi-label rate: 0-33%

Default variant: comprehensive

To see detailed info: --show-prompt-info <variant_name>
================================================================================
```

### View Detailed Prompt Information

```bash
# Show complete metadata for a specific prompt
uv run python scripts/classify_documents.py --show-prompt-info comprehensive
```

**Output**:
```
================================================================================
Prompt Variant: Comprehensive Multi-Label (V1)
================================================================================

Description: Captures all possible document functions. Use when you need high recall
for search/tagging systems. Models are encouraged to include all relevant labels,
not just the primary one. Best for workflows where finding ALL documents with a
particular function is more important than decisiveness.

Use Cases:
  - search
  - tagging
  - comprehensive_analysis
  - document_discovery
  - multi-tag_recommendation

Characteristics:
  Recall Priority: high
  Precision Priority: moderate
  Decisiveness: low
  Typical Labels/Doc: 1.5-2.0
  Multi-Label Rate: 35-70%

Recommended Models:
  - anthropic/claude-3-haiku
  - mistralai/mistral-large-2411
  - openai/gpt-oss-120b

Metadata:
  Version: multilabel-v1
  Author: User Preference (Primary Research)
  Created: 2025-10-14
  Modified: 2025-10-14

Changelog:
  Initial baseline prompt for comprehensive multi-label classification.
  Optimized for high recall in search/tagging workflows. Tested on 5 models
  with 20 documents. Avg 1.5-2.0 labels/doc, 35-70% multi-label rate.

Output Directory: output/classification_multilabel
================================================================================
```

### Use a Specific Prompt Variant

```bash
# Comprehensive prompt (default for --multi-label)
uv run python scripts/classify_documents.py \
  --execute \
  --multi-label \
  --prompt-variant comprehensive \
  --model anthropic/claude-3-haiku \
  --files sample_pdf/famas_dispute/Transaction_Fee_Invoice.pdf

# Decisive prompt (optimized for routing/triage)
uv run python scripts/classify_documents.py \
  --execute \
  --multi-label \
  --prompt-variant decisive \
  --model openai/gpt-4o-mini \
  --files sample_pdf/famas_dispute/Transaction_Fee_Invoice.pdf

# Backward compatibility: v1/v2 flags still work
uv run python scripts/classify_documents.py \
  --execute \
  --multi-label \
  --prompt-variant v1 \  # Maps to "comprehensive"
  --model meta-llama/llama-3.3-70b-instruct \
  --glob 'sample_pdf/**/*'
```

---

## Architecture

### Registry Structure

The registry is defined in `src/core/prompt_registry.py` using dataclasses for type-safe configuration:

```python
@dataclass
class PromptCharacteristics:
    """Documented characteristics of a classification prompt strategy."""
    recall_priority: str      # "high", "moderate", "low"
    precision_priority: str   # "high", "moderate", "low"
    decisiveness: str         # "high", "moderate", "low", "maximum"
    typical_labels_per_doc: str  # e.g., "1.5-2.0", "1.0-1.5"
    multi_label_rate: str     # e.g., "35-70%", "0-33%"

@dataclass
class PromptVariant:
    """A classification prompt variant with metadata."""
    name: str                    # Human-readable name
    prompt_text: str             # Actual prompt content
    use_cases: List[str]         # Recommended use cases
    characteristics: PromptCharacteristics
    recommended_models: List[str]
    description: str
    version: str                 # e.g., "multilabel-v1", "multilabel-v2"
    output_directory: str        # Where to save results

    # Authorship and versioning metadata
    author: str                  # Author or team responsible
    created_date: str            # ISO format: "YYYY-MM-DD"
    modified_date: str           # ISO format: "YYYY-MM-DD"
    changelog: Optional[str]     # Brief description of changes
```

### Registry Dictionary

All prompts are stored in a central registry:

```python
CLASSIFICATION_PROMPT_REGISTRY: Dict[str, PromptVariant] = {
    "comprehensive": PromptVariant(...),
    "decisive": PromptVariant(...),
    # Add custom variants here
}
```

---

## Available Prompt Variants

### Comprehensive Multi-Label (V1)

**Registry Key**: `comprehensive`

**When to Use**:
- Search and discovery workflows
- Document tagging systems
- Comprehensive document analysis
- High recall requirements (find ALL relevant documents)

**Characteristics**:
- **Recall Priority**: High (captures all possible functions)
- **Precision Priority**: Moderate (may include secondary functions)
- **Decisiveness**: Low (encourages multi-labeling)
- **Typical Labels/Doc**: 1.5-2.0
- **Multi-Label Rate**: 35-70%

**Recommended Models**:
- `anthropic/claude-3-haiku` - 1.90 labels/doc, 80% recall
- `mistralai/mistral-large-2411` - 2.05 labels/doc, most comprehensive
- `openai/gpt-oss-120b` - 1.82 labels/doc, 100% recall

**Metadata**:
- **Author**: User Preference (Primary Research)
- **Version**: multilabel-v1
- **Created**: 2025-10-14
- **Modified**: 2025-10-14
- **Output Directory**: `output/classification_multilabel`

**Changelog**:
> Initial baseline prompt for comprehensive multi-label classification.
> Optimized for high recall in search/tagging workflows. Tested on 5 models
> with 20 documents. Avg 1.5-2.0 labels/doc, 35-70% multi-label rate.

---

### Decisive Single-Label Default (V2)

**Registry Key**: `decisive`

**When to Use**:
- Document routing and triage
- Workflow automation (e.g., "which folder does this go in?")
- Primary classification decisions
- Document type detection

**Characteristics**:
- **Recall Priority**: Moderate
- **Precision Priority**: High (commits to primary label)
- **Decisiveness**: High (default to single label)
- **Typical Labels/Doc**: 1.0-1.5
- **Multi-Label Rate**: 0-33%

**Recommended Models**:
- `openai/gpt-oss-120b` - 1.00 labels/doc, 0% multi-label
- `meta-llama/llama-3.3-70b-instruct` - 1.05 labels/doc, 5% multi-label
- `openai/gpt-4o-mini` - 1.05 labels/doc, 5% multi-label

**Metadata**:
- **Author**: Optimization Research Team
- **Version**: multilabel-v2
- **Created**: 2025-10-14
- **Modified**: 2025-10-14
- **Output Directory**: `output/classification_multilabel_v2`

**Changelog**:
> Optimized V2 prompt with explicit 'DEFAULT TO SINGLE-LABEL' guidance.
> Reduces hedging by 37.6% average across 5 models while maintaining quality.
> Improved cross-model consistency (variance reduced by 1.7%). Best for routing/triage workflows.

---

## Workflow: Choosing the Right Prompt

### Decision Tree

```
START: What is your primary goal?
│
├─ "I need to FIND documents by searching for specific functions"
│  └─ Use: comprehensive (high recall)
│     └─ Models: claude-3-haiku, mistral-large-2411, gpt-oss-120b
│
├─ "I need to ROUTE documents to the correct folder/workflow"
│  └─ Use: decisive (high precision)
│     └─ Models: gpt-oss-120b, llama-3.3-70b, gpt-4o-mini
│
├─ "I need a document to have ALL possible tags"
│  └─ Use: comprehensive (multi-label encouragement)
│
└─ "I need ONE clear decision per document"
   └─ Use: decisive (single-label default)
```

### Example Use Cases

**Case 1: Legal Research Platform (Search)**
```bash
# Goal: Users search "show me all agreements"
# Requirement: High recall - don't miss any relevant documents
# Solution: Comprehensive prompt with high-recall model

uv run python scripts/classify_documents.py \
  --execute \
  --multi-label \
  --prompt-variant comprehensive \
  --model anthropic/claude-3-haiku \
  --glob 'legal_docs/**/*.pdf'
```

**Case 2: Document Management System (Routing)**
```bash
# Goal: Automatically route documents to correct departments
# Requirement: One clear decision per document
# Solution: Decisive prompt with high-precision model

uv run python scripts/classify_documents.py \
  --execute \
  --multi-label \
  --prompt-variant decisive \
  --model openai/gpt-oss-120b \
  --glob 'incoming_docs/**/*.pdf'
```

**Case 3: Document Analysis Dashboard (Tagging)**
```bash
# Goal: Tag documents with all applicable categories for analytics
# Requirement: Comprehensive tagging for filtering and grouping
# Solution: Comprehensive prompt with multi-label model

uv run python scripts/classify_documents.py \
  --execute \
  --multi-label \
  --prompt-variant comprehensive \
  --model mistralai/mistral-large-2411 \
  --glob 'contract_archive/**/*.pdf'
```

---

## Adding Custom Prompt Variants

### Step 1: Define Your Prompt

Create a new prompt variant in `src/core/prompt_registry.py`:

```python
CLASSIFICATION_PROMPT_REGISTRY["my_custom"] = PromptVariant(
    name="Custom Domain-Specific Prompt",
    prompt_text=MY_CUSTOM_PROMPT,  # Define in src/core/constants.py
    use_cases=[
        "domain_specific_task",
        "specialized_workflow",
    ],
    characteristics=PromptCharacteristics(
        recall_priority="moderate",
        precision_priority="high",
        decisiveness="moderate",
        typical_labels_per_doc="1.2-1.5",
        multi_label_rate="20-40%",
    ),
    recommended_models=[
        "model/variant-1",
        "model/variant-2",
    ],
    description=(
        "Detailed description of when and why to use this prompt. "
        "Explain the trade-offs and ideal use cases."
    ),
    version="custom-v1",
    output_directory="output/classification_custom",
    author="Your Name or Team",
    created_date="2025-10-14",
    modified_date="2025-10-14",
    changelog="Initial custom prompt for [specific use case]. "
             "Tested on [N] models with [M] documents. "
             "Optimized for [specific requirement].",
)
```

### Step 2: Define Prompt Text

Add your prompt to `src/core/constants.py`:

```python
MY_CUSTOM_PROMPT = """
You are a specialized legal document classifier for [domain].

Your task: Review the document and classify it according to [specific schema].

[Include detailed instructions, examples, and output format]

Return valid JSON only:
{
  "classes": ["class1", "class2"],
  "primary": "class1",
  "confidence": 0.85,
  "rationale": "Brief explanation"
}
"""
```

### Step 3: Test Your Prompt

```bash
# Test with a small batch
uv run python scripts/classify_documents.py \
  --execute \
  --multi-label \
  --prompt-variant my_custom \
  --model anthropic/claude-3-haiku \
  --files sample_pdf/test_doc.pdf

# Run comprehensive testing
uv run python scripts/classify_documents.py \
  --execute \
  --multi-label \
  --prompt-variant my_custom \
  --model anthropic/claude-3-haiku \
  --glob 'test_corpus/**/*.pdf'

# Compare against baseline
uv run python scripts/aggregate_multilabel_v2_classifications.py
```

### Step 4: Document Results

Update the `changelog` field with empirical findings:

```python
changelog="Custom prompt optimized for [use case]. "
         "Tested on 5 models with 50 documents. "
         "Avg 1.2 labels/doc (vs 1.5 for comprehensive). "
         "Multi-label rate 25% (vs 50% for comprehensive). "
         "Primary accuracy 85% (vs 75% for comprehensive)."
```

---

## Programmatic Usage

### Python API

```python
from src.core.prompt_registry import (
    get_prompt_variant,
    get_prompt_text,
    get_output_directory,
    get_prompt_version,
    list_prompt_variants,
)

# List all available variants
variants = list_prompt_variants()
# ['comprehensive', 'decisive']

# Get full variant object with metadata
variant = get_prompt_variant("comprehensive")
print(variant.name)  # "Comprehensive Multi-Label (V1)"
print(variant.author)  # "User Preference (Primary Research)"
print(variant.created_date)  # "2025-10-14"

# Get just the prompt text
prompt = get_prompt_text("decisive")

# Get output directory for a variant
output_dir = get_output_directory("comprehensive")
# "output/classification_multilabel"

# Get version identifier
version = get_prompt_version("decisive")
# "multilabel-v2"

# Backward compatibility helper
from src.core.prompt_registry import get_prompt_for_v1_v2_flag

variant_name = get_prompt_for_v1_v2_flag("v1")  # Returns "comprehensive"
variant_name = get_prompt_for_v1_v2_flag("v2")  # Returns "decisive"
variant_name = get_prompt_for_v1_v2_flag("comprehensive")  # Returns "comprehensive"
```

### Integration with classify_documents.py

The script automatically routes prompt variants:

```python
# In scripts/classify_documents.py

# Get registry variant name (maps v1/v2 to comprehensive/decisive)
registry_variant = get_prompt_for_v1_v2_flag(args.prompt_variant)

# Get prompt text and metadata
prompt_text = get_prompt_text(registry_variant)
prompt_version = get_prompt_version(registry_variant)
output_dir = get_output_directory(registry_variant)

# Use in classification
system_message, user_message = build_multilabel_prompt(
    doc_title,
    doc_excerpt,
    variant=registry_variant
)
```

---

## Versioning Best Practices

### ISO Date Format

Use ISO 8601 format for all dates:

```python
created_date="2025-10-14"  # ✅ Correct
modified_date="2025-10-14"  # ✅ Correct

created_date="Oct 14, 2025"  # ❌ Incorrect
modified_date="10/14/2025"  # ❌ Incorrect
```

### Changelog Guidelines

Write concise, data-driven changelogs:

**Good Example**:
```python
changelog="Optimized V2 prompt with explicit 'DEFAULT TO SINGLE-LABEL' guidance. "
         "Reduces hedging by 37.6% average across 5 models while maintaining quality. "
         "Improved cross-model consistency (variance reduced by 1.7%). Best for routing/triage workflows."
```

**Bad Example**:
```python
changelog="Made some improvements to the prompt."  # ❌ Too vague
changelog="This is the best prompt ever created and will solve all problems."  # ❌ No data
```

### Version Naming Convention

Follow semantic versioning for prompt versions:

- **Major version**: Fundamental prompt strategy change (e.g., `multilabel-v1` → `multilabel-v2`)
- **Minor version**: Refinements or optimizations (e.g., `multilabel-v2.1`)
- **Patch version**: Typo fixes or clarifications (e.g., `multilabel-v2.1.1`)

Example:
```python
version="multilabel-v2"      # Major version 2
version="multilabel-v2.1"    # Minor update to v2
version="multilabel-v2.1.1"  # Patch to v2.1
```

---

## Testing and Validation

### Regression Testing

When adding or modifying prompts, run comprehensive tests:

```bash
# 1. Test new prompt with all recommended models
for model in anthropic/claude-3-haiku \
             openai/gpt-4o-mini \
             meta-llama/llama-3.3-70b-instruct; do
  uv run python scripts/classify_documents.py \
    --execute \
    --multi-label \
    --prompt-variant my_custom \
    --model $model \
    --glob 'sample_pdf/**/*' \
    --max-chars 1600 \
    --temperature 0.0
done

# 2. Generate comparison report
uv run python scripts/aggregate_multilabel_v2_classifications.py

# 3. Review output directories
ls -lh output/classification_custom/
```

### Metrics to Track

Monitor these key metrics when testing prompts:

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Avg Labels/Doc** | 1.0-2.0 | Sum of all labels ÷ documents |
| **Multi-Label Rate** | 0-70% | Documents with >1 label ÷ total |
| **Primary Accuracy** | >75% | Correct primary label ÷ total (requires ground truth) |
| **Any-Label Recall** | >80% | Ground truth in classes list ÷ total |
| **Cross-Model Variance** | <20% | Std dev of labels/doc across models |

### Validation Checklist

- [ ] Prompt produces valid JSON output (no parsing errors)
- [ ] All required fields present (`classes`, `primary`, `confidence`, `rationale`)
- [ ] Labels/doc within expected range (1.0-2.5)
- [ ] Multi-label rate within expected range (0-70%)
- [ ] Output directory created and files saved correctly
- [ ] Metadata fields populated (author, dates, changelog)
- [ ] Recommended models tested and verified
- [ ] Comparison report generated and reviewed
- [ ] Documentation updated with findings

---

## Troubleshooting

### Common Issues

**Issue**: "Unknown prompt variant: 'xyz'"
```bash
# Solution: Check available variants
uv run python scripts/classify_documents.py --list-prompts
```

**Issue**: Empty output directory
```bash
# Check: Did you forget --execute flag?
uv run python scripts/classify_documents.py \
  --execute \  # ← Required for real API calls
  --multi-label \
  --prompt-variant comprehensive \
  --model anthropic/claude-3-haiku \
  --files sample.pdf
```

**Issue**: Results in wrong directory
```bash
# Check: Prompt variant controls output directory
# comprehensive → output/classification_multilabel/
# decisive → output/classification_multilabel_v2/

# Override if needed
uv run python scripts/classify_documents.py \
  --execute \
  --multi-label \
  --prompt-variant comprehensive \
  --output-dir output/my_custom_dir/ \
  --model anthropic/claude-3-haiku \
  --files sample.pdf
```

---

## References

### Key Files

- **Registry Definition**: `src/core/prompt_registry.py:60-136`
- **Prompt Text**: `src/core/constants.py:LEGAL_CLASSIFICATION_MULTILABEL_PROMPT`
- **Classification Script**: `scripts/classify_documents.py:147-163`
- **Aggregation Script**: `scripts/aggregate_multilabel_v2_classifications.py`

### Related Documentation

- **V1 vs V2 Comparison Report**: `docs/reports/classification-multilabel-prompt-optimization.md`
- **Comprehensive Analysis**: `docs/reports/classification-comprehensive-analysis.md`
- **Multi-Label Analysis**: `docs/reports/classification-multilabel-analysis.md`
- **Ground Truth Workflow**: `docs/guides/ground-truth-workflow.md`

### Command-Line Reference

```bash
# List all prompt variants
--list-prompts

# Show detailed info for a variant
--show-prompt-info <variant_name>

# Use a specific variant
--prompt-variant <variant_name>

# Backward compatibility
--prompt-variant v1  # Maps to "comprehensive"
--prompt-variant v2  # Maps to "decisive"
```

---

*Last Updated: 2025-10-14*
*Maintainer: Classification Prompt Registry Team*
