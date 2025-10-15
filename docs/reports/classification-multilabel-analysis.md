# Multi-Label Classification Analysis Report

**Date**: 2025-10-13
**Analysis**: Multi-label vs single-label classification comparison
**Models Tested**: 5 production models with multi-label prompt

---

## Executive Summary

Tested **multi-label classification** on 20 documents across 5 models:

- **Claude 3 Haiku**: Avg 1.90 labels/doc (65.0% multi-label)
- **GPT-4o-mini**: Avg 1.50 labels/doc (35.0% multi-label)
- **GPT-OSS-120B**: Avg 1.82 labels/doc (36.4% multi-label)
- **Llama 3.3 70B**: Avg 1.35 labels/doc (35.0% multi-label)
- **Mistral Large 2411**: Avg 2.05 labels/doc (70.0% multi-label)

**Key Finding**: 8 documents consistently receive multiple labels (3+ models agree)

---

## Accuracy vs Single-Label Ground Truth

Comparing against GPT-5 single-label classifications:

| Model | Primary Accuracy | Any-Label Recall | Gain |
|-------|------------------|------------------|------|
| **Claude 3 Haiku** | 50.0% | 80.0% | +30.0% |
| **GPT-4o-mini** | 60.0% | 70.0% | +10.0% |
| **GPT-OSS-120B** | 90.9% | 100.0% | +9.1% |
| **Llama 3.3 70B** | 75.0% | 85.0% | +10.0% |
| **Mistral Large 2411** | 65.0% | 75.0% | +10.0% |

**Interpretation**:
- **Primary Accuracy**: Does the `primary` field match GPT-5's single-label classification?
- **Any-Label Recall**: Is GPT-5's label anywhere in the `classes` array?
- **Gain**: How much better is any-label recall vs primary-only accuracy?

---

## Label Count Analysis

Average labels per document (hedging behavior):

| Model | Avg Labels | Min | Max | Single-Label Docs | Multi-Label Docs | Multi-Label Rate |
|-------|------------|-----|-----|-------------------|------------------|------------------|
| **Claude 3 Haiku** | 1.90 | 1 | 4 | 7 | 13 | 65.0% |
| **GPT-4o-mini** | 1.50 | 1 | 4 | 13 | 7 | 35.0% |
| **GPT-OSS-120B** | 1.82 | 1 | 5 | 7 | 4 | 36.4% |
| **Llama 3.3 70B** | 1.35 | 1 | 2 | 13 | 7 | 35.0% |
| **Mistral Large 2411** | 2.05 | 1 | 5 | 6 | 14 | 70.0% |

---

## Consistently Multi-Label Documents

Documents where 3+ models provided multiple labels (total: 8):

**Amrapali_Allotment_Letter** (3/5 models multi-label):
- Claude 3 Haiku: 2 labels
- Llama 3.3 70B: 2 labels
- Mistral Large 2411: 2 labels

**Amrapali_No_Objection** (4/5 models multi-label):
- Claude 3 Haiku: 2 labels
- GPT-4o-mini: 2 labels
- Llama 3.3 70B: 2 labels
- Mistral Large 2411: 2 labels

**Amrapali_Reciepts__1st_Buyer** (4/5 models multi-label):
- Claude 3 Haiku: 2 labels
- GPT-4o-mini: 2 labels
- Llama 3.3 70B: 2 labels
- Mistral Large 2411: 2 labels

**Answer_to_request_for_Arbitration-_Case_Reference__DIS-IHK-2025-01180-_Famas_GmbH_vs_Elcomponics_Sales_Pvt_Ltd** (5/5 models multi-label):
- Claude 3 Haiku: 2 labels
- GPT-4o-mini: 3 labels
- GPT-OSS-120B: 3 labels
- Llama 3.3 70B: 2 labels
- Mistral Large 2411: 2 labels

**FAMAS_CASE_NARRATIVE_SUMMARY** (4/5 models multi-label):
- Claude 3 Haiku: 3 labels
- GPT-4o-mini: 2 labels
- GPT-OSS-120B: 2 labels
- Mistral Large 2411: 2 labels

**FaMAS_GmbH_Vs_Elcomponics_Sales_Pvt._Ltd,_O_s_Amount_Euro_245,000,_File_Ref_#_29260CFIN_2024** (5/5 models multi-label):
- Claude 3 Haiku: 3 labels
- GPT-4o-mini: 2 labels
- GPT-OSS-120B: 3 labels
- Llama 3.3 70B: 2 labels
- Mistral Large 2411: 3 labels

**ambiguous_dates_document** (4/5 models multi-label):
- Claude 3 Haiku: 3 labels
- GPT-4o-mini: 2 labels
- Llama 3.3 70B: 2 labels
- Mistral Large 2411: 2 labels

**mixed_date_formats_document** (5/5 models multi-label):
- Claude 3 Haiku: 4 labels
- GPT-4o-mini: 4 labels
- GPT-OSS-120B: 5 labels
- Llama 3.3 70B: 2 labels
- Mistral Large 2411: 3 labels

---

## Key Insights

### 1. Multi-Label vs Single-Label Performance
- **Claude 3 Haiku** showed the biggest gain (+30.0%) in any-label recall vs primary accuracy
- This suggests the model understands document complexity but may prioritize differently than GPT-5

### 2. Label Count Behavior
- **Mistral Large 2411** uses multi-labels most frequently (70.0% of documents)
- This could indicate either comprehensive understanding or hedging behavior

### 3. Document Complexity Signals
- 8 documents are genuinely multi-purpose (multiple models agree)
- These documents likely serve multiple legal functions simultaneously

---

## Recommendations

### Production Use Cases

**Use Multi-Label When:**
- Document serves multiple functions (e.g., email containing contract)
- Need to capture all aspects for comprehensive tagging
- Building search/filter systems that benefit from multiple tags

**Use Single-Label When:**
- Need one clear primary classification
- Building routing/triage systems
- Simplicity is preferred over comprehensiveness

### Model Selection for Multi-Label
1. **Best Any-Label Recall**: GPT-OSS-120B (100.0%)
2. **Balanced Approach**: Use primary accuracy as main metric, any-label recall as validation

---

## Reproducibility

### Commands Used
```bash
# Multi-label classification with Claude 3 Haiku
uv run python scripts/classify_documents.py --execute --multi-label \
  --model anthropic/claude-3-haiku --glob 'sample_pdf/**/*' \
  --max-chars 1600 --temperature 0.0

# Aggregate multi-label results
uv run python scripts/aggregate_multilabel_classifications.py
```

### Data Directories
- **Multi-label results**: `output/classification_multilabel/` (91 JSON files)
- **Ground truth**: `output/classification/` (GPT-5 single-label classifications)

---

*End of Report*