# Ground Truth Workflow Guide

**Version**: 1.0
**Date**: 2025-10-11
**Status**: Active

## Table of Contents

1. [Overview](#overview)
2. [What is Ground Truth?](#what-is-ground-truth)
3. [Why Use Ground Truth Models?](#why-use-ground-truth-models)
4. [Available Ground Truth Models](#available-ground-truth-models)
5. [Step-by-Step Workflow](#step-by-step-workflow)
6. [Best Practices](#best-practices)
7. [Cost Optimization Strategies](#cost-optimization-strategies)
8. [Comparison Methodology](#comparison-methodology)
9. [Quality Metrics](#quality-metrics)
10. [Troubleshooting](#troubleshooting)

---

## Overview

This guide explains how to use **premium ground truth models** to create reference extraction datasets for validating production model quality. Ground truth datasets serve as quality benchmarks, enabling you to test cheaper production models against known-good extractions.

**Key Benefit**: Validate that a $0.001/doc production model matches the quality of a $0.05/doc premium model, reducing costs by 50x while maintaining quality.

---

## What is Ground Truth?

**Ground truth** refers to reference data created using the highest-quality models available. In legal event extraction:

- **Ground Truth Dataset**: Legal events extracted using premium models (Claude Sonnet 4.5, GPT-5, Gemini 2.5 Pro)
- **Production Dataset**: Legal events extracted using budget models (Claude Haiku, GPT-4o-mini, Gemini 2.0 Flash)
- **Validation**: Comparing production outputs against ground truth to measure quality

**Analogy**: Ground truth is like having an expert lawyer review documents to create a "gold standard" extraction, then testing if a paralegal can achieve similar results.

---

## Why Use Ground Truth Models?

### 1. **Cost Optimization**
- Test if cheaper models match premium quality
- Reduce ongoing extraction costs by 5-50x
- Make data-driven decisions about model selection

### 2. **Quality Assurance**
- Establish quality baselines for production models
- Catch extraction errors before they affect downstream processes
- Validate prompt changes don't degrade quality

### 3. **Model Evaluation**
- Compare new models against established benchmarks
- A/B test different providers on same documents
- Quantify quality differences objectively

### 4. **Regulatory Compliance**
- Demonstrate due diligence in AI system quality
- Provide audit trail for critical legal document processing
- Meet accuracy requirements for legal workflows

---

## Available Ground Truth Models

### Tier 1: Recommended (Best Balance)

**Claude Sonnet 4.5** (`claude-sonnet-4-5`)
- **Provider**: Anthropic Direct API
- **Pricing**: $3/M input • $15/M output • 200K context
- **Cost per 15-page doc**: ~$0.02-0.05
- **Best for**: Most ground truth creation tasks
- **Strengths**: "Best coding model in the world" (Sep 2025), optimal quality/cost balance
- **Use case**: Default choice for creating reference datasets

### Tier 2: Specialized Use Cases

**GPT-5** (`gpt-5`)
- **Provider**: OpenAI Direct API
- **Pricing**: TBD (estimated $5-10/M) • 128K context
- **Cost per 15-page doc**: ~$0.03-0.08 (estimated)
- **Best for**: Alternative validation, coding-focused extraction
- **Strengths**: Latest OpenAI flagship, strong reasoning capabilities
- **Use case**: Second opinion for Tier 1 validation, OpenAI-specific testing

**Gemini 2.5 Pro** (`gemini-2.5-pro`)
- **Provider**: Google LangExtract/Gemini
- **Pricing**: TBD • 2M context window
- **Cost per doc**: ~$0.04-0.10 (estimated for long docs)
- **Best for**: Long documents (50+ pages), contracts with extensive appendices
- **Strengths**: 2M context window (vs 200K for others)
- **Use case**: Documents that exceed 200K token limit

### Tier 3: Validation Only

**Claude Opus 4** (`claude-opus-4`)
- **Provider**: Anthropic Direct API
- **Pricing**: $15/M input • $75/M output • 200K context
- **Cost per 15-page doc**: ~$0.08-0.20
- **Best for**: Quality validation of Tier 1 outputs
- **Strengths**: Highest quality model, best for complex reasoning
- **Use case**: Spot-check Tier 1 extractions on critical documents only

---

## Step-by-Step Workflow

### Phase 1: Create Ground Truth Dataset

**Step 1: Select Documents**
```bash
# Recommended sample size: 10-30 documents
# Balance: Small enough to be affordable, large enough to be representative

# Good sample selection:
- Mix of document types (contracts, filings, correspondence)
- Range of complexities (simple to complex)
- Representative of production workload
```

**Step 2: Configure Ground Truth Model**
```bash
# 1. Start Streamlit app
uv run streamlit run app.py

# 2. In UI:
#    - Select "Anthropic" from provider dropdown
#    - Select "Claude Sonnet 4.5" from model dropdown
#    - Upload sample documents (10-30 files)
```

**Step 3: Process and Export**
```bash
# 3. Process documents (wait for extraction to complete)
# 4. Review extraction quality in UI table
# 5. Export as CSV: Click "Download as CSV" button
# 6. Save file as: ground_truth_sonnet45_2025-10-11.csv
```

**Cost Estimate**:
- 20 documents × $0.03/doc = $0.60 total
- One-time cost for creating reference dataset

### Phase 2: Test Production Model

**Step 4: Configure Production Model**
```bash
# 1. Same Streamlit app session
# 2. In UI:
#    - Keep "Anthropic" selected
#    - Select "Claude 3 Haiku" from model dropdown
#    - Upload SAME documents as Phase 1
```

**Step 5: Process and Export**
```bash
# 3. Process documents (should be faster than ground truth)
# 4. Export as CSV: Click "Download as CSV" button
# 5. Save file as: production_haiku_2025-10-11.csv
```

**Cost Estimate**:
- 20 documents × $0.0005/doc = $0.01 total
- 60x cheaper than ground truth!

### Phase 3: Compare Results

**Step 6: Manual Comparison (Quick Validation)**
```bash
# Open both CSV files side-by-side
# Compare event counts, dates, citations, particulars

# Quick checks:
- Did production model find all ground truth events?
- Are dates consistent?
- Are citations present where ground truth has them?
- Are event descriptions sufficiently detailed?
```

**Step 7: Automated Comparison (Comprehensive)**
```bash
# Use comparison script (if available)
uv run python scripts/compare_extractions.py \
  ground_truth_sonnet45_2025-10-11.csv \
  production_haiku_2025-10-11.csv \
  --output comparison_report_2025-10-11.json

# Metrics computed:
# - Event count agreement
# - Date extraction accuracy
# - Citation recall
# - Text similarity (cosine/Jaccard)
```

### Phase 4: Decide and Deploy

**Step 8: Evaluate Results**
```bash
# Decision criteria:
- Production model >= 90% event recall? → PASS
- Production model >= 95% date accuracy? → PASS
- Production model >= 80% citation recall? → PASS
- Event descriptions sufficiently detailed? → PASS

# If all PASS → Deploy production model
# If any FAIL → Try different production model or improve prompt
```

**Step 9: Deploy Production Model**
```bash
# Update .env with validated production model
echo "ANTHROPIC_MODEL=claude-3-haiku-20240307" >> .env

# Deploy to production pipeline
# Monitor first 100 documents for quality
# Re-validate monthly or after prompt changes
```

---

## Best Practices

### 1. **Document Selection**
- ✅ **DO**: Select representative sample covering typical use cases
- ✅ **DO**: Include edge cases (long docs, scanned PDFs, unusual formats)
- ✅ **DO**: Use real production documents (anonymized if needed)
- ❌ **DON'T**: Use only simple documents (doesn't test model limits)
- ❌ **DON'T**: Cherry-pick documents that favor one model

### 2. **Sample Size**
- **Minimum**: 10 documents (quick validation)
- **Recommended**: 20-30 documents (reliable benchmark)
- **Comprehensive**: 50-100 documents (statistical significance)
- **Overkill**: 200+ documents (diminishing returns, high cost)

### 3. **Ground Truth Model Selection**
- **Default**: Claude Sonnet 4.5 (Tier 1) for most use cases
- **Long docs**: Gemini 2.5 Pro (Tier 2) for 50+ page contracts
- **Validation**: Claude Opus 4 (Tier 3) for spot-checking critical docs

### 4. **Version Control**
- Save ground truth datasets with version identifiers
- Include model name and date in filename
- Track prompt version used (V1 vs V2)
- Document any preprocessing or configuration changes

### 5. **Revalidation Triggers**
- **Prompt changes**: Always revalidate after modifying `LEGAL_EVENTS_PROMPT`
- **Model updates**: Revalidate when provider updates models
- **Monthly**: Periodic revalidation for production monitoring
- **Quality issues**: Immediate revalidation if production issues detected

---

## Cost Optimization Strategies

### Strategy 1: Tiered Validation

```bash
# Tier 1 (Always): Claude Sonnet 4.5 ground truth
Ground truth (20 docs): $0.60

# Tier 2 (If needed): Claude Opus 4 spot-check
Validation (5 critical docs): $0.50

Total: $1.10 one-time cost for comprehensive ground truth
```

### Strategy 2: Incremental Validation

```bash
# Start small, expand if needed
Phase 1 (10 docs): $0.30 → Validate production model
Phase 2 (20 more docs): $0.60 → If Phase 1 shows issues
Phase 3 (50 more docs): $1.50 → If Phase 2 needs more data

# Only proceed to next phase if validation fails
# Expected cost: $0.30-0.90 (most cases pass in Phase 1-2)
```

### Strategy 3: Focused Validation

```bash
# Target specific document types or edge cases
Contract validation (10 contracts): $0.30
Filing validation (10 filings): $0.30
Correspondence validation (10 emails): $0.30

Total: $0.90 for type-specific ground truth
# More efficient than validating 30 mixed documents
```

### Strategy 4: Amortized Ground Truth

```bash
# Create ground truth once, test multiple production models

Initial ground truth (30 docs, Sonnet 4.5): $0.90

Test model 1 (Haiku): $0.02
Test model 2 (GPT-4o-mini): $0.05
Test model 3 (Gemini 2.0 Flash): $0.03
Test model 4 (DeepSeek): $0.01

Total: $1.01 to test 4 production models
# Ground truth cost shared across all model tests
```

---

## Comparison Methodology

### Manual Comparison Checklist

**Event Count Comparison**
- [ ] Count events in ground truth extraction
- [ ] Count events in production extraction
- [ ] Calculate recall: `production_events / ground_truth_events × 100%`
- [ ] **Target**: ≥90% recall (acceptable to miss some minor events)

**Date Extraction Accuracy**
- [ ] For each ground truth event with date, check if production has same date
- [ ] Count matches and mismatches
- [ ] Calculate accuracy: `matches / total_ground_truth_dates × 100%`
- [ ] **Target**: ≥95% date accuracy

**Citation Recall**
- [ ] For each ground truth event with citation, check if production has citation
- [ ] Count citation presence (don't require exact match)
- [ ] Calculate recall: `production_citations / ground_truth_citations × 100%`
- [ ] **Target**: ≥80% citation recall

**Event Particulars Quality**
- [ ] Compare description completeness (2-8 sentences)
- [ ] Check if key details present (parties, context, implications)
- [ ] Assess if sufficient for legal analysis
- [ ] **Target**: Subjective "sufficient detail" judgment

### Automated Comparison Metrics

**Jaccard Similarity** (Set Overlap)
```python
# For each event, compute word overlap
jaccard = len(ground_truth_words ∩ production_words) / len(ground_truth_words ∪ production_words)

# Target: ≥0.60 (60% word overlap indicates similar content)
```

**Cosine Similarity** (Semantic Similarity)
```python
# Use sentence embeddings to compute semantic similarity
cosine_sim = dot(ground_truth_embedding, production_embedding) / (||ground_truth|| × ||production||)

# Target: ≥0.75 (75% semantic similarity indicates aligned meaning)
```

**ROUGE-L** (Longest Common Subsequence)
```python
# Measure longest matching subsequence of words
rouge_l = LCS(ground_truth, production) / len(ground_truth)

# Target: ≥0.50 (50% LCS indicates significant content overlap)
```

---

## Quality Metrics

### Primary Metrics (Always Measure)

| Metric | Formula | Target | Criticality |
|--------|---------|--------|-------------|
| **Event Recall** | `production_events / ground_truth_events` | ≥90% | HIGH |
| **Date Accuracy** | `correct_dates / total_ground_truth_dates` | ≥95% | HIGH |
| **Citation Recall** | `production_citations / ground_truth_citations` | ≥80% | MEDIUM |

### Secondary Metrics (If Available)

| Metric | Formula | Target | Criticality |
|--------|---------|--------|-------------|
| **Jaccard Similarity** | `|A ∩ B| / |A ∪ B|` | ≥0.60 | MEDIUM |
| **Cosine Similarity** | `dot(A, B) / (||A|| × ||B||)` | ≥0.75 | MEDIUM |
| **Description Length** | `avg_words_per_event` | 50-300 words | LOW |

### Pass/Fail Criteria

**PASS** (Production model acceptable):
- Event recall ≥90% **AND**
- Date accuracy ≥95% **AND**
- Citation recall ≥80%

**CONDITIONAL PASS** (Acceptable with caveats):
- Event recall 80-90% (may miss minor events)
- Citation recall 70-80% (some citation loss acceptable)
- **Requires**: Manual review of missed events/citations

**FAIL** (Production model not acceptable):
- Event recall <80% (missing too many events)
- Date accuracy <95% (date errors are critical)
- Description quality insufficient

---

## Troubleshooting

### Issue: Production Model Misses Events

**Symptoms**:
- Event recall <80%
- Production extraction has significantly fewer events than ground truth

**Diagnosis**:
```bash
# Check which events were missed
1. Review ground truth events not in production
2. Look for patterns:
   - Are missed events all minor procedural updates?
   - Are missed events all in specific document sections?
   - Are missed events all without dates?
```

**Solutions**:
1. **Prompt adjustment**: If missing specific event types, update prompt to emphasize those
2. **Model upgrade**: Try next-tier production model (e.g., Haiku → Sonnet 3.5)
3. **Acceptable loss**: If missed events are minor, document as acceptable trade-off

### Issue: Date Extraction Errors

**Symptoms**:
- Date accuracy <95%
- Production dates don't match ground truth dates

**Diagnosis**:
```bash
# Identify error patterns
1. Wrong date format? (e.g., "Jan 5" vs "January 5, 2024")
2. Date hallucination? (dates not in document)
3. Missing dates? (ground truth has date, production doesn't)
```

**Solutions**:
1. **Format normalization**: Accept equivalent date formats as matches
2. **Prompt clarification**: Emphasize "only extract explicitly stated dates"
3. **Model upgrade**: Date extraction critical, may require better model

### Issue: Citation Recall Low

**Symptoms**:
- Citation recall <70%
- Production often has "No citation available" where ground truth has citations

**Diagnosis**:
```bash
# Check citation patterns
1. Are citations in footnotes or inline?
2. Are citations complex (e.g., "29 U.S.C. § 794(a)")?
3. Are citations mentioned far from event description?
```

**Solutions**:
1. **Prompt adjustment**: Add examples of complex citation formats
2. **Accept trade-off**: Low citation recall may be acceptable if event recall is high
3. **Post-processing**: Consider citation extraction as separate pipeline step

### Issue: High Cost

**Symptoms**:
- Ground truth creation cost exceeds budget
- Need to validate many documents

**Solutions**:
1. **Reduce sample size**: Start with 10 docs instead of 30
2. **Use Tier 1 only**: Skip Tier 3 validation (Claude Opus 4)
3. **Amortize cost**: Test multiple production models against same ground truth
4. **Incremental validation**: Only expand sample if initial validation fails

---

## Conclusion

Ground truth model selection enables **data-driven quality assurance** for legal event extraction pipelines. By investing $0.30-1.00 in creating reference datasets, you can:

- Validate production models achieve ≥90% quality
- Reduce ongoing extraction costs by 5-50x
- Make confident decisions about model selection
- Establish auditable quality benchmarks

**Next Steps**:
1. Select 10-20 representative documents
2. Create ground truth with Claude Sonnet 4.5 (~$0.30-0.60)
3. Test production model on same documents
4. Compare results using manual checklist
5. Deploy production model if metrics pass

**Questions?** See main project documentation or contact repository maintainers.

---

**Document History**:
- 2025-10-11: Initial version (v1.0)
