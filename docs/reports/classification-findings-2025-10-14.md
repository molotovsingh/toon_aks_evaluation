# Classification Findings Report

Date: 2025-10-14

## Executive Summary
- Llama 3.3 70B ranks best for correctness across the current corpus, matching proprietary models on real documents and handling synthetic edge cases cleanly.
- Claude 3 Haiku and GPT‑4o‑mini are dependable fallbacks; Claude leads on confidence, GPT‑4o‑mini on cost/perf balance.
- GPT‑OSS‑120B is viable as an Apache 2.0 privacy hedge but fails on synthetic/edge cases; pair with a fallback.
- Multi‑label V1 prompts caused over‑hedging; V2 “default to single” reduced extra labels 30–60% across models without hurting recall.

## Sources Reviewed
- docs/reports/classification-small-models.md
- docs/reports/classification-multilabel-analysis.md
- docs/reports/classification-multilabel-prompt-optimization.md

## Model Ranking (Correctness)
- Primary: Llama 3.3 70B — most consistent accuracy; robust to synthetic docs.
- Secondary: Claude 3 Haiku — high confidence; strong agreement on real docs.
- Tertiary: GPT‑4o‑mini — solid correctness; best budget fallback.
- Hedge: GPT‑OSS‑120B — use when Apache 2.0 is mandatory; enforce fallback for synthetic/short/OCR‑sparse inputs.

## Multi‑Label Findings
- V1 prompt (multi‑label encouraged) improved any‑label recall but over‑tagged (35–70% multi‑label rate by model). Some documents are genuinely multi‑purpose (NOC, receipts, narrative summaries).
- Mistral Large 2411 showed the highest multi‑label rate (2.05 labels/doc), suggesting broader coverage or hedging.
- Recommendation: expose multi‑label as a toggle for discovery/search; keep single‑label as default for routing/triage.

## Prompt Optimization
- V2 prompt (“default single label; allow multi when necessary”) cut multi‑label rates dramatically without hurting recall:
  - Haiku: 65.0% → 33.3%
  - Llama 3.3 70B: 35.0% → 5.0%
  - Mistral Large: 70.0% → 10.0%
  - GPT‑4o‑mini: 35.0% → 5.0%
  - GPT‑OSS‑120B: 36.4% → 0.0%
- Cross‑model behavior is more consistent; accuracy/recall remained stable or improved. Approved for production default.

## Common Misclassifications
- Evidence vs Other: receipts, cheques, invoices mislabeled as Other (OCR/sparse text).
- Email attachments vs pleadings: transmittal emails should be Correspondence; some models labeled Pleading.
- Synthetic timelines: should map to Case Summary/Chronology; some drift to Agreement/Contract.

## Recommendations
1. Default model: Llama 3.3 70B with V2 prompt; keep Claude and GPT‑4o‑mini as selectable fallbacks.
2. Keep GPT‑OSS‑120B as an opt‑in privacy hedge with enforced fallback policy.
3. Adopt V2 prompt in CLI/Streamlit; retain V1 behind a flag for audits requiring multi‑tag coverage.
4. Add Mistral (mistral‑small/large) to the official bench and measure against the same fixtures.
5. Implement cost preview post‑Docling (order: cost-estimator-001) so users see token/cost estimates per provider before running.
6. Centralize/run logging in DuckDB (order: duckdb-ingestion-001) for trend analysis and prompt/model audits.
7. Ensure metadata records the true runtime model and correct timings (metadata-runtime-model-accuracy-001) so evals remain trustworthy.

## Next Steps
- Switch classifier default to Llama 3.3 70B + V2 prompt in UI/CLI.
- Schedule a small Mistral bake‑off on the 22‑doc set; append results to classification-small-models.md.
- Enable the multi‑label toggle and document when to use it (search/discovery vs routing).

