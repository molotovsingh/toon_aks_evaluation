# Cost Estimator Reference

**Modules**:
- `src/ui/cost_estimator.py` - Event extraction cost estimation (Layer 2)
- `src/ui/document_page_estimator.py` - Document page count estimation (Layer 1)
- `src/core/document_extractor_catalog.py` - Document extractor pricing catalog (Layer 1)

**Purpose**: Pre-extraction cost estimation for the full legal document processing pipeline, including both document extraction (OCR/vision) and event extraction (LLM) costs.

---

## Overview

The cost estimator provides rough order-of-magnitude cost estimates **before** running expensive LLM extractions. This helps users:

- **Budget Planning**: Estimate costs for large document batches before processing
- **Model Selection**: Compare costs across different providers and models
- **Cost Control**: Identify expensive documents early and adjust processing strategy
- **Layer Analysis**: Understand cost breakdown between document processing and event extraction

**Key Features**:
- **Two-layer cost estimation**: Document extraction (Layer 1) + Event extraction (Layer 2)
- **Pre-extraction page counting**: Estimate document pages without running paid vision APIs
- **Character-based token estimation**: No API calls required for event extraction costs
- **Multi-model cost comparison**: Compare across 20+ models
- **Document extractor catalog**: Pricing for Docling (free), Qwen-VL ($0.00512/page), Gemini ($0.0583/page)
- **Integration with model catalog**: Accurate LLM pricing for event extraction
- **CLI and Streamlit UI support**: Show costs in both interfaces
- **Calibration helpers**: Improve accuracy over time

---

## Two-Layer Cost Architecture

The legal document processing pipeline has **two cost layers**:

### Layer 1: Document Extraction (Pre-processing)
- **Purpose**: Convert PDF/DOCX to plain text
- **Technologies**: Docling (local OCR), Qwen-VL (vision API), Gemini 2.5 (vision API)
- **Pricing model**: Per-page or per-image cost
- **Cost range**: $0 (Docling) to $0.0583/page (Gemini)
- **Example**: 15-page PDF → $0.077 (Qwen-VL) or $0.875 (Gemini)

### Layer 2: Event Extraction (LLM Processing)
- **Purpose**: Extract legal events from text using LLM
- **Technologies**: GPT-4, Claude, Gemini, DeepSeek, etc.
- **Pricing model**: Per-token cost (input + output)
- **Cost range**: $0.03/M (DeepSeek) to $15/M (Claude Opus 4)
- **Example**: 15-page doc (~30K tokens) → $0.0045 (GPT-4o Mini) or $0.45 (Claude Opus 4)

### Total Cost = Layer 1 + Layer 2

For a typical 15-page legal document:
- **Budget option**: Docling ($0) + DeepSeek R1 ($0.0009) = **$0.0009**
- **Production option**: Docling ($0) + GPT-4o Mini ($0.0045) = **$0.0045**
- **Premium option**: Qwen-VL ($0.077) + Claude Sonnet 4.5 ($0.09) = **$0.167**
- **Ultra-premium**: Gemini 2.5 ($0.875) + Claude Opus 4 ($0.45) = **$1.325**

---

## Estimation Methodology

### Layer 1: Page Count Estimation (Document Extraction)

The document page estimator solves the "chicken-and-egg" problem: **how to estimate document extraction costs without actually extracting the document** (which costs money for vision APIs).

#### Strategy Hierarchy (Best to Fallback)

1. **Format-specific metadata read** (High confidence):
   - **PDF**: Use `pypdfium2` to read page count from metadata (no extraction, no API call)
   - **DOCX**: Use `python-docx` to count paragraphs (rough: 3 paragraphs = 1 page)
   - **PPTX**: Use `python-pptx` to count slides (each slide = 1 page)
   - **Images**: Always 1 page (JPG, PNG, TIFF, BMP)

2. **File size heuristic** (Medium confidence):
   - **Scanned PDFs**: ~300KB per page → `page_count = file_size / 300000`
   - **Text PDFs**: ~30KB per page → `page_count = file_size / 30000`
   - **DOCX**: ~20KB per page → `page_count = file_size / 20000`

3. **Default fallback** (Low confidence):
   - **Unknown formats**: Use `DEFAULT_PAGE_ESTIMATE` (default: 15 pages)
   - **Configurable** via environment variable

**Confidence Levels**:
- `high`: Metadata read from PDF/PPTX/images (exact page count)
- `medium`: File size heuristic (±30% accuracy)
- `low`: Default fallback (actual may vary significantly)

**Example**:
```python
from src.ui.document_page_estimator import estimate_document_pages

# PDF with metadata (high confidence)
page_count, confidence = estimate_document_pages("contract.pdf")
# Returns: (23, 'high')

# Unknown format (low confidence)
page_count, confidence = estimate_document_pages("document.xyz")
# Returns: (15, 'low')  # DEFAULT_PAGE_ESTIMATE
```

---

### Layer 2: Token Estimation Heuristic (Event Extraction)

The estimator uses a **4 characters = 1 token** rule of thumb:

```python
CHARS_PER_TOKEN = 4.0

def estimate_tokens(text: str) -> int:
    """Estimate tokens using 4 chars = 1 token heuristic"""
    return max(int(len(text) / CHARS_PER_TOKEN), 1)
```

**Rationale**:
- Based on OpenAI tokenizer analysis for English text
- English text typically ranges from 3.5-4.5 chars/token
- Conservative estimate (4.0) provides reasonable safety margin
- Expected accuracy: ±20% for most legal documents

**Trade-offs**:
- **Pros**: Fast, no API calls, works offline, language-agnostic baseline
- **Cons**: Less accurate than real tokenizers, varies by language/content type
- **Acceptable for**: Order-of-magnitude estimates, cost budgeting, model comparison

### Input/Output Token Split

The estimator assumes a **90% input, 10% output** token distribution:

```python
INPUT_TOKEN_RATIO = 0.9   # 90% of tokens are input (document text)
OUTPUT_TOKEN_RATIO = 0.1  # 10% of tokens are output (extracted events JSON)
```

**Rationale**:
- Legal event extraction is **read-heavy**: Full document text goes in
- Structured output is **compact**: JSON events are much smaller than source
- Empirical observation: Most extractions use 5-15% of total tokens for output
- 90/10 split is conservative (slightly over-estimates output costs)

**Why this matters**:
- Many models charge different rates for input vs output tokens
- Example: GPT-4o Mini charges $0.15/M input, $0.60/M output (4x difference)
- Accurate split improves cost estimates by 10-20%

---

## Usage Guide

### CLI: `classify_documents.py`

Show cost estimates without making API calls:

```bash
# Two-layer estimate with Docling (free document extraction)
uv run python scripts/classify_documents.py \
  --estimate-only \
  --doc-extractor docling \
  --model gpt-4o-mini

# Two-layer estimate with Qwen-VL (budget vision API)
uv run python scripts/classify_documents.py \
  --estimate-only \
  --doc-extractor qwen_vl \
  --model claude-3-haiku-20240307 \
  --max-chars 1600

# Two-layer estimate with Gemini (premium vision API)
uv run python scripts/classify_documents.py \
  --estimate-only \
  --doc-extractor gemini \
  --model gpt-4o \
  --files sample_pdf/famas_dispute/*.pdf

# Compare different document extractors (same event model)
uv run python scripts/classify_documents.py --estimate-only --doc-extractor docling --model gpt-4o-mini
uv run python scripts/classify_documents.py --estimate-only --doc-extractor qwen_vl --model gpt-4o-mini
uv run python scripts/classify_documents.py --estimate-only --doc-extractor gemini --model gpt-4o-mini
```

**Two-Layer Output Format**:
```
================================================================================
TWO-LAYER COST ESTIMATES (Document + Event Extraction)
================================================================================

Estimation method:
  Layer 1 (Doc Extraction): Page count metadata (no paid extraction run)
  Layer 2 (Event Extraction): 4 chars = 1 token (±20% accuracy)
  Token split assumption: 90% input, 10% output

--------------------------------------------------------------------------------
Layer / Metric                           Value
--------------------------------------------------------------------------------
LAYER 1: Document Extraction
  Pages processed                                                            15
  Extractor                                              Qwen3-VL (Budget Vision)
  Cost                                                                   $0.0768

LAYER 2: Event Extraction
  Total tokens                                                           30,000
  Model                                                            GPT-4o Mini
  Cost                                                                   $0.0045

--------------------------------------------------------------------------------
TOTAL ESTIMATED COST                                                     $0.0813
--------------------------------------------------------------------------------

Note: Two-layer estimate (document + event extraction). Actual costs may vary ±20-30%.
================================================================================
```

**Notes**:
- **`--doc-extractor`**: Specifies document extraction method (`docling`, `qwen_vl`, `gemini`)
  - `docling`: Free local OCR (default)
  - `qwen_vl`: Budget vision API ($0.00512/page)
  - `gemini`: Premium vision API ($0.0583/page)
- Cost preview does NOT trigger paid document extractors - uses metadata/heuristics instead
- Layer 1 cost is based on estimated page count (from PDF metadata or file size)
- Layer 2 cost uses character-based token estimation (no API calls)

### Streamlit UI: `app.py`

Cost estimates appear automatically after uploading files:

1. Upload documents in Streamlit app
2. Expandable "Cost Estimates" panel appears below file upload
3. Shows token count and estimated cost for selected model
4. Click "Process Files" to run actual extraction

**UI Features**:
- Non-blocking: Errors in estimation don't prevent extraction
- Expandable: Collapse to reduce visual noise
- Model-specific: Shows cost for currently selected provider/model
- Multi-model mode: Optional comparison table across all models

### Python API

#### Two-Layer Cost Estimation (Recommended)

```python
from src.ui.cost_estimator import estimate_cost_two_layer, estimate_all_models_two_layer

# Estimate full pipeline cost (document extraction + event extraction)
result = estimate_cost_two_layer(
    uploaded_files=["contract.pdf", "agreement.docx"],
    doc_extractor="qwen_vl",
    event_model="gpt-4o-mini",
    extracted_texts=None  # Optional: provide pre-extracted texts
)

print(f"📄 Document extraction: {result['document_cost_display']}")
print(f"🤖 Event extraction: {result['event_cost_display']}")
print(f"💰 Total estimated: {result['total_cost_display']}")
print(f"📊 Page count: {result['page_count']} (confidence: {result['page_confidence']})")

# Multi-model comparison with two-layer costs
estimates = estimate_all_models_two_layer(
    uploaded_files=["contract.pdf"],
    doc_extractor="docling",  # Free local OCR
    provider="openai",
    extracted_texts=None
)

for model_id, est in estimates.items():
    print(f"{model_id}: {est['total_cost_display']} total")
```

#### Single-Layer Cost Estimation (Legacy - Event Extraction Only)

```python
from src.ui.cost_estimator import estimate_cost, estimate_all_models, get_cost_summary

# Single model estimate (Layer 2 only)
text = document.plain_text
result = estimate_cost(text, "gpt-4o-mini")

print(f"Estimated cost: ${result['cost_usd']:.4f}")
print(f"Total tokens: {result['tokens_total']:,}")
print(f"Input tokens: {result['tokens_input']:,}")
print(f"Output tokens: {result['tokens_output']:,}")

# Multi-model comparison (Layer 2 only)
estimates = estimate_all_models(text, provider="openai")
for model_id, est in estimates.items():
    print(f"{model_id}: ${est['cost_usd']:.4f}")

# Cost summary statistics
summary = get_cost_summary(text, recommended_only=True)
print(f"Cheapest model: {summary['cheapest_model']}")
print(f"Cost range: {summary['cost_range_display']}")
```

---

## API Reference

### Core Functions

#### `estimate_tokens(text: str) -> int`

Estimate token count from text using character-based heuristic.

**Parameters**:
- `text` (str): Input text to estimate

**Returns**:
- `int`: Estimated token count (minimum 1)

**Example**:
```python
>>> estimate_tokens("Hello world")
2  # 11 chars / 4 = 2.75 → 2 tokens
```

---

#### `split_input_output_tokens(total_tokens: int) -> tuple[int, int]`

Split total tokens into input/output using 90/10 ratio.

**Parameters**:
- `total_tokens` (int): Total estimated token usage

**Returns**:
- `tuple[int, int]`: (input_tokens, output_tokens)

**Example**:
```python
>>> split_input_output_tokens(1000)
(900, 100)
```

---

#### `estimate_cost(text: str, model_id: str, input_output_split: Optional[tuple[int, int]] = None) -> Dict`

Estimate cost for processing text with a specific model.

**Parameters**:
- `text` (str): Input text to process
- `model_id` (str): Model identifier from model catalog (e.g., `"gpt-4o-mini"`)
- `input_output_split` (Optional[tuple[int, int]]): Override default 90/10 split

**Returns**:
- `Dict` with keys:
  - `model_id` (str): Model identifier
  - `display_name` (str): Human-readable model name
  - `tokens_total` (int): Total estimated tokens
  - `tokens_input` (int): Input tokens (90%)
  - `tokens_output` (int): Output tokens (10%)
  - `cost_usd` (float): Estimated cost in USD
  - `cost_display` (str): Formatted cost string
  - `pricing_available` (bool): Whether pricing data exists
  - `note` (str): Caveat about estimate accuracy

**Example**:
```python
>>> result = estimate_cost("Legal document...", "gpt-4o-mini")
>>> result
{
    'model_id': 'gpt-4o-mini',
    'display_name': 'GPT-4o Mini',
    'tokens_total': 1200,
    'tokens_input': 1080,
    'tokens_output': 120,
    'cost_usd': 0.000234,
    'cost_display': '$0.0002',
    'pricing_available': True,
    'note': 'Estimates only — actual billing may vary by ±20%'
}
```

---

#### `estimate_all_models(text: str, provider: Optional[str] = None, category: Optional[str] = None, recommended_only: bool = False) -> Dict[str, Dict]`

Estimate costs across multiple models for comparison.

**Parameters**:
- `text` (str): Input text to process
- `provider` (Optional[str]): Filter by provider (`"anthropic"`, `"openai"`, `"google"`, etc.)
- `category` (Optional[str]): Filter by category (`"Ground Truth"`, `"Production"`, `"Budget"`)
- `recommended_only` (bool): Only include recommended models

**Returns**:
- `Dict[str, Dict]`: Mapping of model_id to estimate dict

**Example**:
```python
>>> estimates = estimate_all_models(text, provider="openai")
>>> sorted_by_cost = sorted(estimates.items(), key=lambda x: x[1]['cost_usd'])
>>> cheapest = sorted_by_cost[0]
>>> print(f"Cheapest: {cheapest[0]} at ${cheapest[1]['cost_usd']:.4f}")
Cheapest: gpt-4o-mini at $0.0002
```

---

#### `get_cost_summary(text: str, provider: Optional[str] = None) -> Dict`

Get summary statistics for cost estimation across models.

**Parameters**:
- `text` (str): Input text to process
- `provider` (Optional[str]): Filter by provider

**Returns**:
- `Dict` with keys:
  - `token_estimate` (int): Total estimated tokens
  - `models_analyzed` (int): Number of models with pricing
  - `cost_min` (float): Minimum cost across models
  - `cost_max` (float): Maximum cost across models
  - `cost_range_display` (str): Formatted cost range
  - `cheapest_model` (str): Model ID with lowest cost
  - `most_expensive_model` (str): Model ID with highest cost

---

### Two-Layer Functions

#### `estimate_cost_two_layer(uploaded_files: List, doc_extractor: str, event_model: str, extracted_texts: Optional[List[str]] = None) -> Dict`

Estimate total cost including both document extraction (Layer 1) and event extraction (Layer 2).

**Parameters**:
- `uploaded_files` (List[Union[Path, str, BinaryIO]]): List of file paths or file-like objects
- `doc_extractor` (str): Document extractor ID (`"docling"`, `"qwen_vl"`, `"gemini"`)
- `event_model` (str): Event extraction model ID from model catalog
- `extracted_texts` (Optional[List[str]]): Pre-extracted texts (if available, for Layer 2 only)

**Returns**:
- `Dict` with layered cost breakdown:
  - `document_cost` (float): Layer 1 cost in USD
  - `document_cost_display` (str): Layer 1 formatted cost
  - `document_extractor` (str): Extractor display name
  - `event_cost` (float): Layer 2 cost in USD
  - `event_cost_display` (str): Layer 2 formatted cost
  - `event_model` (str): Model display name
  - `total_cost` (float): Combined cost in USD
  - `total_cost_display` (str): Combined formatted cost
  - `page_count` (int): Total estimated pages
  - `page_confidence` (str): Confidence level (`"high"`, `"medium"`, `"low"`)
  - `tokens_total` (int): Event extraction tokens
  - `tokens_input` (int): Input tokens
  - `tokens_output` (int): Output tokens
  - `file_count` (int): Number of files
  - `pricing_available` (bool): Both layers have pricing
  - `note` (str): Accuracy caveat

**Example**:
```python
>>> result = estimate_cost_two_layer(
...     uploaded_files=["contract.pdf", "agreement.pdf"],
...     doc_extractor="qwen_vl",
...     event_model="gpt-4o-mini"
... )
>>> result
{
    'document_cost': 0.0102,
    'document_cost_display': '$0.0102',
    'document_extractor': 'Qwen3-VL (Budget Vision)',
    'event_cost': 0.0045,
    'event_cost_display': '$0.0045',
    'event_model': 'GPT-4o Mini',
    'total_cost': 0.0147,
    'total_cost_display': '$0.0147',
    'page_count': 20,
    'page_confidence': 'high',
    'tokens_total': 30000,
    'pricing_available': True,
    'note': 'Two-layer estimate (document + event extraction). Actual costs may vary ±20-30%.'
}
```

---

#### `estimate_all_models_two_layer(uploaded_files: List, doc_extractor: str, provider: Optional[str] = None, recommended_only: bool = False, extracted_texts: Optional[List[str]] = None) -> Dict[str, Dict]`

Estimate costs across multiple event extraction models (all with same document extractor).

Useful for comparing Layer 2 costs while keeping Layer 1 constant.

**Parameters**:
- `uploaded_files` (List): List of file paths or file-like objects
- `doc_extractor` (str): Document extractor ID (shared across all estimates)
- `provider` (Optional[str]): Filter event models by provider
- `recommended_only` (bool): Only include recommended event models
- `extracted_texts` (Optional[List[str]]): Pre-extracted texts

**Returns**:
- `Dict[str, Dict]`: Mapping of model_id to two-layer estimate dict

**Example**:
```python
>>> estimates = estimate_all_models_two_layer(
...     uploaded_files=["contract.pdf"],
...     doc_extractor="docling",
...     provider="openai"
... )
>>> for model_id, est in sorted(estimates.items(), key=lambda x: x[1]['total_cost']):
...     print(f"{model_id}: {est['total_cost_display']}")
gpt-4o-mini: $0.0045
gpt-4o: $0.0150
gpt-4-turbo: $0.0300
```

---

### Document Page Estimator

#### `estimate_document_pages(file_path: Union[Path, str, BinaryIO], filename: str = None) -> Tuple[int, str]`

Estimate document page count without extraction (no API calls).

**Parameters**:
- `file_path` (Union[Path, str, BinaryIO]): File path, path string, or file-like object
- `filename` (Optional[str]): Original filename (required for BytesIO objects)

**Returns**:
- `Tuple[int, str]`: (page_count, confidence)
  - `page_count` (int): Estimated number of pages/images
  - `confidence` (str): `"high"` (metadata), `"medium"` (file size), or `"low"` (default)

**Example**:
```python
>>> from src.ui.document_page_estimator import estimate_document_pages
>>> estimate_document_pages("contract.pdf")
(23, 'high')  # PDF metadata read

>>> estimate_document_pages("scan.jpg")
(1, 'high')  # Single image

>>> import io
>>> pdf_bytes = io.BytesIO(open("doc.pdf", "rb").read())
>>> estimate_document_pages(pdf_bytes, filename="doc.pdf")
(15, 'medium')  # File size heuristic
```

---

#### `get_confidence_message(confidence: str, page_count: int) -> str`

Get user-friendly confidence message for UI display.

**Parameters**:
- `confidence` (str): Confidence level (`"high"`, `"medium"`, `"low"`)
- `page_count` (int): Estimated page count

**Returns**:
- `str`: Human-readable confidence message

**Example**:
```python
>>> from src.ui.document_page_estimator import get_confidence_message
>>> get_confidence_message('high', 15)
'~15 pages (from metadata)'

>>> get_confidence_message('medium', 12)
'~12 pages (estimated from file size ±30%)'

>>> get_confidence_message('low', 15)
'~15 pages (default estimate - actual may vary)'
```

---

### Document Extractor Catalog

#### `get_doc_extractor(extractor_id: str) -> Optional[DocExtractorEntry]`

Get document extractor metadata by ID.

**Parameters**:
- `extractor_id` (str): Extractor identifier (`"docling"`, `"qwen_vl"`, `"gemini"`)

**Returns**:
- `DocExtractorEntry` if found, `None` otherwise

**Example**:
```python
>>> from src.core.document_extractor_catalog import get_doc_extractor
>>> extractor = get_doc_extractor("qwen_vl")
>>> extractor.cost_per_page
0.00512
>>> extractor.cost_display
'$0.077 per 15-page doc'
```

---

#### `estimate_doc_cost(extractor_id: str, page_count: int) -> Dict`

Estimate document extraction cost for a specific extractor.

**Parameters**:
- `extractor_id` (str): Extractor identifier
- `page_count` (int): Number of pages to process

**Returns**:
- `Dict` with cost estimate:
  - `extractor_id` (str): Extractor identifier
  - `display_name` (str): Extractor display name
  - `page_count` (int): Number of pages
  - `cost_per_page` (float): Cost per page in USD
  - `cost_usd` (float): Total cost in USD
  - `cost_display` (str): Formatted cost string
  - `pricing_available` (bool): Whether pricing data exists

**Example**:
```python
>>> from src.core.document_extractor_catalog import estimate_doc_cost
>>> estimate_doc_cost("qwen_vl", page_count=15)
{
    'extractor_id': 'qwen_vl',
    'display_name': 'Qwen3-VL (Budget Vision)',
    'page_count': 15,
    'cost_per_page': 0.00512,
    'cost_usd': 0.0768,
    'cost_display': '$0.0768',
    'pricing_available': True
}
```

---

### Calibration Helpers

#### `calculate_accuracy(estimated_tokens: int, actual_tokens: int) -> float`

Calculate estimation accuracy as percentage error.

Use this with actual token counts from LLM API responses to calibrate the estimator.

**Parameters**:
- `estimated_tokens` (int): Token count from `estimate_tokens()`
- `actual_tokens` (int): Actual token count from API response

**Returns**:
- `float`: Accuracy as float (1.0 = perfect, 0.8 = 20% error)

**Example**:
```python
>>> accuracy = calculate_accuracy(1000, 950)
>>> print(f"Estimation accuracy: {accuracy:.1%}")
Estimation accuracy: 95.0%
```

---

#### `suggest_calibration_factor(estimated_tokens_list: List[int], actual_tokens_list: List[int]) -> float`

Suggest calibration factor based on historical estimates vs actuals.

**Parameters**:
- `estimated_tokens_list` (List[int]): List of estimated token counts
- `actual_tokens_list` (List[int]): List of actual token counts (same length)

**Returns**:
- `float`: Suggested multiplier for `CHARS_PER_TOKEN` (e.g., 0.9 means reduce by 10%)

**Example**:
```python
>>> factor = suggest_calibration_factor([1000, 2000], [1100, 2200])
>>> print(f"Recommended adjustment: {factor:.2f}x")
Recommended adjustment: 1.10x
```

---

## Calibration Workflow

Over time, you can improve estimation accuracy by comparing estimates against actual token usage:

### Step 1: Collect Actual Token Data

After processing documents, the pipeline stores actual token counts in `pipeline_metadata.py`:

```python
{
    "tokens_input": 1050,   # From LLM API response
    "tokens_output": 125,
    "tokens_total": 1175
}
```

### Step 2: Compare Estimates vs Actuals

```python
from src.ui.cost_estimator import estimate_tokens, calculate_accuracy

# Get estimate
estimated = estimate_tokens(document_text)

# Get actual from metadata
actual = metadata["tokens_total"]

# Calculate accuracy
accuracy = calculate_accuracy(estimated, actual)
print(f"Accuracy: {accuracy:.1%}")
```

### Step 3: Calibrate CHARS_PER_TOKEN

After collecting 10+ samples:

```python
from src.ui.cost_estimator import suggest_calibration_factor

estimated_list = [1000, 2000, 1500, ...]  # From estimate_tokens()
actual_list = [1100, 2150, 1625, ...]     # From API responses

factor = suggest_calibration_factor(estimated_list, actual_list)
print(f"Suggested adjustment: {factor:.2f}x")

# If factor = 1.10, update CHARS_PER_TOKEN:
# OLD: CHARS_PER_TOKEN = 4.0
# NEW: CHARS_PER_TOKEN = 4.0 / 1.10 = 3.64
```

### Step 4: Monitor and Iterate

- Track accuracy over time using `calculate_accuracy()`
- Re-calibrate when accuracy drops below 80%
- Consider per-provider calibration if token counts vary significantly

---

## Future Enhancements

### Short-term (Next 3 months)

1. **Per-Provider Calibration**: Different models may tokenize differently
   - OpenAI: GPT-4 tokenizer (cl100k_base)
   - Anthropic: Claude tokenizer (slightly different)
   - Google: Gemini tokenizer (may differ for non-English)

2. **Confidence Intervals**: Show range instead of point estimate
   - Example: "1000-1200 tokens (±10%)"
   - Based on historical variance for similar documents

3. **Document Type Heuristics**: Adjust estimates by document type
   - Legal contracts: 3.8 chars/token (dense, formal language)
   - Court filings: 4.2 chars/token (narrative, conversational)
   - Financial docs: 4.5 chars/token (numbers, tables, whitespace)

### Long-term (6+ months)

4. **Regression Model**: Train ML model on actual token data
   - Features: char count, word count, punctuation ratio, document type
   - Target: actual token count from API responses
   - Expected improvement: ±20% → ±5% accuracy

5. **Real-time Tokenization**: Use actual tokenizers when available
   - `tiktoken` for OpenAI models
   - `anthropic-tokenizer` for Claude models
   - Trade-off: Requires library dependencies, slower than char heuristic

6. **Batch Optimization Recommendations**: Suggest cost-saving strategies
   - "Combine small documents to reduce per-request overhead"
   - "Use gpt-4o-mini instead of gpt-4o to save 95% ($X.XX → $X.XX)"
   - "Enable caching for repeated extractions (50% cost reduction)"

---

## Testing

The cost estimator has comprehensive unit tests covering both single-layer and two-layer functionality:

```bash
# Run all cost estimator tests
uv run python tests/test_cost_estimator.py

# Or with pytest
uv run python -m pytest tests/test_cost_estimator.py -v

# Run specific test class
uv run python -m pytest tests/test_cost_estimator.py::TestTwoLayerCostEstimation -v

# Run with coverage
uv run python -m pytest tests/test_cost_estimator.py --cov=src/ui/cost_estimator
```

**Test Coverage** (72 unit tests):

**Single-Layer Tests** (28 tests):
- Token estimation accuracy (6 tests)
- Cost calculation with real pricing (6 tests)
- Multi-model comparison (5 tests)
- Cost summary statistics (3 tests)
- Calibration helpers (6 tests)
- Model catalog integration (2 tests)

**Two-Layer Tests** (44 tests):
- Document page estimator (8 tests)
  - Metadata reads (PDF, DOCX, PPTX, images)
  - File size heuristics
  - Confidence levels
  - Edge cases (zero-size files, unknown formats)
- Document extractor catalog (18 tests)
  - Extractor retrieval (Docling, Qwen-VL, Gemini)
  - Filtering (provider, vision support, cost, recommendation)
  - Pricing queries
  - Cost estimation
  - Validation utilities
- Two-layer cost estimation (13 tests)
  - Various extractor + model combinations
  - Multiple files
  - Unknown extractors/models
  - Pre-extraction scenarios
  - Multi-model comparison
  - Confidence propagation
  - Cost display formatting
- Edge cases (5 tests)
  - Empty file lists
  - Mismatched file/text counts
  - Very large documents
  - Zero-cost scenarios

---

## Troubleshooting

### "Model not found in catalog"

**Symptom**: Estimate shows `pricing_available: False` and `"Model 'xyz' not found in catalog"`

**Cause**: Model ID doesn't match model catalog entries

**Solution**: Use native model IDs from model catalog, not OpenRouter IDs
- ❌ Wrong: `anthropic/claude-3-haiku` (OpenRouter ID)
- ✅ Correct: `claude-3-haiku-20240307` (Catalog ID)

**List available models**:
```python
from src.core.model_catalog import get_model_catalog

catalog = get_model_catalog()
for model in catalog.list_models():
    print(f"{model.model_id}: {model.display_name}")
```

### Estimates are consistently off by >30%

**Symptom**: `calculate_accuracy()` shows accuracy < 0.7 (>30% error)

**Possible causes**:
1. **Non-English text**: Char/token ratio varies by language (Chinese: ~1.5, English: ~4.0)
2. **Code snippets**: Source code has different tokenization (often more tokens per char)
3. **Tables/JSON**: Structured data may tokenize differently than prose

**Solutions**:
1. Use calibration workflow to adjust `CHARS_PER_TOKEN` for your corpus
2. Consider document-type-specific calibration factors
3. For critical budgeting, use real tokenizers (see Future Enhancements)

### Cost estimates are $0.00 but pricing should exist

**Symptom**: `estimate_cost()` returns `cost_usd: 0.0` even though model has pricing

**Cause**: Model exists in catalog but `cost_input_per_1m` and `cost_output_per_1m` are `None`

**Solution**: Update model catalog entry with pricing data
```python
# In src/core/model_catalog.py
ModelEntry(
    model_id="your-model",
    provider="your-provider",
    cost_input_per_1m=0.15,   # Add pricing
    cost_output_per_1m=0.60,  # Add pricing
    ...
)
```

---

## Related Documentation

**Layer 2 (Event Extraction)**:
- **Model Catalog**: `src/core/model_catalog.py` - Centralized model metadata and pricing
- **Pipeline Metadata**: `src/core/pipeline_metadata.py` - Actual token usage tracking

**Layer 1 (Document Extraction)**:
- **Document Extractor Catalog**: `src/core/document_extractor_catalog.py` - Document extractor pricing
- **Document Page Estimator**: `src/ui/document_page_estimator.py` - Pre-extraction page counting

**UI Integration**:
- **Streamlit Integration**: `src/ui/streamlit_common.py` - UI cost display helpers
- **CLI Integration**: `scripts/classify_documents.py` - Command-line cost estimation

**Implementation**:
- **Cost Estimator Order**: `docs/orders/cost-estimator-002.json` - Two-layer cost system requirements
- **Unit Tests**: `tests/test_cost_estimator.py` - Comprehensive test suite (72 tests)

---

## Change Log

**v2.0.0** (2025-10-17) - Two-Layer Cost Estimation
- ✨ **New**: Two-layer cost estimation (document extraction + event extraction)
- ✨ **New**: Document page estimator with pypdfium2 metadata reads
- ✨ **New**: Document extractor catalog (Docling, Qwen-VL, Gemini pricing)
- ✨ **New**: `estimate_cost_two_layer()` function for full pipeline costs
- ✨ **New**: `estimate_all_models_two_layer()` for multi-model comparison
- ✨ **New**: Confidence levels for page estimates (high/medium/low)
- 🧪 **Test**: 44 new unit tests for two-layer functionality (72 total)
- 📝 **Docs**: Comprehensive two-layer documentation with examples
- 🎨 **UI**: Streamlit displays layered cost breakdown (3-column metrics)
- 🐛 **Fix**: Handle empty file lists gracefully
- 🐛 **Fix**: Add display_name fallback for unknown models

**v1.0.0** (2025-10-16)
- Initial release with character-based token estimation
- 90/10 input/output split assumption
- Multi-model cost comparison
- CLI and Streamlit integration
- Calibration helpers for future refinement
- 28 unit tests for single-layer estimation
