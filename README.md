# ⚖️ Paralegal Date Extraction Test

**Testing docling + langextract combination for paralegal application**

## 🚨 SECURITY NOTICE

**⚠️ IMPORTANT: API Key Security**

This project requires a Google API key for LangExtract functionality. **NEVER commit API keys to version control.**

### Security Checklist:
- [ ] `.env` file contains placeholder, not real API key
- [ ] Real API key stored securely (environment variables, secret manager)
- [ ] `.env` file is properly excluded in `.gitignore`
- [ ] No API keys in commit history

### Safe Setup:
1. Copy `.env.example` to `.env`
2. Replace `your_google_api_key_here` with actual key
3. Verify `.env` is in `.gitignore` before committing

## 🎯 Purpose

**Legal Events Extraction:** Documents In → Five-Column Legal Events Table Out

This is a **proof-of-concept** to evaluate document parsing and AI-based legal event extraction for paralegal applications. The system extracts structured legal events with dates, event particulars, citations, and document references.

## 🧪 Test Scope

- **Core Pipeline:** [Docling](https://github.com/DS4SD/docling) for document parsing + Multiple AI providers for event extraction
- **Business Use Case:** Legal event extraction from court documents, contracts, correspondence, and legal filings
- **Goal:** Test parser + extractor combinations to identify optimal configurations for quality, cost, and speed
- **Provider Flexibility:** Supports LangExtract (Gemini), OpenRouter (11+ tested models), and OpenCode Zen with in-app provider switching

## 🚀 Quick Start

1. **Install UV** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Enter the project directory**:
   ```bash
   cd docling_langextract_testing
   ```

3. **Install dependencies**:
   ```bash
   uv sync
   ```

4. **Launch the test app**:
   ```bash
   uv run streamlit run app.py
   ```

5. **Open your browser** to: `http://localhost:8501`

## 🔄 Provider Selection

This system now supports **multiple event extraction providers** with two selection methods:

### 1. **In-App UI Selection** (Recommended)

The Streamlit application (`app.py`) includes a **provider selector** in the Processing panel that lets you switch between providers without restarting:

- **OpenRouter** (Unified API) - ⭐ **Recommended Default** - Best quality/cost/speed balance
- **Anthropic** (Claude 3 Haiku) - **Speed/Cost Champion** - 10x cheaper, 4x faster
- **OpenAI** (GPT-4o/4-mini) - **Quality Champion** - Most detailed extraction
- **LangExtract** (Google Gemini) - **Completeness Champion** - Captures all details
- **DeepSeek** (Direct API) - **Research Champion** - Advanced reasoning model
- **OpenCode Zen** (Model Router)

**📊 Provider Comparison** (based on 2025-10-03 testing):

**For Digital PDFs** (clean text):
- **OpenRouter**: 8/10 quality, ~$0.015/doc, ~14s ⭐ **Best Overall**
- **Anthropic**: 7/10 quality, $0.003/doc, 4.4s → High-volume processing
- **OpenAI**: 8/10 quality, $0.03/doc, 18s → High-stakes legal work
- **LangExtract**: 6/10 quality, ~$0.01/doc, 36s → Comprehensive analysis

**For Scanned PDFs** (requires OCR):
- **Anthropic**: 10/10 quality, $0.0005/doc, 2.05s ⭐ **OCR Champion** (4x cheaper, 3x faster)
- **OpenAI**: 10/10 quality, $0.0039/doc, 5.96s → Maximum detail
- **OpenRouter**: 10/10 quality, ~$0.008/doc, 8.43s → Consistent quality
- **LangExtract**: 7/10 quality, ~$0.002/doc, 3.82s → Comprehensive but noisy

**Key Finding**: ✅ **OCR does NOT degrade extraction quality** - Docling OCR is production-ready. Anthropic becomes the top choice for scanned documents due to speed/cost advantages when OCR is the bottleneck.

**Phase 4 Benchmark Results** (2025-10-04, 6-provider comparison with 3-judge panel):

| Provider | Overall Quality | Completeness | Accuracy | Citation Quality | Win Rate |
|----------|----------------|--------------|----------|------------------|----------|
| **OpenRouter** | 6.25/10 ⭐ | 6.75/10 | 8.5/10 | 5.0/10 | 100% (2/2) |
| **OpenAI** | 6.25/10 | 6.75/10 | 8.5/10 | 5.0/10 | Tied 2nd |
| **Anthropic** | 4.0/10 | 5.0/10 | 5.5/10 | 2.5/10 | 0% |
| **LangExtract** | 2.75/10 | 5.5/10 | 6.0/10 | **0.0/10** ❌ | 0% |
| **DeepSeek** | N/A | - | - | - | No API key |
| **OpenCode Zen** | 0.0/10 | 0.0/10 | 0.0/10 | 0.0/10 | Extraction failures |

**Key Finding**: **Citation quality is paramount for legal work** - LangExtract extracted 4-5 events but scored lowest due to missing citations. 1 well-cited event beats 5 events without citations.

See detailed evaluations:
- Phase 4 (6-provider benchmark): `config/benchmarks/results/phase4_judge_results_20251004_183300.json`
- Phase 2 (manual eval): `docs/reports/phase2-comparison-2025-10-04.md`
- Digital PDFs: `docs/benchmarks/2025-10-03-manual-comparison.md`
- Scanned PDFs: `docs/benchmarks/2025-10-03-ocr-comparison.md`

**⚠️ Important**: Each provider requires **provider-specific** API keys. The pipeline validates only the key needed for your selected provider:

**Required API Keys (Provider-Specific):**
- **OpenRouter**: `OPENROUTER_API_KEY` (recommended default)
- **Anthropic**: `ANTHROPIC_API_KEY` (for speed/cost optimization)
- **OpenAI**: `OPENAI_API_KEY` (for maximum quality)
- **LangExtract**: `GEMINI_API_KEY` or `GOOGLE_API_KEY` (either one)
- **DeepSeek**: `DEEPSEEK_API_KEY` (for advanced reasoning)
- **OpenCode Zen**: `OPENCODEZEN_API_KEY` (alternative model router)

**How Validation Works:**
- Selecting LangExtract → Validates `GEMINI_API_KEY` only
- Selecting OpenRouter → Validates `OPENROUTER_API_KEY` only (no Gemini key needed)
- Selecting OpenCode Zen → Validates `OPENCODEZEN_API_KEY` only (no Gemini key needed)

If you switch providers, you only need the API key for that specific provider. The app will display a clear error message if the required key is missing.

The selector automatically initializes the pipeline with your chosen provider and displays required credentials in the tooltip.

### 2. **Environment Variable Override**

You can also set a default provider via the `EVENT_EXTRACTOR` environment variable:

```bash
# Use LangExtract (default)
export EVENT_EXTRACTOR=langextract
export GEMINI_API_KEY=your_google_api_key_here

# Use OpenRouter
export EVENT_EXTRACTOR=openrouter
export OPENROUTER_API_KEY=your_openrouter_api_key_here

# Use OpenCode Zen
export EVENT_EXTRACTOR=opencode_zen
export OPENCODEZEN_API_KEY=your_opencode_zen_api_key_here

# Use OpenAI
export EVENT_EXTRACTOR=openai
export OPENAI_API_KEY=your_openai_api_key_here

# Use Anthropic
export EVENT_EXTRACTOR=anthropic
export ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Use DeepSeek
export EVENT_EXTRACTOR=deepseek
export DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

**Note**: The Streamlit UI selector takes precedence over the environment variable during interactive sessions.

## 📊 What Gets Tested

### Core Pipeline:
1. **📄 Docling** - Extracts text from legal documents (PDF, DOCX, TXT, PPTX, HTML)
2. **🌍 Langextract** - Detects document language
3. **📅 Date Extraction** - Finds and normalizes dates (the key business value)

### Test Metrics:
- **Docling Success Rate** by file type
- **Date Extraction Results** (number of dates found)
- **Pipeline Success** (successful text extraction + date extraction)
- **Language Detection** accuracy

## 📁 Project Structure

```
docling_langextract_testing/
├── app.py               # Main Streamlit test application
├── src/main.py          # Command-line version (alternative)
├── .env.example         # Template for environment variables
├── pyproject.toml       # Project configuration
└── README.md            # This documentation
```

## 🏗️ Repository Structure

**Core Directories:**
- **`src/`** - Core pipeline logic, interfaces, and modular components
  - `core/` - Pipeline orchestration, interfaces, and configuration
  - `extractors/` - Individual extractor implementations
  - `ui/` - Shared Streamlit UI components
  - `utils/` - File handling utilities
  - `visualization/` - Data visualization components
- **`tests/`** - Comprehensive test suites and validation procedures
- **`examples/`** - Demo Streamlit applications (5 apps for different testing scenarios)
- **`docs/`** - Design documents, architecture decisions, and project orders
  - `adr/` - Architecture Decision Records
  - `orders/` - Active housekeeping orders for contributors
  - `reports/` - Completion reports and documentation
- **`scripts/`** - Development utilities and troubleshooting guides
- **`output/`** - Generated results and extracted data files

**Key Design Documents:**
- [📋 Pluggable Extractors PRD](docs/pluggable_extractors_prd.md) - Product requirements and specifications
- [🏛️ ADR-001: Pluggable Extractors](docs/adr/ADR-001-pluggable-extractors.md) - Architecture decision record

## 🔬 Test Results

The app provides:
- ✅ **Success/Failure** indicators for each library
- 📊 **Visual charts** showing performance by document type
- 📋 **Data export** of extracted dates
- 📈 **Success rate metrics**

## ⚠️ Important Notes

- This is a **TEST SCRIPT** - guaranteed five-column table output with fallback rows on failures
- **Multiple event extractors supported**: LangExtract (Gemini), OpenRouter, OpenCode Zen (select via UI)
- **Pure testing environment** to evaluate parser+extractor combinations
- Results help determine which combination suits paralegal applications

## ⏱️ Performance Metrics

The system includes **built-in performance timing** to measure document processing speed and identify bottlenecks in the extraction pipeline.

### How It Works

Performance timing captures execution duration for two critical phases:
1. **Docling Extraction** - PDF parsing, OCR, and text extraction time
2. **Event Extraction** - LLM API call and legal event extraction time

Timing is captured **per-document** (all events from the same document share identical timing values).

### Enabling/Disabling Timing

Control timing instrumentation via the `ENABLE_PERFORMANCE_TIMING` environment variable:

```bash
# Enable timing (default for development/testing)
ENABLE_PERFORMANCE_TIMING=true

# Disable timing (recommended for production to reduce overhead)
ENABLE_PERFORMANCE_TIMING=false
```

**Default**: `true` (timing enabled)

### Where Timing Data Appears

#### 1. **Console Logs**
When timing is enabled, the pipeline logs performance metrics for each document:

```
⏱️  Answer to Request for Arbitration.pdf: Docling=2.341s, Extractor=3.567s, Total=5.908s
```

#### 2. **Streamlit UI**
The web interface displays **Performance Metrics** with average timing across all processed documents:

- **Avg Docling Time** - Average PDF parsing duration
- **Avg Extractor Time** - Average LLM extraction duration
- **Avg Total Time** - End-to-end processing time

#### 3. **Export Files (CSV/JSON/XLSX)**
All exports include three additional timing columns when timing is enabled:

| Column Name | Description | Example Value |
|------------|-------------|---------------|
| `Docling_Seconds` | Docling parsing time | 2.341 |
| `Extractor_Seconds` | LLM extraction time | 3.567 |
| `Total_Seconds` | Combined processing time | 5.908 |

**Example CSV Export**:
```csv
No,Date,Event Particulars,Citation,Document Reference,Docling_Seconds,Extractor_Seconds,Total_Seconds
1,2024-09-21,Lease agreement entered,RTA 2010,lease.pdf,1.234,2.567,3.801
2,2024-10-01,Security deposit paid,RTA 2010,lease.pdf,1.234,2.567,3.801
```

### Expected Performance

Timing varies based on document size, complexity, OCR requirements, and API latency:

| Document Size | Docling Time | Extractor Time | Total Time |
|--------------|--------------|----------------|------------|
| Small PDF (~15 pages) | 1-3s | 2-5s | 3-8s |
| Medium PDF (~50 pages) | 3-10s | 3-8s | 6-18s |
| Large PDF (100+ pages) | 10-30s | 5-15s | 15-45s |

**Factors Affecting Performance**:
- **OCR Requirement**: Documents needing OCR take significantly longer
- **Table Complexity**: `DOCLING_TABLE_MODE=ACCURATE` increases processing time
- **Provider Latency**: LLM API response times vary by provider and model
- **Document Format**: PDF requires more processing than TXT/HTML

### Interpreting Timing Data

Use timing metrics to:
- **Identify Bottlenecks**: If `Docling_Seconds` dominates, consider faster parsing settings
- **Optimize Provider Selection**: Compare extractor performance across providers
- **Capacity Planning**: Estimate processing time for large document batches
- **Cost Analysis**: Longer extraction times often correlate with higher API costs

### Timing Precision

- Uses `time.perf_counter()` for high-resolution timing (millisecond precision)
- Displays 3 decimal places (e.g., `2.341s`)
- Timing is document-level, not per-event (all events from same doc share timing)

### When Timing is Disabled

When `ENABLE_PERFORMANCE_TIMING=false`:
- No timing capture or logging
- Exports contain only the core 5 columns (no timing columns)
- Streamlit UI shows no performance metrics section
- Eliminates timing overhead for production use

## 🤖 Assistant Guardrails

When using Claude (or any other AI helper) with this repository, keep it focused on the documented proof-of-concept scope:

- **Stay on mission:** Only propose or modify code that contributes directly to "documents in → legal events table out" testing. Skip unrelated refactors, new features, or production hardening.
- **No fake extractor usage:** Do not mark event extractors (LangExtract, OpenRouter, OpenCode Zen) as successful unless the real API is called. If API access is missing, halt and report the gap—do **not** introduce regex or other fallbacks.
- **Respect toggles and config:** Any automation must treat the Streamlit UI selections (provider choice, model selection) as the source of truth—no hard-coded overrides.
- **Guard sample data:** Avoid inventing or writing back large synthetic documents; use small snippets that illustrate test cases.
- **Document assumptions:** When APIs are mocked, call that out explicitly so downstream testing knows the difference.

## 🔌 This Is How LangExtract Can Be Pinged

The snippet below demonstrates a minimal end-to-end call against the real LangExtract extractor using the `GEMINI_API_KEY` defined in your `.env` file. It loads a short legal paragraph, supplies a concrete few-shot example, and prints the structured results returned by Gemini.

```python
import os
from dotenv import load_dotenv
import langextract as lx

load_dotenv()  # ensures GEMINI_API_KEY is available via environment

legal_text = """
This Lease Agreement is entered into on September 21, 2025. The lease begins on
October 1, 2025 and rent is due on the 5th of every month.
""".strip()

examples = [
    lx.data.ExampleData(
        text="This contract was signed on March 15, 2024 and becomes effective on April 1, 2024.",
        extractions=[
            lx.data.Extraction(
                extraction_class="contract_date",
                extraction_text="March 15, 2024",
                attributes={"normalized_date": "2024-03-15", "type": "signing_date"},
            ),
            lx.data.Extraction(
                extraction_class="effective_date",
                extraction_text="April 1, 2024",
                attributes={"normalized_date": "2024-04-01", "type": "effective_date"},
            ),
        ],
    )
]

response = lx.extract(
    text_or_documents=legal_text,
    prompt_description="Extract every legally meaningful date and provide a normalized ISO date.",
    examples=examples,
    model_id="gemini-2.0-flash",
    api_key=os.environ["GEMINI_API_KEY"],
)

for item in response.extractions:
    attrs = item.attributes or {}
    print(item.extraction_class, "→", item.extraction_text, attrs.get("normalized_date"))
```

⚠️ Run this only when you have valid Gemini access—each invocation makes a real LLM call and may incur usage costs.

## 🚀 Gemini 2.0 Flash Model

This project now uses **Gemini 2.0 Flash** (`gemini-2.0-flash`) for enhanced legal event extraction capabilities. The newer model provides improved accuracy and better handling of complex legal document structures.

### Model Configuration

- **Default Model**: `gemini-2.0-flash` (configured in `src/core/constants.py`)
- **Environment Override**: Set `GEMINI_MODEL_ID` environment variable to use a different model
- **API Access**: Ensure your Google Cloud project has access to Gemini 2.0 Flash models

### Model Override Example

```bash
# Use a different model for testing
export GEMINI_MODEL_ID="gemini-2.0-flash"
uv run streamlit run app.py

# Use experimental model
export GEMINI_MODEL_ID="gemini-2.0-flash-exp"
uv run streamlit run app.py
```

**Note**: Different environments may require different model access. Use the environment variable override if your deployment environment has different model availability.

### ✅ Verified Active Configuration

**Current Status** (as of latest verification):
- ✅ Default model: `gemini-2.0-flash`
- ✅ Environment override: Not set (using default)
- ✅ API endpoints: `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash`
- ✅ Extraction performance: 13 events extracted with 499-character descriptions

**Environment Variable Precedence:**
- If `GEMINI_MODEL_ID` is set → Uses that model (overrides default)
- If `GEMINI_MODEL_ID` is unset → Uses `DEFAULT_MODEL = "gemini-2.0-flash"`

**Troubleshooting Model Issues:**
```bash
# Check current environment
env | grep GEMINI_MODEL_ID

# Clear override to use default 2.0 Flash
unset GEMINI_MODEL_ID

# Verify model in logs
uv run python3 -c "from src.core.langextract_client import LangExtractClient; print(f'Model: {LangExtractClient().model_id}')"
```

## 📋 LangExtract Prompt Contract

This project enforces a **standardized prompt contract** for consistent legal events extraction. All LangExtract calls must return exactly four JSON keys to ensure reliable five-column table output.

### Required JSON Schema

Every extracted legal event must include these four keys:

```json
{
  "event_particulars": "Complete description (2-8 sentences) with comprehensive context, parties, procedural background, and implications",
  "citation": "Legal citation or reference (empty string if none exists)",
  "document_reference": "Source document filename (automatically set)",
  "date": "Specific date mentioned (empty string if not found)"
}
```

### Key Policies

- **Enhanced Context**: The `event_particulars` field requires 2-8 sentences as appropriate to provide comprehensive legal context for AI summarization
- **No Hallucinated Citations**: The `citation` field must remain empty (`""`) when no verbatim legal reference exists in the text
- **Anchored Document References**: The `document_reference` field is automatically populated with the source filename passed to the extraction method
- **Empty Strings for Missing Data**: Use empty strings (`""`) instead of placeholder text for missing values
- **Required Keys**: All four keys must be present in every extraction
- **Character Offsets**: When available from LangExtract, character offsets are captured in the raw payload for precise source attribution

### Implementation

The standardized prompt is defined in `src/core/constants.py` as `LEGAL_EVENTS_PROMPT` and used consistently across all LangExtract operations. This ensures:

- Consistent extraction format across the entire application
- No invented legal citations that could mislead users
- Reliable document reference tracking
- Maintainable prompt updates from a single location
- Enhanced context (2-8 sentences) for future GPT-5 integration and AI summarization
- Character offset capture when available for precise source attribution

### Example Output

```json
[
  {
    "event_particulars": "On January 15, 2024, the plaintiff filed a motion to dismiss the complaint pursuant to Rule 12(b)(6) of the Federal Rules of Civil Procedure. This motion challenges the legal sufficiency of the complaint, arguing that the plaintiff has failed to state a claim upon which relief can be granted. The filing of this motion suspends the defendant's obligation to file an answer until the court rules on the motion. If granted, the motion would result in dismissal of some or all claims without the need for further discovery or trial proceedings.",
    "citation": "Fed. R. Civ. P. 12(b)(6)",
    "document_reference": "legal_filing.pdf",
    "date": "2024-01-15"
  },
  {
    "event_particulars": "A settlement conference was scheduled for April 10, 2024, to facilitate negotiations between the parties before proceeding to trial. This judicial settlement conference will be overseen by a magistrate judge who will help the parties explore potential resolution of the dispute. The conference represents a crucial opportunity for both sides to assess the strengths and weaknesses of their positions and potentially reach a mutually agreeable resolution.",
    "citation": "",
    "document_reference": "legal_filing.pdf",
    "date": "2024-04-10"
  }
]
```

**Enhanced Context Features:**
- **Comprehensive Descriptions**: Each `event_particulars` contains 2-8 sentences with full legal context
- **Empty Citations**: The second event has an empty `citation` field since no legal reference was mentioned
- **Character Offsets**: When available, `char_start` and `char_end` attributes provide precise source location (hidden from table display)
- **GPT-5 Ready**: Rich context enables advanced AI summarization and analysis

## 🔧 Pluggable Pipeline Architecture

This project features a **modular, configurable pipeline** that allows easy swapping of document processing and event extraction components.

### Core Architecture

The system uses **adapter interfaces** to decouple processing logic from specific implementations:

- **DocumentExtractor**: Interface for text extraction (currently: Docling)
  - Returns `ExtractedDocument` with `markdown`, `plain_text`, and `metadata` fields
- **EventExtractor**: Interface for legal events extraction (currently: LangExtract)
  - Returns `EventRecord` instances with `attributes` field containing LangExtract metadata
- **ExtractorFactory**: Creates configured extractors based on environment settings

### Configuration Options

#### Docling Document Processing
Control Docling behavior via environment variables:

```bash
# OCR and table processing
DOCLING_DO_OCR=true                    # Enable/disable OCR (default: true)
DOCLING_DO_TABLE_STRUCTURE=true        # Enable table structure detection (default: true)
DOCLING_TABLE_MODE=FAST                # Table mode: FAST or ACCURATE (default: FAST)
DOCLING_DO_CELL_MATCHING=true          # Enable table cell matching (default: true)

# Backend and performance
DOCLING_BACKEND=default                # Backend: default or v2 (default: default)
DOCLING_ACCELERATOR_DEVICE=cpu         # Device: cpu, cuda, mps (default: cpu)
DOCLING_ACCELERATOR_THREADS=4          # Thread count (default: 4)
DOCLING_DOCUMENT_TIMEOUT=300           # Processing timeout in seconds (default: 300)

# Optional paths
DOCLING_ARTIFACTS_PATH=/path/to/cache  # Cache directory (optional)
```

#### LangExtract Event Processing
Configure LangExtract behavior:

```bash
# Model settings
GEMINI_MODEL_ID=gemini-2.0-flash       # Override default model (default: gemini-2.0-flash)
LANGEXTRACT_TEMPERATURE=0.0            # Model temperature (default: 0.0)
LANGEXTRACT_MAX_WORKERS=10             # Parallel workers (default: 10)
LANGEXTRACT_DEBUG=false                # Debug mode (default: false)

# Required API access
GEMINI_API_KEY=your_google_api_key_here
```

#### Extractor Selection
Choose different implementations:

```bash
# Component selection (for future extensibility)
DOC_EXTRACTOR=docling                  # Document extractor type (default: docling)
EVENT_EXTRACTOR=langextract             # Event extractor type (default: langextract)
```

These selections are now managed through the `ExtractorConfig` dataclass, providing type-safe configuration.

### Example: High-Performance Configuration

```bash
# Optimized for accuracy and performance
DOCLING_DO_OCR=true
DOCLING_TABLE_MODE=ACCURATE
DOCLING_BACKEND=v2
DOCLING_ACCELERATOR_DEVICE=cuda
DOCLING_ACCELERATOR_THREADS=8
DOCLING_DOCUMENT_TIMEOUT=600

GEMINI_MODEL_ID=gemini-2.0-flash
LANGEXTRACT_TEMPERATURE=0.0
LANGEXTRACT_MAX_WORKERS=20

uv run streamlit run app.py
```

### Adding New Extractors

The adapter pattern makes it easy to add new implementations:

1. **Implement the interface** (`DocumentExtractor` or `EventExtractor`)
2. **Register in factory** (`extractor_factory.py`)
3. **Configure via environment** variables

This design enables **A/B testing**, **gradual migrations**, and **vendor flexibility** without changing application logic.

### Tesseract OCR Setup (Recommended)

Tesseract is the default OCR engine and is **3x faster** than EasyOCR with **31% better text extraction**.

**Installation:**

```bash
# macOS (Homebrew)
brew install tesseract

# Linux (Ubuntu/Debian)
sudo apt install tesseract-ocr libtesseract-dev

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

**Configuration:**

Set the `TESSDATA_PREFIX` environment variable to point to Tesseract's language data directory:

```bash
# macOS (Homebrew)
export TESSDATA_PREFIX=/usr/local/opt/tesseract/share/tessdata

# Linux (apt)
export TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

# Windows
set TESSDATA_PREFIX=C:\Program Files\Tesseract-OCR\tessdata
```

**Verify Installation:**

```bash
tesseract --version
echo $TESSDATA_PREFIX  # Should show the path
```

**Performance:** Tesseract processes scanned legal documents at **22s/page** vs EasyOCR's **70s/page** (3x faster).

See benchmark: [`docs/benchmarks/2025-10-03-ocr-engine-war.md`](docs/benchmarks/2025-10-03-ocr-engine-war.md)

### Environment Variables Quick Reference

| Variable | Default | Description |
|----------|---------|-------------|
| **Extractor Selection** |  |  |
| `DOC_EXTRACTOR` | `docling` | Document extractor type |
| `EVENT_EXTRACTOR` | `langextract` | Event extractor type (langextract, openrouter, opencode_zen) |
| **Docling Configuration** |  |  |
| `DOCLING_DO_OCR` | `true` | Enable/disable OCR |
| `DOCLING_DO_TABLE_STRUCTURE` | `true` | Enable table structure detection |
| `DOCLING_TABLE_MODE` | `FAST` | Table mode: FAST or ACCURATE |
| `DOCLING_DO_CELL_MATCHING` | `true` | Enable table cell matching |
| `DOCLING_BACKEND` | `default` | Backend: default or v2 |
| `DOCLING_ACCELERATOR_DEVICE` | `cpu` | Device: cpu, cuda, mps |
| `DOCLING_ACCELERATOR_THREADS` | `4` | Thread count |
| `DOCLING_DOCUMENT_TIMEOUT` | `300` | Processing timeout (seconds) |
| `DOCLING_ARTIFACTS_PATH` | _(unset)_ | Cache directory (optional) |
| **LangExtract Configuration** |  |  |
| `GEMINI_API_KEY` | _(required)_ | Google API key for Gemini |
| `GEMINI_MODEL_ID` | `gemini-2.0-flash` | Override default model |
| `LANGEXTRACT_TEMPERATURE` | `0.0` | Model temperature |
| `LANGEXTRACT_MAX_WORKERS` | `10` | Parallel workers |
| `LANGEXTRACT_DEBUG` | `false` | Debug mode |
| **OpenRouter Configuration** |  |  |
| `OPENROUTER_API_KEY` | _(required for OpenRouter)_ | OpenRouter API key |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter API base URL |
| `OPENROUTER_MODEL` | `anthropic/claude-3-haiku` | OpenRouter model to use |
| `OPENROUTER_TIMEOUT` | `30` | Request timeout in seconds |
| **OpenCode Zen Configuration** |  |  |
| `OPENCODEZEN_API_KEY` | _(required for OpenCode Zen)_ | OpenCode Zen API key |
| `OPENCODEZEN_BASE_URL` | `https://api.opencode-zen.example/v1` | OpenCode Zen API base URL |
| `OPENCODEZEN_MODEL` | `opencode-zen/legal-extractor` | OpenCode Zen model to use |
| `OPENCODEZEN_TIMEOUT` | `30` | Request timeout in seconds |

**Note**: New extractors can be added by implementing the interfaces in `src/core/interfaces.py` and registering them in `src/core/extractor_factory.py`.

---

**Testing library combination for paralegal date extraction use case** ⚖️
