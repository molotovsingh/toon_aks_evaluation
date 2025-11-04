# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Mantras

- **Start small, scale smart** - Land the simplest working version before expanding scope
- **Prove value fast** - Deliver end-user gains ahead of infrastructure rewrites
- **Design for the next plug** - Every new chunk should snap into the existing pipeline with minimal glue
- **Keep it boring** - Favor clear, well-understood patterns over clever abstractions
- **Own the blast radius** - Guardrails belong near the upload—fail fast and document why
- **Surface reality** - Log classification scores, extraction outcomes, and review warnings so humans stay in the loop
- **Respect the prompt contract** - One prompt (`LEGAL_EVENTS_PROMPT`), one schema; change it deliberately and update every consumer
- **Measure before you optimize** - Let data expose bottlenecks or costs before re-architecting
- **Leave breadcrumbs** - Docstrings, comments, and docs explain why a choice exists, not just how it works
- **Ship with toggles** - Feature flags and environment switches let us test without breaking the happy path
## Project Overview

This is a **proof-of-concept testing environment** for evaluating combinations of Docling (document processing) + pluggable event extractors (legal event extraction) for paralegal applications. The core pipeline: Documents In → Legal Events Out.

**Event Extractors Supported**:
- **LangExtract** (Gemini) - Default, uses Google's Gemini 2.0 Flash
- **OpenRouter** (Unified API) - 10 curated models from OpenAI, Anthropic, DeepSeek, Meta, Mistral (Oct 2025 testing)
- **OpenCode Zen** (Legal AI) - Specialized legal extraction models
- **OpenAI** (Direct API) - GPT-4o, GPT-4o-mini via OpenAI SDK
- **Anthropic** (Direct API) - Claude 3.5 Sonnet, Claude 3 Haiku via Anthropic SDK
- **DeepSeek** (Direct API) - DeepSeek-Chat via OpenAI-compatible API

**Provider Selection**: Use Streamlit UI dropdown (overrides environment) or set `EVENT_EXTRACTOR` environment variable.

**Architecture**: Registry pattern in `extractor_factory.py` - add new providers by implementing `EventExtractor` interface and registering in `EVENT_PROVIDER_REGISTRY`.

**Phase 1 Status (Week 3)**: 6 providers integrated (LangExtract, OpenRouter, OpenCode Zen, OpenAI, Anthropic, DeepSeek). Target: 8 providers by Week 4.

**Key Goal**: Test which parser+extractor combination can reliably extract legal events from various document types.

## Development Commands

### Environment Setup
```bash
# Install dependencies
uv sync

# Install Tesseract OCR (recommended - 3x faster than EasyOCR)
# macOS
brew install tesseract
export TESSDATA_PREFIX=/usr/local/opt/tesseract/share/tessdata

# OR Linux (Ubuntu/Debian)
sudo apt install tesseract-ocr libtesseract-dev
export TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

# Create environment file from template
cp .env.example .env
# Then edit .env with real API keys:
#   - GEMINI_API_KEY (required for LangExtract)
#   - OPENROUTER_API_KEY (optional, for OpenRouter)
#   - OPENCODEZEN_API_KEY (optional, for OpenCode Zen)
```

### Running Applications
```bash
# Main app (provider selector, supports LangExtract/OpenRouter/OpenCode Zen/OpenAI)
uv run streamlit run app.py

# Legacy examples for specific scenarios
uv run streamlit run examples/legal_events_app.py
uv run streamlit run examples/simple_legal_table_app.py
```

### Testing
```bash
# Complete test suite with detailed reporting
uv run python tests/run_all_tests.py

# Quick tests (skip performance benchmarks)
uv run python tests/run_all_tests.py --quick

# Individual test suites
uv run python -m pytest tests/test_acceptance_criteria.py -v
uv run python -m pytest tests/test_performance_integration.py -v

# Single test case (useful for debugging)
uv run python -m pytest tests/test_acceptance_criteria.py::AcceptanceCriteriaTests::test_docling_extraction -v
```

### Diagnostic Scripts
```bash
# Provider connectivity checks (quick validation)
uv run python scripts/check_langextract.py
uv run python scripts/check_openrouter.py
uv run python scripts/check_opencode_zen.py

# Comprehensive provider testing (10-level validation)
uv run python scripts/test_openrouter.py
uv run python scripts/test_opencode_zen.py

# Model comparison and benchmarking
uv run python scripts/test_all_models.py          # Side-by-side 5-model comparison
uv run python scripts/test_fallback_models.py     # 18 OpenRouter models test
uv run python scripts/test_opencode_zen_models.py # OpenCode Zen models test
uv run python scripts/test_deepseek.py            # DeepSeek R1 specific testing

# Verification utilities
uv run python scripts/verify_langextract_examples.py
uv run python scripts/probe_opencode_zen.py

# Cost Estimation & Tiktoken Integration (NEW)
uv run python scripts/test_tiktoken_integration.py  # Validate token counting accuracy
```

### Two-Stage Cost-Aware Model Selection

**NEW FEATURE (Nov 2025)**: The system now supports cost-aware model selection with exact token counting using tiktoken.

**Architecture**:
```
Stage 1 (FREE): Extract text with Docling
    ↓
Stage 2 (INFORMED): Calculate exact costs for ALL models using tiktoken
    ↓
User selects model based on cost/accuracy tradeoff
    ↓
Stage 3 (PAID): Run event extraction with selected model
```

**Key Components**:
- **`src/utils/token_counter.py`**: Exact token counting using OpenAI's tiktoken library
  - 19 models supported (GPT-4o, Claude, DeepSeek, Gemini, Llama, etc.)
  - Model→encoding mapping (o200k_base for GPT-4o, cl100k_base for others)
  - Token estimation for output prediction

- **`src/ui/cost_estimator.py`**: Enhanced cost calculation
  - `estimate_all_models_with_tiktoken()`: Precise cost table for all models
  - `estimate_all_models_with_heuristic()`: Fallback when tiktoken unavailable

- **`src/ui/cost_comparison.py`**: Interactive Streamlit UI
  - `show_cost_comparison_selector()`: Category-organized model selection by cost
  - `show_cost_comparison_table()`: Full comparison table view
  - `show_cost_breakdown()`: Detailed cost analysis for selected model

- **`src/ui/streamlit_common.py`**: Two-stage workflow integration
  - `show_cost_aware_model_selection()`: Complete two-stage flow in Streamlit

**Benefits**:
- **Cost Transparency**: Users see exact costs before making expensive API calls
- **Budget Flexibility**: Choose between free (Gemini), budget ($0.0002), or premium ($0.0079) models
- **Accuracy**: Tiktoken counts are 25-30% more accurate than character-based heuristic
- **No Hidden Costs**: Estimates vs actual billing variance <2% (vs ±20% with heuristic)

**Usage Example** (In Streamlit app):
```python
from src.ui.streamlit_common import show_cost_aware_model_selection

uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)
if uploaded_files:
    selected_model = show_cost_aware_model_selection(
        uploaded_files=uploaded_files,
        doc_extractor='docling'
    )
    if selected_model:
        st.success(f"Processing with {selected_model}")
        # Continue with extraction
```

**Cost Breakdown Example** (Famas PDF, 1,080 characters):
```
Model                    Tokens    Cost        Quality   Speed
────────────────────────────────────────────────────────────────
Gemini 2.5 Flash        386       $0.0000     9/10      4s
DeepSeek Chat           386       $0.0001     10/10     5s
Claude Haiku            351       $0.0002     10/10     4s
GPT-4o Mini             339       $0.0008     9/10      6s
Claude Sonnet 4.5       386       $0.0016     10/10     4s
Claude Opus 4           386       $0.0079     10/10     5s
```

**Testing** (Validation Results):
- ✅ Token counting: All 19 models supported with correct encoding detection
- ✅ Accuracy: Tiktoken counts 25-30% more accurate than character heuristic
- ✅ Pricing: Ranges from FREE (Gemini) to $0.0079 (Claude Opus) per document
- ✅ Cost variance: <2% difference between tiktoken estimate and actual API usage

## Core Architecture

### Pipeline Flow
```
Document Upload → DoclingAdapter → Text Extraction → EventExtractor → Five-Column Table
                                                            ↓
                                                    Provider Registry
                                                    (langextract, openrouter, etc.)
```

### Protocol-Based Design
The system uses **Protocol interfaces** (PEP 544) for swappable components:

- **`DocumentExtractor`** (`src/core/interfaces.py`) - Returns `ExtractedDocument(markdown, plain_text, metadata)`
- **`EventExtractor`** (`src/core/interfaces.py`) - Returns `List[EventRecord]` with five-column schema

### Provider Registry Pattern
New extractors are added via **registry pattern** in `extractor_factory.py`:

```python
EVENT_PROVIDER_REGISTRY: Dict[str, Callable] = {
    "langextract": _create_langextract_event_extractor,
    "openrouter": _create_openrouter_event_extractor,
    "opencode_zen": _create_opencode_zen_event_extractor,
    "openai": _create_openai_event_extractor,
}
```

To add a new provider:
1. Create adapter implementing `EventExtractor` protocol
2. Add factory function to registry
3. Add config dataclass to `config.py`
4. Update `load_provider_config()` to handle new provider type

### Document Extractor Catalog

The system uses a **centralized catalog** for Layer 1 document extraction configuration in `document_extractor_catalog.py`:

**Purpose**: Single source of truth for document extractor metadata (pricing, capabilities, prompts, availability)

**Key Features**:
- **Pricing Metadata**: Cost per page, display strings for UI cost estimation
- **Capability Flags**: PDF/DOCX/image support, vision capabilities, OCR quality
- **Prompt Management**: Named prompts in `doc_extractor_prompts.py` with inline override support
- **Enabled Toggle**: Control extractor availability without code changes
- **Dynamic UI/CLI**: Options auto-generated from `catalog.list_extractors(enabled=True)`

**Catalog Entry Schema** (`DocExtractorEntry`):
```python
DocExtractorEntry(
    # === Identification ===
    extractor_id="qwen_vl",           # Unique identifier
    display_name="Qwen3-VL (Budget Vision)",
    provider="openrouter",            # 'local', 'openrouter', 'google'

    # === Pricing (Layer 1) ===
    cost_per_page=0.00512,            # USD per page/image
    cost_display="$0.077 per 15-page doc",

    # === Capabilities ===
    supports_pdf=True,
    supports_vision=True,             # Multimodal understanding
    processing_speed="medium",        # 'fast', 'medium', 'slow'
    ocr_quality="high",               # 'high', 'medium', 'low', 'n/a'

    # === Registry Control ===
    enabled=True,                     # Toggle availability in UI/CLI
    prompt_id="qwen_vl_doc",         # Reference to doc_extractor_prompts.QWEN_VL_DOC_PROMPT
    prompt_override=None,             # Inline prompt (overrides prompt_id)
    factory_callable="src.core.extractor_factory._create_qwen_vl_document_extractor",  # Factory function reference

    # === Metadata ===
    recommended=False,
    notes="Budget vision API for multimodal parsing. Use when Docling OCR fails.",
    documentation_url="https://openrouter.ai/models/qwen/qwen3-vl-8b-instruct"
)
```

**Prompt Resolution Priority**:
1. `prompt_override` (inline override if specified)
2. `prompt_id` (lookup in `doc_extractor_prompts.py`)
3. `None` (extractor uses default behavior)

**Usage Examples**:
```python
# Get catalog instance
from src.core.document_extractor_catalog import get_doc_extractor_catalog
catalog = get_doc_extractor_catalog()

# Query extractors
enabled_extractors = catalog.list_extractors(enabled=True)
vision_extractors = catalog.list_extractors(supports_vision=True)
free_extractors = catalog.list_extractors(free_only=True)

# Get pricing for cost estimation
pricing = catalog.get_pricing('qwen_vl')
# Returns: {'cost_per_page': 0.00512, 'cost_display': '$0.077 per 15-page doc'}

# Estimate extraction cost
cost_estimate = catalog.estimate_cost('qwen_vl', page_count=15)
# Returns: {'cost_usd': 0.0768, 'cost_display': '$0.0768', ...}

# Get extractor prompt
prompt = catalog.get_prompt('qwen_vl')
# Returns: QWEN_VL_DOC_PROMPT from doc_extractor_prompts.py
```

**Factory Integration** (Dynamic Bootstrapping):
- **DOC_PROVIDER_REGISTRY** is dynamically built from catalog at module load time
- Factory imports factory functions from `factory_callable` string references
- Whitelist validation: Only `src.core.*` imports allowed (security)
- Graceful degradation: Invalid entries are logged and skipped
- Enabled flag enforced: Disabled extractors excluded from registry
- Prompts auto-injected from catalog into adapter constructors
- Example: `Qwen3VLDocumentExtractor(api_key=..., prompt=catalog.get_prompt('qwen_vl'))`

**Adding New Document Extractors**:
1. **Create adapter** in `src/core/` implementing `DocumentExtractor` protocol
2. **Add factory function** to `extractor_factory.py`:
   ```python
   def _create_my_new_extractor(doc_config, _event_config, _extractor_config):
       return MyNewExtractor(doc_config)
   ```
3. **Add catalog entry** to `_DOC_EXTRACTOR_REGISTRY` in `document_extractor_catalog.py`:
   ```python
   DocExtractorEntry(
       extractor_id="my_new_extractor",
       display_name="My New Extractor",
       provider="openrouter",  # or "local", "google", etc.
       cost_per_page=0.01,
       cost_display="$0.15 per 15-page doc",
       enabled=True,  # Toggle to disable
       factory_callable="src.core.extractor_factory._create_my_new_extractor",  # NEW: Factory reference
       prompt_id="my_extractor_prompt",  # Optional: If using custom prompt
       # ... other metadata
   )
   ```
4. **Optional: Add custom prompt** to `doc_extractor_prompts.py` if using vision model
5. **Restart app** → UI/CLI automatically detect new extractor (no code changes needed)

**Disabling Extractors**:
- Set `enabled=False` in catalog entry → Extractor disappears from UI, CLI, and factory registry
- No need to comment out code or edit factory map
- Change takes effect on app restart

**Current Extractors**:
- **docling** (Local OCR): FREE, fast, production-ready, recommended for most documents
- **qwen_vl** (Budget Vision): $0.00512/page, multimodal, fallback for poor quality scans

### Event Extractor Catalog

The system uses a **centralized catalog** for Layer 2 event extraction configuration in `event_extractor_catalog.py`:

**Purpose**: Single source of truth for event extractor provider metadata (capabilities, prompts, availability)

**Key Features**:
- **Capability Flags**: Runtime model support, single vs multi-model providers
- **Prompt Management**: Named prompts with inline override support
- **Enabled Toggle**: Control provider availability without code changes
- **Dynamic Factory**: EVENT_PROVIDER_REGISTRY auto-built from `catalog.list_extractors(enabled=True)`
- **Security**: Whitelist validation (`src.core.*` imports only)

**Catalog Entry Schema** (`EventExtractorEntry`):
```python
EventExtractorEntry(
    # === Identification ===
    provider_id="openrouter",           # Unique identifier
    display_name="OpenRouter",

    # === Registry Control ===
    enabled=True,                       # Toggle availability in UI/CLI/factory
    factory_callable="src.core.extractor_factory._create_openrouter_event_extractor",  # Factory function reference
    prompt_id=None,                     # Reference to named prompt (optional)
    prompt_override=None,               # Inline prompt (overrides prompt_id)

    # === Capabilities ===
    supports_runtime_model=True,        # Multi-model selection support

    # === Metadata ===
    recommended=True,
    notes="Unified API for 10+ curated models. Best for A/B testing.",
    documentation_url="https://openrouter.ai/docs"
)
```

**Usage Examples**:
```python
# Get catalog instance
from src.core.event_extractor_catalog import get_event_extractor_catalog
catalog = get_event_extractor_catalog()

# Query extractors
enabled_providers = catalog.list_extractors(enabled=True)
multi_model_providers = catalog.list_extractors(supports_runtime_model=True)
recommended_providers = catalog.list_extractors(recommended_only=True)

# Validate provider ID
is_valid = catalog.validate_provider_id('openrouter')  # Returns: True

# Get provider prompt (if configured)
prompt = catalog.get_prompt('openrouter')  # Returns: None (uses LEGAL_EVENTS_PROMPT default)
```

**Factory Integration** (Dynamic Bootstrapping):
- **EVENT_PROVIDER_REGISTRY** is dynamically built from catalog at module load time
- Factory imports factory functions from `factory_callable` string references
- Whitelist validation: Only `src.core.*` imports allowed (security)
- Graceful degradation: Invalid entries are logged and skipped
- Enabled flag enforced: Disabled providers excluded from registry
- LangExtract fallback: Always available as safe default

**Adding New Event Extractors**:
1. **Create adapter** in `src/core/` implementing `EventExtractor` protocol:
   ```python
   class MyProviderAdapter:
       def extract_events(self, text: str, metadata: Dict) -> List[EventRecord]:
           # Return EventRecord with five-column schema

       def is_available(self) -> bool:
           # Check API key exists
   ```

2. **Add factory function** to `extractor_factory.py`:
   ```python
   def _create_myprovider_event_extractor(doc_config, event_config, extractor_config):
       return MyProviderAdapter(event_config)
   ```

3. **Add catalog entry** to `_EVENT_EXTRACTOR_REGISTRY` in `event_extractor_catalog.py`:
   ```python
   EventExtractorEntry(
       provider_id="myprovider",
       display_name="My Provider",
       enabled=False,  # Start disabled for testing
       factory_callable="src.core.extractor_factory._create_myprovider_event_extractor",
       supports_runtime_model=False,  # True if provider supports model selection
       notes="My custom event extraction provider"
   )
   ```

4. **Add config class** to `config.py`:
   ```python
   @dataclass
   class MyProviderConfig:
       api_key: str = field(default_factory=lambda: env_str("MYPROVIDER_API_KEY", ""))
       model: str = field(default_factory=lambda: env_str("MYPROVIDER_MODEL", "default-model"))
   ```

5. **Update** `load_provider_config()` in `config.py` to handle new provider

6. **Test with enabled=False** → Verify factory skips, UI hides provider

7. **Set enabled=True** → Restart app → Provider appears everywhere automatically

**Disabling Providers**:
- Set `enabled=False` in catalog entry → Provider disappears from factory/UI/CLI
- No need to comment out code or edit factory map
- Change takes effect on app restart

**Current Providers**:
- **langextract** (Gemini): Default, recommended, supports runtime model override
- **openrouter** (Unified API): Recommended, 10+ curated models, runtime model selection
- **openai** (Direct API): GPT-4o-mini/GPT-4o/GPT-5, runtime model selection
- **anthropic** (Direct API): Claude 3 Haiku/Sonnet 4.5/Opus 4, runtime model selection
- **deepseek** (Direct API): DeepSeek-Chat, single model
- **opencode_zen** (Legal AI): Specialized legal extraction, single model

### The Prompt Contract
**Critical**: `LEGAL_EVENTS_PROMPT` in `src/core/constants.py` defines the extraction schema. All providers must return this exact JSON structure:

```json
{
  "event_particulars": "2-8 sentence description with legal context",
  "citation": "Legal reference or empty string (NO hallucinations)",
  "document_reference": "Source filename (auto-populated)",
  "date": "Specific date or empty string"
}
```

These map to the **Five-Column Table**:
1. No (sequence number)
2. Date
3. Event Particulars
4. Citation
5. Document Reference

**When modifying the prompt**:
- Update `LEGAL_EVENTS_PROMPT` in `constants.py` (single source of truth)
- All adapters use this same prompt via import
- Test with all providers before committing changes

#### Prompt Versions and Rollback

The system supports **two prompt versions** with instant rollback capability:

**V1 (Baseline)**: Current production prompt
**V2 (Enhanced)**: Improved granularity and recall (2025-10-11)
- Date granularity rule: Separate event per distinct dated action
- Interim milestones guidance: Inspections, negotiations, status updates
- Limited details handling: Create events even with partial info
- Refined flexibility: 1-8 sentences as appropriate

**Feature Flag Toggle**:
```bash
# Use V1 (default - baseline)
# (no environment variable needed)

# Use V2 (enhanced prompt)
USE_ENHANCED_PROMPT=true
```

**A/B Testing**:
```bash
# Test baseline prompt
uv run python scripts/test_openrouter.py

# Test enhanced prompt
USE_ENHANCED_PROMPT=true uv run python scripts/test_openrouter.py
```

**Rollback Options**:
1. **Instant**: Remove `USE_ENHANCED_PROMPT` from `.env` (reverts to V1)
2. **Git**: `git checkout HEAD -- src/core/constants.py`
3. **Emergency**: Both prompts in code - toggle is non-destructive

**Expected Improvements** (based on 2025-10-03 benchmarks):
- Gemini: 83 events → 40-50 (reduced noise)
- Anthropic: 2 events → 6-10 (better recall)
- OpenRouter/OpenAI: 4-6 events → 8-12 (more complete)
- Variance: 41x → 10x (improved consistency)

### Configuration System

#### Provider Selection
- `EVENT_EXTRACTOR`: Choose provider (`langextract`|`openrouter`|`opencode_zen`|`openai`|`anthropic`|`deepseek`)
- Streamlit UI selector overrides environment variable
- Each provider requires provider-specific API key (see `.env.example`)

#### Key Environment Variables
| Variable | Default | Purpose |
|----------|---------|---------|
| `GEMINI_API_KEY` | _(required for LangExtract)_ | Google Gemini access |
| `GEMINI_MODEL_ID` | `gemini-2.0-flash` | Override default model |
| `OPENROUTER_API_KEY` | _(required for OpenRouter)_ | Multi-provider unified API |
| `OPENROUTER_MODEL` | `openai/gpt-4o-mini` | Budget: `deepseek/deepseek-r1-distill-llama-70b` ($0.03/M) |
| `OPENAI_API_KEY` | _(required for OpenAI)_ | Direct OpenAI API access |
| `OPENAI_MODEL` | `gpt-4o-mini` | GPT-4o or GPT-4o-mini |
| `ANTHROPIC_API_KEY` | _(required for Anthropic)_ | Direct Anthropic API access |
| `ANTHROPIC_MODEL` | `claude-3-haiku-20240307` | Claude 3.5 or Claude 3 models |
| `DEEPSEEK_API_KEY` | _(required for DeepSeek)_ | Direct DeepSeek API access |
| `DEEPSEEK_MODEL` | `deepseek-chat` | DeepSeek-Chat model |
| `DOCLING_DO_OCR` | `false` | Enable/disable OCR (auto-detects scanned PDFs for 16x speedup on digital docs) |
| `DOCLING_TABLE_MODE` | `FAST` | `FAST` or `ACCURATE` |
| `DOCLING_ACCELERATOR_DEVICE` | `cpu` | `cpu`, `cuda`, or `mps` |

**Curated Model Recommendations** (Oct 2025 testing):
- **Recommended starting point**: `openai/gpt-4o-mini` ($0.15/M, 9/10 quality, 128K context)
- **Budget champion**: `deepseek/deepseek-r1-distill-llama-70b` ($0.03/M, 10/10 quality, 128K) - 50x cheaper!
- **Ultra-budget option**: `qwen/qwq-32b` ($0.115/M, 7/10 quality, 128K) - ⚠️ May miss events on complex docs
- **Speed champion**: `anthropic/claude-3-haiku` ($0.25/M, 10/10 quality, 200K, 4.4s extraction)
- **Long documents (50+ pages)**: `anthropic/claude-3-5-sonnet` ($3/M, 10/10 quality, 200K context)
- **Open source option**: `meta-llama/llama-3.3-70b-instruct` ($0.60/M, 10/10 quality, 128K)
- **Privacy/sovereignty hedge**: `openai/gpt-oss-120b` ($0.31/M, 10/10 quality, 128K, Apache 2.0 licensed)
  - Strategic value: Insurance policy against vendor lock-in and future privacy requirements
  - Can self-host if privacy/sovereignty concerns emerge (unlike proprietary APIs)
  - Not dependent on Chinese AI (DeepSeek) or changing US corporate policies (OpenAI)
  - Cost: $0.28/M premium over DeepSeek for control and optionality
  - Use case: Production systems that may need private deployment in future

Total: 11 battle-tested models (9-10/10 quality, plus 1 budget option at 7/10). **Excluded**: All Gemini variants, Cohere, Perplexity (failed JSON mode tests).

**Oct 5, 2025 Adapter Fix** - Conditional JSON Mode Support:
- Fixed OpenRouter adapter to support prompt-based JSON (not just native `response_format`)
- Added markdown wrapper stripping for OSS model compatibility
- Result: Qwen QwQ 32B now works (7/10 quality, ultra-cheap at $0.115/M)

**Additional exclusions** (Oct 5, 2025 OSS model testing):
- Mistral Small 3.1: Weak real-doc extraction (7/10 quality, extracted 1 event vs ≥3 expected)

**Oct 13, 2025 DISCOVERY - GPT-OSS-120B WORKS**:
- ✅ **openai/gpt-oss-120b** passes all tests (10/10 quality, prompt-based JSON)
  - Oct 5 test failed due to test script unconditionally adding `response_format` parameter
  - Production adapter conditionally adds `response_format` only for compatible models
  - GPT-OSS-120B succeeds via prompt-based JSON (no native JSON mode needed)
  - Apache 2.0 licensed, 117B MoE architecture, $0.31/M blended cost
  - **Added to curated model list** in Open Source / EU Hosting category
- ❌ **openai/gpt-oss-20b** returns empty responses (smaller variant fundamentally broken, 1/10)

**Key insights**:
1. Native JSON mode (`response_format`) is NOT mandatory - many models work via prompts
2. Fixed adapter unlocked Qwen QwQ 32B ($0.115/M, exceptional cost-effectiveness)
3. Real legal document testing remains essential (Qwen QwQ passes synthetic tests but weak on complex docs)

See `scripts/test_new_oss_models.py` for methodology and `src/core/openrouter_adapter.py:17-31` for JSON mode compatibility list.

#### Runtime Model Override (Ground Truth Model Selection)

The system supports **runtime model selection** for all providers through the `runtime_model` parameter, enabling ground truth dataset creation with premium models.

**Architecture Pattern**:
- **UI Layer** (`app.py`): Model selector dropdowns pass `runtime_model` to pipeline
- **Config Layer** (`config.py:load_provider_config()`): Applies `runtime_model` override to provider config
- **Adapter Layer**: Each adapter uses `config.model` (or `config.model_id` for Gemini) from config object

**Provider-Specific Implementation**:
```python
# OpenAI/Anthropic/DeepSeek pattern
if runtime_model:
    event_config.model = runtime_model

# LangExtract/Gemini pattern (different field name)
if runtime_model:
    event_config.model_id = runtime_model

# OpenRouter pattern (special property)
if runtime_model:
    event_config.runtime_model = runtime_model  # Uses @property active_model
```

**Ground Truth Models**:
- **Tier 1 (Recommended)**: Claude Sonnet 4.5 (`claude-sonnet-4-5`) - $3/M, best balance
- **Tier 2 (Alternative)**: GPT-5 (`gpt-5`), Gemini 2.5 Pro (`gemini-2.5-pro`) - pending pricing
- **Tier 3 (Validation)**: Claude Opus 4 (`claude-opus-4`) - $15/M, highest quality

**Model Selectors in UI** (`app.py:249-384`):
- `create_anthropic_model_selector()` - Claude Sonnet 4.5, Opus 4, plus production models
- `create_openai_model_selector()` - GPT-5, GPT-4o, GPT-4o-mini
- `create_langextract_model_selector()` - Gemini 2.5 Pro, 2.0 Flash

**How It Works**:
1. User selects provider → Provider-specific model selector appears
2. User selects model → Dropdown returns model identifier string
3. `create_provider_selection()` returns `(provider, selected_model)` tuple
4. Pipeline call: `process_documents_with_spinner(provider=..., runtime_model=...)`
5. Factory creates adapter with overridden model via `load_provider_config(provider, runtime_model=...)`

**Configuration Precedence** (highest to lowest):
1. UI `runtime_model` parameter (user selection)
2. Environment variable (`OPENAI_MODEL`, `ANTHROPIC_MODEL`, `GEMINI_MODEL_ID`)
3. Config dataclass default (defined in `config.py`)

**Use Cases**:
- **Ground Truth Creation**: Process documents with premium models to create reference datasets
- **A/B Testing**: Compare production vs premium model outputs on same documents
- **Quality Validation**: Verify production model extractions against ground truth
- **Cost Optimization**: Test if cheaper models match ground truth quality

**Example Workflow**:
```bash
# 1. Create ground truth with Claude Sonnet 4.5
uv run streamlit run app.py
# Select "Anthropic" → "Claude Sonnet 4.5" → Process docs → Export as ground_truth.csv

# 2. Test production model against ground truth
# Select "Anthropic" → "Claude 3 Haiku" → Process same docs → Export as production.csv

# 3. Compare outputs
uv run python scripts/compare_extractions.py ground_truth.csv production.csv
```

**Implementation Files**:
- Model constants: `src/core/constants.py:74-99`
- Config override logic: `src/core/config.py:224-277`
- UI selectors: `app.py:249-384` and `app.py:602-663`
- Adapter implementations: `src/core/{openai,anthropic,langextract}_adapter.py`

## Development Guidelines

### Core Principles
1. **Proof-of-Concept Focus**: Stay within "documents in → legal events out" scope
2. **No Fake Extractors**: Never mock extractor success without real API calls - if a provider fails, report it clearly
3. **Respect Configuration**: Honor Streamlit toggles and environment variables
4. **Guard Sample Data**: Use small test snippets, avoid large synthetic documents

### Thinking Mode
- Always use deep reasoning for complex problems
- For architectural decisions, performance optimization, or unfamiliar codebase analysis, apply maximum thinking budget
- When uncertain about approach, think through alternatives thoroughly

### Test Data Guidelines
**Real legal documents are already in the repo** - use existing files for testing:

#### Case-Based Test Files (Organized by Legal Matter)

- **`sample_pdf/amrapali_case/`** (34MB, 9 PDFs)
  - Real estate transaction in India with buyer agreements, bank statements, affidavits
  - Large documents (one is 17MB) - good for stress testing
  - Use `Amrapali Allotment Letter.pdf` (1.4MB) for quick real-world tests

- **`sample_pdf/famas_dispute/`** (2.3MB, 7 files: 2 PDF + 4 EML + 1 DOCX)
  - International arbitration case (Famas GmbH vs Elcomponics Sales)
  - **Best for manual quality evaluation**: `Answer to Request for Arbitration.pdf` (930KB, ~15 pages)
  - Mix of document formats - tests parser versatility
  - Recommended for Phase 2 provider comparison testing

- **`tests/test_documents/`** (28KB, 6 files)
  - Synthetic HTML files with edge cases (ambiguous dates, multiple events, no dates)
  - Small PDF for quick unit tests
  - Use for automated testing, not manual review

**When to use what**:
- Quick adapter test → `sample_pdf/famas_dispute/Answer to Request for Arbitration.pdf` (~15 pages, real legal events)
- Provider comparison → Same Famas arbitration PDF (medium complexity, not overwhelming)
- Stress test → `sample_pdf/amrapali_case/Amrapali Builder Buyer Agreement.pdf` (17MB)
- Edge case validation → `tests/test_documents/*.html` files

All sample files are tracked in git for reproducible testing across environments.

### Testing Legal Events Extraction
When testing LangExtract integration, use this minimal example from the README:

```python
import os
from dotenv import load_dotenv
import langextract as lx

load_dotenv()

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
```

### File Structure Understanding
- **Root level**: Multiple Streamlit apps for different testing scenarios
- **`src/core/`**: Core pipeline logic, interfaces, and configuration
- **`src/extractors/`**: Individual extractor implementations (legacy - do not add new extractors here)
- **`src/ui/`**: Shared Streamlit UI components
- **`src/utils/`**: File handling utilities
- **`tests/`**: Acceptance criteria and performance tests
- **`docs/`**: Design documents, architecture decisions, and project orders
- **`scripts/`**: Development utilities and troubleshooting guides

### Adding New Event Extractors
1. Create adapter in `src/core/` (not `src/extractors/` - that's legacy) implementing `EventExtractor` protocol:
   ```python
   class MyProviderAdapter:
       def extract_events(self, text: str, metadata: Dict) -> List[EventRecord]:
           # Must return EventRecord with five-column schema

       def is_available(self) -> bool:
           # Check API key exists
   ```

2. Add factory function and register in `extractor_factory.py`:
   ```python
   def _create_myprovider_event_extractor(doc_config, event_config, extractor_config):
       return MyProviderAdapter(event_config)

   EVENT_PROVIDER_REGISTRY["myprovider"] = _create_myprovider_event_extractor
   ```

3. Add config class to `config.py` and update `load_provider_config()`

4. Test with real legal PDF using diagnostic script pattern (see `scripts/test_openrouter.py`)

This architecture enables A/B testing, gradual migrations, and vendor flexibility without changing core application logic.

## Key Files Reference

### Critical Architecture Files
| File | Purpose |
|------|---------|
| `src/core/constants.py` | **LEGAL_EVENTS_PROMPT** (single source of truth for extraction schema) |
| `src/core/interfaces.py` | Protocol definitions for DocumentExtractor and EventExtractor |
| `src/core/extractor_factory.py` | **EVENT_PROVIDER_REGISTRY** (add new providers here) |
| `src/core/config.py` | Configuration dataclasses and environment loading |
| `src/core/legal_pipeline_refactored.py` | Main pipeline orchestration |

### Adapter Implementations
- `src/core/docling_adapter.py` - Document text extraction (PDF/DOCX/HTML/PPTX)
- `src/core/langextract_adapter.py` - Gemini 2.0 Flash event extraction
- `src/core/openrouter_adapter.py` - Multi-provider unified API (11+ models)
- `src/core/opencode_zen_adapter.py` - Legal AI specialized extraction
- `src/core/openai_adapter.py` - Direct OpenAI API integration
- `src/core/anthropic_adapter.py` - Direct Anthropic API integration
- `src/core/deepseek_adapter.py` - Direct DeepSeek API integration (OpenAI-compatible)

### Testing Infrastructure
- `tests/run_all_tests.py` - Master test suite runner with reporting
- `tests/test_acceptance_criteria.py` - Core functionality validation
- `tests/test_performance_integration.py` - Performance benchmarks
- `scripts/test_fallback_models.py` - Provider comparison framework (18 models tested)
- when planning try to be updated on api and library documentation by researching the internet