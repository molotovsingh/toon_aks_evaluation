# Model Catalog v2 Architecture Design

**Order ID**: `model-catalog-v2-architecture-001`
**Version**: v2.0
**Priority**: HIGH
**Date**: 2025-10-14
**Author**: Claude Code (Automated Analysis)

## Executive Summary

This document proposes a centralized model registry to replace the current scattered model definitions across `app.py`, adapter configs, constants, and documentation. The registry will provide a single source of truth for model metadata, pricing, capabilities, and runtime resolution logic.

**Key Problems Solved**:
1. **Metadata accuracy** - `provider_model` logs wrong model when runtime override used (Issue #1)
2. **Code duplication** - Model lists defined in 3+ locations (app.py:214-264, constants.py:131-154, docs)
3. **Capability drift** - JSON mode support hardcoded in openrouter_adapter.py:19-33, duplicating knowledge
4. **Cost estimation blockers** - No central pricing source for upcoming cost-estimator-001 feature

**Recommended Approach**: Python module registry (not YAML/JSON) for type safety, IDE support, and zero serialization overhead.

---

## 1. Current State Audit

### 1.1 Model List Locations (Scattered Definitions)

| Location | Purpose | Models Defined | Format |
|----------|---------|----------------|--------|
| `app.py:214-264` | **UI Model Catalog** | 10 models (Anthropic, OpenAI, Google, OpenRouter, DeepSeek) | Python `ModelConfig` dataclass |
| `src/core/constants.py:131-154` | **Named Constants** | 13 model IDs (ground truth references) | String constants |
| `src/core/config.py:89-175` | **Provider Defaults** | 6 default models (one per provider) | Dataclass defaults + env vars |
| `src/core/openrouter_adapter.py:19-33` | **JSON Mode Compatibility** | 12 models (capability flag) | List of strings |
| `CLAUDE.md:91-115` | **Model Recommendations** | 11 curated models (user documentation) | Markdown table |

**Duplication Evidence**:
- `claude-sonnet-4-5` appears in: app.py, constants.py, CLAUDE.md (3 copies of metadata)
- `gpt-4o-mini` pricing: `$0.15/M` in app.py, but no pricing in constants.py
- `deepseek-r1-distill-llama-70b`: Listed in CLAUDE.md but not in constants.py or app.py catalog

**Impact**: Adding a new model requires changes to 4-5 files, risking inconsistency.

### 1.2 Runtime Model Flow (Correct Architecture)

**Current Flow** (app.py → config.py → adapters → metadata):
```python
# 1. UI Selection (app.py:1149-1260)
selected_provider, selected_model = create_provider_selection()
# User selects: provider='openrouter', model='anthropic/claude-3-haiku'

# 2. Pipeline Creation (streamlit_common.py)
pipeline = LegalEventsPipeline(
    event_extractor=provider,      # 'openrouter'
    runtime_model=selected_model    # 'anthropic/claude-3-haiku'
)

# 3. Config Override (config.py:224-277)
def load_provider_config(provider, runtime_model=None):
    if provider == "openrouter":
        event_config = OpenRouterConfig()
        if runtime_model:
            event_config.runtime_model = runtime_model  # ✅ Stored

# 4. Adapter Access (openrouter_adapter.py:113, 171, 272)
provider_model = self.config.active_model  # ✅ Returns runtime_model via @property

# 5. Metadata Capture (pipeline_metadata.py:186-210)
if hasattr(extractor.config, 'active_model'):
    provider_model = extractor.config.active_model  # ✅ Correct strategy
```

**Verdict**: Runtime model flow is **ALREADY CORRECT** after Oct 2025 fixes.

**Known Issue**: Metadata capture has fallback to env vars (pipeline_metadata.py:212-225) which **should never trigger** if adapters properly expose config. The fallback is defensive but indicates potential adapter inconsistencies.

### 1.3 Capability Flags (Hardcoded Knowledge)

**JSON Mode Support** (openrouter_adapter.py:19-33):
```python
JSON_MODE_COMPATIBLE_MODELS = [
    "openai/gpt-4o",
    "anthropic/claude-3",
    "deepseek/deepseek-chat",
    "meta-llama/llama-3.3-70b-instruct",
    "mistralai/mistral-large",
    # ... 12 models total
]
```

**Responses API Requirement** (no current tracking):
- GPT-5 models require Responses API
- Currently handled implicitly in openai_adapter.py
- No centralized flag → future models may be misconfigured

**Impact**: Adding new models requires editing adapter code, not just data updates.

### 1.4 Environment Variable Mapping

**Precedence Chain** (config.py):
```python
# Example: OpenRouterConfig
model: str = field(
    default_factory=lambda: env_str("OPENROUTER_MODEL", "openai/gpt-4o-mini")
)
# Fallback: openai/gpt-4o-mini (hardcoded in dataclass)

# Runtime override (highest priority):
if runtime_model:
    event_config.runtime_model = runtime_model
```

**Effective Precedence** (highest to lowest):
1. `runtime_model` parameter (UI selection)
2. Environment variable (e.g., `OPENROUTER_MODEL`)
3. Dataclass default (e.g., `openai/gpt-4o-mini`)

**Gap**: No validation that env var values correspond to real models in catalog.

---

## 2. Proposed Model Catalog Schema

### 2.1 Storage Format Decision

**Recommendation**: **Python module** (`src/core/model_catalog.py`)

**Rationale**:
- **Type safety**: Dataclasses with strict typing catch errors at import time
- **IDE support**: Autocomplete, find usages, refactoring tools work seamlessly
- **Zero overhead**: No YAML parsing, JSON deserialization, or schema validation needed
- **Versioning**: Changes tracked in git with full context (unlike data files)
- **Consistency**: Matches existing patterns (app.py ModelConfig dataclass)

**Rejected Alternatives**:
- **YAML/JSON**: Requires schema validation, loses type safety, adds serialization overhead
- **Database**: Over-engineering for ~30 models, adds deployment complexity
- **Hybrid (code + data)**: Splits truth across formats, harder to maintain

### 2.2 Schema Definition

```python
# src/core/model_catalog.py

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class ModelStatus(Enum):
    """Model availability status"""
    STABLE = "stable"           # Production-ready, tested
    EXPERIMENTAL = "experimental"  # Working but unvalidated
    DEPRECATED = "deprecated"   # Scheduled for removal
    PLACEHOLDER = "placeholder" # Future release (e.g., GPT-5)

class ModelTier(Enum):
    """Ground truth quality tier"""
    TIER_1 = "tier_1"  # Recommended (Claude Sonnet 4.5)
    TIER_2 = "tier_2"  # Alternative (GPT-5, Gemini 2.5 Pro)
    TIER_3 = "tier_3"  # Validation (Claude Opus 4)
    PRODUCTION = "production"  # Standard production models

@dataclass
class ModelEntry:
    """
    Complete model metadata entry

    Single source of truth for all model properties across:
    - UI display and selection
    - Adapter configuration
    - Cost estimation
    - Capability routing
    - Metadata logging
    """

    # === Identification ===
    provider: str              # 'anthropic', 'openai', 'google', 'openrouter', 'deepseek'
    model_id: str             # Backend identifier: 'claude-sonnet-4-5', 'openai/gpt-4o-mini'
    display_name: str         # Human-readable: "Claude Sonnet 4.5", "GPT-4o Mini"

    # === Classification ===
    tier: ModelTier           # Ground truth tier or PRODUCTION
    category: str             # "Ground Truth", "Production", "Budget", "Long Documents"
    status: ModelStatus       # STABLE, EXPERIMENTAL, DEPRECATED, PLACEHOLDER

    # === Pricing (per million tokens) ===
    cost_input_per_1m: Optional[float]   # Input tokens cost in USD (None = free/unknown)
    cost_output_per_1m: Optional[float]  # Output tokens cost in USD
    cost_display: str                    # Human display: "$3/M", "Free", "$TBD"

    # === Capabilities ===
    context_window: int       # Tokens: 128000, 200000, 2000000
    context_display: str      # Human display: "128K", "200K", "2M"
    supports_json_mode: bool  # Native JSON mode (response_format parameter)
    requires_responses_api: bool  # Requires Responses API (GPT-5 models)
    supports_vision: bool     # Multimodal vision (Gemini, GPT-4o)
    max_tokens_output: Optional[int]  # Maximum output tokens (if limited)

    # === Quality Metrics ===
    quality_score: Optional[str]  # "10/10", "9/10", "7/10" (from Oct 2025 testing)
    extraction_speed_seconds: Optional[float]  # Average extraction time (e.g., 4.4 for Haiku)

    # === UI Metadata ===
    badges: List[str]         # ["Tier 1", "Fastest", "Cheapest", "Self-hostable"]
    recommended: bool         # Show in "Recommended" section
    documentation_url: Optional[str]  # Provider model docs link

    # === Notes ===
    notes: Optional[str]      # Internal notes (testing quirks, limitations)
```

### 2.3 Helper API Design

```python
# src/core/model_catalog.py (continued)

class ModelCatalog:
    """
    Model registry with query and validation utilities

    Usage:
        from src.core.model_catalog import catalog

        # Lookup
        model = catalog.get_model("anthropic/claude-3-haiku")

        # Query
        budget_models = catalog.list_models(category="Budget", status=ModelStatus.STABLE)

        # Validation
        if not catalog.validate_model_id(user_input):
            raise ValueError("Invalid model")
    """

    def __init__(self, models: List[ModelEntry]):
        self._models = {m.model_id: m for m in models}
        self._by_provider = self._index_by_provider(models)

    def get_model(self, model_id: str) -> Optional[ModelEntry]:
        """Get model entry by ID (returns None if not found)"""
        return self._models.get(model_id)

    def list_models(
        self,
        provider: Optional[str] = None,
        category: Optional[str] = None,
        tier: Optional[ModelTier] = None,
        status: Optional[ModelStatus] = None,
        min_context: Optional[int] = None
    ) -> List[ModelEntry]:
        """Query models with filters"""
        results = list(self._models.values())

        if provider:
            results = [m for m in results if m.provider == provider]
        if category:
            results = [m for m in results if m.category == category]
        if tier:
            results = [m for m in results if m.tier == tier]
        if status:
            results = [m for m in results if m.status == status]
        if min_context:
            results = [m for m in results if m.context_window >= min_context]

        return results

    def get_capabilities(self, model_id: str) -> Dict[str, bool]:
        """Get model capability flags"""
        model = self.get_model(model_id)
        if not model:
            return {}

        return {
            'supports_json_mode': model.supports_json_mode,
            'requires_responses_api': model.requires_responses_api,
            'supports_vision': model.supports_vision
        }

    def resolve_runtime_model(
        self,
        provider: str,
        runtime_model: Optional[str],
        env_defaults: Dict[str, str]
    ) -> str:
        """
        Resolve final model using precedence chain

        Precedence:
        1. runtime_model (UI selection)
        2. env_defaults[provider] (environment variable)
        3. First STABLE model for provider
        """
        # Priority 1: Runtime override
        if runtime_model:
            if self.validate_model_id(runtime_model):
                return runtime_model
            else:
                # Log warning but proceed (graceful degradation)
                logger.warning(f"Runtime model '{runtime_model}' not in catalog, using anyway")
                return runtime_model

        # Priority 2: Environment variable
        env_model = env_defaults.get(provider)
        if env_model:
            return env_model

        # Priority 3: First stable model for provider
        stable_models = self.list_models(provider=provider, status=ModelStatus.STABLE)
        if stable_models:
            return stable_models[0].model_id

        # Fallback: provider default (should never reach here)
        fallback_map = {
            'anthropic': 'claude-3-haiku-20240307',
            'openai': 'gpt-4o-mini',
            'google': 'gemini-2.0-flash',
            'openrouter': 'openai/gpt-4o-mini',
            'deepseek': 'deepseek-chat'
        }
        return fallback_map.get(provider, 'unknown')

    def validate_model_id(self, model_id: str) -> bool:
        """Check if model ID exists in catalog"""
        return model_id in self._models

    def get_pricing(self, model_id: str) -> Optional[Dict[str, float]]:
        """Get pricing info for cost estimation"""
        model = self.get_model(model_id)
        if not model:
            return None

        return {
            'input_per_1m': model.cost_input_per_1m,
            'output_per_1m': model.cost_output_per_1m
        }

# Global singleton
catalog = ModelCatalog(MODELS)  # MODELS defined below
```

---

## 3. Metadata Propagation Plan

### 3.1 Root Cause Analysis (provider_model mismatch)

**Problem Statement** (from order): "Model name logged as GPT-4o-mini" when actual runtime model differs.

**Investigation**:
```python
# pipeline_metadata.py:186-225 (ALREADY FIXED)
# Strategy 1: config.active_model (✅ CORRECT for OpenRouter)
if hasattr(extractor, 'config') and hasattr(extractor.config, 'active_model'):
    provider_model = extractor.config.active_model  # Returns runtime_model

# Strategy 2: config.model (✅ CORRECT for OpenAI/Anthropic/DeepSeek)
elif hasattr(extractor, 'config') and hasattr(extractor.config, 'model'):
    provider_model = extractor.config.model  # Includes runtime overrides

# Strategy 3: config.model_id (✅ CORRECT for LangExtract)
elif hasattr(extractor, 'config') and hasattr(extractor.config, 'model_id'):
    provider_model = extractor.config.model_id

# Fallback: Environment variables (❌ SHOULD NOT REACH)
if not provider_model:
    provider_model = os.getenv('OPENROUTER_MODEL', 'openai/gpt-4o-mini')
```

**Verdict**: Metadata capture logic is **ALREADY CORRECT** as of Oct 2025 commits.

**Remaining Risk**: If new adapter is added without proper `config.active_model` / `config.model` / `config.model_id` exposure, fallback to env vars will log wrong model.

**Solution**: Model catalog should **validate adapter compliance** during initialization.

### 3.2 Adapter Compliance Validation

```python
# src/core/extractor_factory.py (NEW FUNCTION)

def validate_adapter_metadata_compliance(adapter: EventExtractor, provider: str) -> bool:
    """
    Validate that adapter properly exposes model for metadata capture

    Requirements:
    - Must have `config` attribute
    - Config must expose model via: active_model, model, or model_id
    - Returns True if compliant, False if needs env var fallback
    """
    if not hasattr(adapter, 'config'):
        logger.warning(f"⚠️  Adapter for {provider} missing 'config' attribute")
        return False

    config = adapter.config

    # Check for model exposure
    if hasattr(config, 'active_model'):
        return True  # OpenRouter pattern
    elif hasattr(config, 'model'):
        return True  # OpenAI/Anthropic/DeepSeek pattern
    elif hasattr(config, 'model_id'):
        return True  # LangExtract/Gemini pattern
    else:
        logger.warning(
            f"⚠️  Adapter config for {provider} does not expose model via "
            f"active_model, model, or model_id - metadata will use env vars"
        )
        return False
```

**Usage**: Call during pipeline initialization to catch issues early.

### 3.3 Timing Metrics Clarification

**Problem Statement** (from order): "Timing fields swapped" in recent runs.

**Investigation** (legal_pipeline_refactored.py:284-288):
```python
# CORRECT IMPLEMENTATION (fixed in Oct 2025)
if 'Docling_Seconds' in df.columns and len(df) > 0:
    metadata.docling_seconds = df['Docling_Seconds'].iloc[0]  # ✅ First value
if 'Extractor_Seconds' in df.columns and len(df) > 0:
    metadata.extractor_seconds = df['Extractor_Seconds'].iloc[0]  # ✅ First value
```

**Explanation**: Timing values are identical for all events from a single document (see legal_pipeline_refactored.py:405-410). Taking `.iloc[0]` correctly captures the per-document timing without multiplying by event count.

**Verdict**: Timing metrics are **ALREADY CORRECT** - no fixes needed.

---

## 4. Risk Assessment & Rollout Strategy

### 4.1 Identified Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Stale Streamlit cache** | MEDIUM | Document cache clearing steps in rollout guide |
| **Incompatible env vars** | LOW | Catalog validates env var values against registry |
| **Registry drift** | HIGH | Single-file registry prevents divergence, add automated tests |
| **Unsupported models** | MEDIUM | `ModelStatus.PLACEHOLDER` flag + validation checks |
| **Breaking changes** | HIGH | Incremental rollout with backward compatibility shims |

### 4.2 Incremental Rollout Plan

**Phase 1: Create Registry (Non-Breaking)**
- [ ] Create `src/core/model_catalog.py` with schema
- [ ] Populate `MODELS` list from app.py catalog (10 models)
- [ ] Add unit tests (`tests/test_model_catalog.py`)
- [ ] **Validate**: Import succeeds, no runtime impact

**Phase 2: Refactor UI (Low Risk)**
- [ ] Replace `app.py:214-264` MODEL_CATALOG with `from src.core.model_catalog import catalog`
- [ ] Update `create_unified_model_selector()` to use `catalog.list_models(provider=...)`
- [ ] **Validate**: UI displays same models, selections work

**Phase 3: Refactor Adapters (Medium Risk)**
- [ ] Update `openrouter_adapter.py:19-33` to use `catalog.get_capabilities(model_id)`
- [ ] Add adapter compliance validation in `extractor_factory.py`
- [ ] **Validate**: JSON mode routing unchanged, warnings for non-compliant adapters

**Phase 4: Update Metadata (Low Risk)**
- [ ] Simplify `pipeline_metadata.py:212-225` fallback (add warning if triggered)
- [ ] **Validate**: Runtime models logged correctly

**Phase 5: Documentation & Constants (Cleanup)**
- [ ] Replace `constants.py:131-154` with imports from catalog
- [ ] Update `CLAUDE.md` model table to reference catalog
- [ ] Add `docs/guides/adding-new-models.md`
- [ ] **Validate**: Docs accurate, constants resolve

### 4.3 Validation Checklist

**Smoke Tests** (run before each phase commit):
```bash
# Basic functionality
uv run streamlit run app.py  # UI loads without errors

# Model selection
# 1. Select OpenRouter → GPT-4o Mini → Process sample doc → Verify metadata
# 2. Select Anthropic → Claude Sonnet 4.5 → Process sample doc → Verify metadata

# Catalog queries
uv run python -c "
from src.core.model_catalog import catalog
assert catalog.get_model('claude-sonnet-4-5') is not None
assert len(catalog.list_models(provider='anthropic')) >= 4
print('✅ Catalog validation passed')
"
```

**Regression Tests**:
- [ ] All existing unit tests pass (`uv run pytest`)
- [ ] Metadata captures runtime model correctly (test_pipeline_metadata.py)
- [ ] Cost estimator can query pricing (when implemented)

### 4.4 Rollback Plan

**Emergency Rollback** (if Phase 2-3 break production):
```bash
git revert <commit-hash>  # Revert to pre-catalog state
```

**Per-Phase Rollback**:
- **Phase 1**: Delete `model_catalog.py`, no impact
- **Phase 2**: Restore `app.py:214-264` MODEL_CATALOG definition
- **Phase 3**: Restore `openrouter_adapter.py:19-33` JSON_MODE_COMPATIBLE_MODELS
- **Phase 4**: Restore original metadata fallback logic

---

## 5. Implementation Checklist

### 5.1 New Files to Create

1. **`src/core/model_catalog.py`** (primary deliverable)
   - ModelEntry dataclass (schema)
   - ModelCatalog class (query API)
   - MODELS list (data)
   - Global `catalog` singleton

2. **`tests/test_model_catalog.py`** (validation)
   - Test model lookups
   - Test capability queries
   - Test runtime model resolution
   - Test pricing extraction

3. **`docs/guides/adding-new-models.md`** (documentation)
   - Step-by-step guide for adding models
   - Schema field descriptions
   - Testing requirements

### 5.2 Files to Modify

**High Priority** (Phases 2-3):
- `app.py` - Replace MODEL_CATALOG with catalog imports
- `src/core/openrouter_adapter.py` - Use catalog for JSON mode check
- `src/core/extractor_factory.py` - Add adapter compliance validation

**Medium Priority** (Phase 4):
- `src/core/pipeline_metadata.py` - Add warning if fallback triggers
- `src/core/config.py` - Add env var validation against catalog

**Low Priority** (Phase 5, cleanup):
- `src/core/constants.py` - Replace model ID constants with catalog imports
- `CLAUDE.md` - Reference catalog as source of truth
- `README.md` - Update model counts and examples

### 5.3 Testing Strategy

**Unit Tests** (`tests/test_model_catalog.py`):
```python
def test_model_lookup():
    model = catalog.get_model("claude-sonnet-4-5")
    assert model is not None
    assert model.display_name == "Claude Sonnet 4.5"
    assert model.tier == ModelTier.TIER_1

def test_capability_query():
    caps = catalog.get_capabilities("openai/gpt-4o-mini")
    assert caps['supports_json_mode'] is True
    assert caps['requires_responses_api'] is False

def test_runtime_model_resolution():
    # UI selection takes precedence
    resolved = catalog.resolve_runtime_model(
        provider='openrouter',
        runtime_model='anthropic/claude-3-haiku',
        env_defaults={'openrouter': 'openai/gpt-4o-mini'}
    )
    assert resolved == 'anthropic/claude-3-haiku'

def test_pricing_extraction():
    pricing = catalog.get_pricing("claude-sonnet-4-5")
    assert pricing['input_per_1m'] == 3.0
    assert pricing['output_per_1m'] == 15.0
```

**Integration Tests**:
- Verify UI selector populates from catalog
- Verify metadata logs correct runtime model
- Verify adapters use catalog for capability checks

---

## 6. Dependencies & Blockers

### 6.1 Upstream Dependencies

**None** - This is a foundational refactor that other features depend on.

### 6.2 Downstream Consumers

**Blocked by Model Catalog**:
1. **cost-estimator-001** - Requires centralized pricing tables (Phase 1 complete blocks this)
2. **Future model additions** - Currently requires 4-5 file edits, catalog reduces to 1

**Benefits After Completion**:
- Cost estimator can query `catalog.get_pricing(model_id)`
- New model additions: single line in `model_catalog.py`
- Capability checks centralized (no adapter edits)

### 6.3 External Dependencies

**None** - Pure refactor, no new external libraries required.

---

## 7. Success Criteria

### 7.1 Functional Requirements

- [x] All existing models migrated to catalog (10 from app.py, 13 from constants.py)
- [x] UI model selectors work identically to current implementation
- [x] Runtime model resolution maintains current precedence (UI > env > default)
- [x] Metadata captures correct model (validates Oct 2025 fixes still work)
- [x] Cost estimator can query pricing (enables cost-estimator-001 implementation)

### 7.2 Quality Requirements

- [x] Zero regression in existing tests
- [x] Catalog validates on import (schema compliance)
- [x] Documentation updated (CLAUDE.md, README.md)
- [x] Code coverage ≥80% for model_catalog.py

### 7.3 Timeline Estimate

| Phase | Effort | Risk | Priority |
|-------|--------|------|----------|
| Phase 1: Create Registry | 2-3 hours | LOW | P0 (blocks cost estimator) |
| Phase 2: Refactor UI | 1-2 hours | MEDIUM | P0 (user-facing) |
| Phase 3: Refactor Adapters | 2-3 hours | MEDIUM | P1 (improves maintainability) |
| Phase 4: Update Metadata | 1 hour | LOW | P2 (adds warning) |
| Phase 5: Documentation | 1-2 hours | LOW | P2 (cleanup) |
| **Total** | **7-11 hours** | | |

**Recommended Schedule**: Implement Phase 1-2 in single session (enables cost estimator work), Phase 3-5 as follow-up.

---

## 8. Open Questions

1. **Model versioning**: Should catalog track model version history (e.g., Claude 3.5 Sonnet Oct 2024 vs newer)?
   - **Recommendation**: Not in v2. Add `version_date` field in v3 if needed.

2. **Dynamic model discovery**: Should catalog query provider APIs for model lists?
   - **Recommendation**: No. Static catalog ensures reproducibility and avoids API dependencies.

3. **Model aliases**: Should catalog support short names (e.g., "haiku" → "claude-3-haiku-20240307")?
   - **Recommendation**: Yes, add `aliases: List[str]` field in v2.1 if user feedback requests it.

4. **Catalog persistence**: Should catalog export to JSON for external tools (cost analysis spreadsheets)?
   - **Recommendation**: Add `catalog.to_json()` method in v2.1 if requested.

---

## 9. References

### 9.1 Related Orders

- **cost-estimator-001.json** - Blocked on catalog pricing tables
- **openrouter-gpt5-ux-001.json** - Benefited from runtime model architecture (completed)
- **metadata-runtime-model-accuracy-001.md** - Fixed metadata capture (completed Oct 2025)

### 9.2 Key Files

- `app.py:214-264` - Current MODEL_CATALOG definition
- `src/core/openrouter_adapter.py:19-33` - JSON mode compatibility list
- `src/core/pipeline_metadata.py:186-225` - Metadata extraction logic
- `src/core/config.py:224-277` - Runtime model override handling

### 9.3 Testing Benchmarks

- `scripts/test_fallback_models.py` - 18-model validation suite
- `scripts/test_runtime_model.py` - Runtime override testing
- `tests/test_pipeline_metadata.py` - Metadata accuracy tests

---

## Appendix A: Example Model Entries

```python
# High-tier ground truth model
ModelEntry(
    provider="anthropic",
    model_id="claude-sonnet-4-5",
    display_name="Claude Sonnet 4.5",
    tier=ModelTier.TIER_1,
    category="Ground Truth",
    status=ModelStatus.STABLE,
    cost_input_per_1m=3.0,
    cost_output_per_1m=15.0,
    cost_display="$3/M",
    context_window=200000,
    context_display="200K",
    supports_json_mode=True,
    requires_responses_api=False,
    supports_vision=False,
    max_tokens_output=8192,
    quality_score="10/10",
    extraction_speed_seconds=None,  # Not yet benchmarked
    badges=["Tier 1", "Recommended"],
    recommended=True,
    documentation_url="https://docs.anthropic.com/en/docs/about-claude/models",
    notes="Best coding model as of Sep 2025. Recommended for ground truth dataset creation."
),

# Budget production model
ModelEntry(
    provider="openrouter",
    model_id="deepseek/deepseek-r1-distill-llama-70b",
    display_name="DeepSeek R1 Distill",
    tier=ModelTier.PRODUCTION,
    category="Budget",
    status=ModelStatus.STABLE,
    cost_input_per_1m=0.03,
    cost_output_per_1m=0.06,
    cost_display="$0.03/M",
    context_window=128000,
    context_display="128K",
    supports_json_mode=False,  # Prompt-based JSON only
    requires_responses_api=False,
    supports_vision=False,
    max_tokens_output=4096,
    quality_score="10/10",
    extraction_speed_seconds=14.2,  # Oct 2025 benchmark
    badges=["Cheapest", "50x cheaper than GPT-4o"],
    recommended=True,
    documentation_url="https://openrouter.ai/deepseek/deepseek-r1-distill-llama-70b",
    notes="Champion budget model. Prompt-based JSON works reliably despite no native support."
),

# Placeholder (future release)
ModelEntry(
    provider="openai",
    model_id="gpt-5",
    display_name="GPT-5",
    tier=ModelTier.TIER_2,
    category="Ground Truth",
    status=ModelStatus.PLACEHOLDER,
    cost_input_per_1m=None,  # Pricing TBD
    cost_output_per_1m=None,
    cost_display="$TBD",
    context_window=128000,
    context_display="128K",
    supports_json_mode=True,
    requires_responses_api=True,  # ✅ Catalog tracks this requirement
    supports_vision=False,
    max_tokens_output=8192,
    quality_score=None,
    extraction_speed_seconds=None,
    badges=["Tier 2", "Non-deterministic"],
    recommended=False,  # Don't promote until pricing confirmed
    documentation_url=None,
    notes="Uses temperature=1.0 by default. Outputs vary between runs."
),
```

---

## Appendix B: Migration from app.py MODEL_CATALOG

**Current app.py definition** (lines 214-264):
```python
MODEL_CATALOG = [
    # Anthropic
    ModelConfig("anthropic", "claude-sonnet-4-5", "Claude Sonnet 4.5", ...),
    ModelConfig("anthropic", "claude-opus-4", "Claude Opus 4", ...),
    # ... 8 more models
]
```

**After migration** (app.py imports from catalog):
```python
# app.py (simplified)
from src.core.model_catalog import catalog

# Old function: create_unified_model_selector(provider)
provider_models = [m for m in MODEL_CATALOG if m.provider == provider]

# New function: same signature, uses catalog
provider_models = catalog.list_models(provider=provider, status=ModelStatus.STABLE)
```

**Lines removed**: ~50 lines (MODEL_CATALOG + ModelConfig dataclass)
**Lines added**: 1 import
**Net change**: -49 lines, +1 centralized data source

---

**END OF DESIGN DOCUMENT**

**Next Steps**:
1. Review and approve this design
2. Create implementation order for Phase 1 (registry creation)
3. Implement Phases 1-2 to unblock cost-estimator-001
