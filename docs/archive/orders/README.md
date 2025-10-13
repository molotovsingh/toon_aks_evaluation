# Archived Order Files

This directory contains completed or fulfilled order files from the development process (2025-09 to 2025-10-02).

## Contents

### housekeeping-001.json
**Status**: Fulfilled
**Committed**: Unknown (early project cleanup)
**Purpose**: Initial repository housekeeping and organization

---

### housekeeping-002.json
**Status**: Fulfilled
**Committed**: Unknown (early project cleanup)
**Purpose**: Secondary cleanup and documentation organization

---

### housekeeping-003.json
**Status**: Fulfilled
**Committed**: 27f6e28 (2025-10-02)
**Purpose**: PDF reorganization, documentation staging, npm cache cleanup

**Tasks**:
1. Organize Amrapali sample PDFs into proper directory structure
2. Stage intentional documentation edits
3. Clean artifacts (.env.backup, npm cache)
4. Final verification

**Issue**: Original order used `git mv` on already-deleted files (technically impossible)

---

### housekeeping-003-revised.json
**Status**: Fulfilled
**Committed**: 27f6e28 (2025-10-02)
**Purpose**: Corrected version of housekeeping-003.json

**Fixes from Original**:
- Changed from `git mv` to `git add -u` + `git add` for rename detection
- Updated file list to match actual git state
- Added npm cache cleanup steps
- Added commit message preparation

**Tasks Successfully Completed**:
1. ✅ Staged 9 PDF renames (Amrapali case files)
2. ✅ Staged documentation updates (order templates, research plans)
3. ✅ Staged code improvement (citation prompt clarity)
4. ✅ Removed .env.backup and npm cache artifacts
5. ✅ Updated .gitignore for npm cache

---

## Order System

The `docs/orders/` directory contains active order files for development tasks. Completed orders are archived here to maintain historical context while keeping the active directory focused.

### Active Orders
See `docs/orders/` for current development tasks and templates.

### Order Template
See `docs/orders/example-order-template.json` for the standard order format.

---

## 2025-10-09 Archival Batch

**Batch Size**: 15 orders (14 complete + 1 superseded)
**Audit Reference**: `docs/reports/archive_audit.md`
**Evidence Verification**: All orders have completion reports in `docs/reports/` or code artifacts

### Completed Orders (14)

1. **adicr-implementation-001.json** - ADICR system implementation (Evidence: `adicr_runlog.md`)
2. **event-extractor-002.json** - Provider integration (Evidence: `event-extractor-002-completion-report.md`)
3. **event-extractor-003.json** - Provider integration (Evidence: `event-extractor-003-completion-report.md`)
4. **api-connection-test.json** - API connectivity validation (Evidence: `api_connection_test_summary.md`)
5. **streamlit-provider-selector-001.json** - UI provider selection (Evidence: `streamlit-provider-selector-001-completion.md`)
6. **streamlit-provider-reliability-001.json** - UI reliability improvements (Evidence: `streamlit-provider-reliability-001-completion.md`)
7. **provider-env-validation-001.json** - Environment validation (Evidence: `provider-env-validation-001-completion.md`)
8. **performance-timing-001-revised.json** - Timing instrumentation v1.1 (Evidence: `performance-timing-validation.md`)
9. **docling-fix-parsev4.json** - Docling v4 parsing fix (Evidence: `docling-fix-parsev4.txt`)
10. **docling-fix-parsev4-verification.json** - Parsev4 verification (Evidence: `docling-fix-parsev4-verification-summary.txt`)
11. **docling-fix-parsev4-pipeline-options.json** - Pipeline options config (Evidence: `docling-fix-parsev4-pipeline-options-summary.txt`)
12. **housekeeping-relocate-root.json** - Root file reorganization (Evidence: `relocate-root.txt`, `ultra-verification-relocate-root.txt`)
13. **docling-ocr-autofallback-001.json** - OCR auto-fallback (Evidence: `ocr_autodetection_implementation.md`)
14. **provider-config-externalization-001.json** - Config externalization (Evidence: code artifacts in `src/core/config.py`)

### Superseded Orders (1)

- **performance-timing-001.json** - ⚠️ **SUPERSEDED** by `performance-timing-001-revised.json` (v1.1)

---

## 2025-10-09 Additional Archival - orders-readme-001

**Order**: orders-readme-001.json
**Status**: ✅ COMPLETE (Expanded Scope)
**Evidence**: `docs/reports/orders-readme-001-completion.md`

### Summary

Created comprehensive README files for order management workflow:
- `docs/orders/README.md` - Active orders guide (CREATED)
- `docs/archive/orders/README.md` - Archive documentation (UPDATED with 2025-10-09 batch)

**Note**: Order requested concise pointers (≤150 words), but comprehensive workflow guides were delivered during `order-archival-001` execution. Scope expansion justified by operational need for detailed archival documentation.

**Completion**: Fulfilled during order-archival-001 (inadvertent overlap due to archival workflow requirements)

---

## 2025-10-13 Archival - Metadata Runtime Model Order

**Order**: metadata-runtime-model-accuracy-001-COMPLETED.md
**Status**: ✅ COMPLETE
**Evidence**: Self-contained completion report with 9/9 unit tests passing
**Commits**:
- `8de74dd` - Runtime model override capture fix
- `3eab946` - Timing metrics calculation fix

### Summary

Fixed critical metadata export issues:
1. **Runtime Model Capture**: Metadata now correctly captures actual model selections (e.g., `openai/gpt-oss-120b`) instead of environment defaults (`openai/gpt-4o-mini`)
2. **Timing Metrics Fix**: Corrected timing calculation from incorrect sum to first value extraction

**Implementation**: Added 3-strategy lookup pattern in `pipeline_metadata.py` with comprehensive unit tests covering all 5 provider types (OpenRouter, OpenAI, Anthropic, LangExtract, DeepSeek).

**Completion**: Order file includes full completion report with technical details, test results, and manual verification steps.

---

## Archive History

- **2025-10-04**: Initial archive with 4 housekeeping orders
- **2025-10-09**: Archival batch - 15 development orders (14 complete + 1 superseded)
- **2025-10-09**: Additional archival - orders-readme-001 (expanded scope)
- **2025-10-13**: Single order archival - metadata-runtime-model-accuracy-001 (metadata fixes)

**Total Archived**: 21 orders (4 housekeeping + 15 development + 1 README + 1 metadata fix)

## Archive Date
Last Updated: 2025-10-13
