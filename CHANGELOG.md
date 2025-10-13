# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-10-13

### Added
- **GPT-5 reasoning model support** with Responses API integration for ground truth extraction ([0f904b8](../../commit/0f904b8), [6004628](../../commit/6004628))
- **Gemini 2.5 Pro** as alternative document extractor with optimized legal prompt ([a8065e2](../../commit/a8065e2))
- **Claude 4.5 Sonnet and Opus 4** as selectable extraction models (llm-model-upgrade-001)
- **Pre-commit hooks** for Python syntax validation and fast test execution ([4cc7fab](../../commit/4cc7fab))
- **Major UX overhaul** with pipeline visualization and 3-stage Excel-style table ([ea512c5](../../commit/ea512c5))
- **Unified model search** with normalized search (e.g., "gpt 5" matches "gpt-5") and advanced filters
- **OpenAI provider UI** with GPT-OSS-120B privacy hedge positioning ([8ff6b67](../../commit/8ff6b67))
- **Email normalization** for clean .eml file text extraction (eml-normalization-001 completed)
- **Image support** for document processing (image-support-001 completed)
- **Testing infrastructure** with benchmarks and 3-judge evaluation system ([ae7eb27](../../commit/ae7eb27))
- **Order archival system** with comprehensive audit trail (21 orders archived)
- **Order index system** for complete order visibility - improved from 8 to 16 orders (100% coverage) ([fe65ee4](../../commit/fe65ee4))

### Fixed
- **Metadata exports** now capture actual runtime model selections instead of environment defaults ([8de74dd](../../commit/8de74dd))
- **Timing metrics calculation** corrected from incorrect sum to first value extraction ([3eab946](../../commit/3eab946))
- **TESSDATA_PREFIX** environment variable loading for OCR availability

### Changed
- **OpenRouter adapter** migrated from requests library to OpenAI SDK for better compatibility ([69237d8](../../commit/69237d8))
- **UI font sizes** adjusted for improved information density ([d9e6ae2](../../commit/d9e6ae2))

### Infrastructure
- Order archival workflow with evidence tracking and completion reports
- Order index with status legend (Template, Completed, Active, Planning, Needs Investigation, Superseded)
- Pre-commit hooks ensuring code quality before commits
- Comprehensive testing framework with quick and full test modes

### Known Limitations
- **Qwen QwQ 32B**: 7/10 quality rating, may miss events on complex documents (budget option at $0.115/M)
- **Speculative models**: GPT-OSS-120B and some OSS models pending full production validation
- **Model catalog v2**: Current model selection is functional but architecture redesign planned
- **Empty document handling**: Fallback table includes extra timing columns (AC-015, cosmetic issue)
- **Large document performance**: Processing may marginally exceed 10s target on complex documents (AC-017)

### Planning (Not Yet Implemented)
- **DuckDB ingestion system** for pipeline metadata with queryable store (duckdb-ingestion-001)
- **Model catalog v2 architecture** with centralized registry and metadata propagation (model-catalog-v2-architecture-001)

---

## [0.1.0] - 2025-09-XX

Initial proof-of-concept release for legal event extraction testing.

### Added
- Core document processing with Docling
- LangExtract integration for Gemini-based event extraction
- Basic Streamlit UI for document upload and processing
- Five-column event table output (No, Date, Event Particulars, Citation, Document Reference)
- Support for PDF, DOCX, HTML, PPTX, and .eml file formats
- OCR support with Tesseract and EasyOCR backends
- Basic provider selection (LangExtract, OpenRouter, OpenCode Zen)
- Export functionality (CSV, XLSX, JSON)

### Notes
- This was the initial proof-of-concept release focused on validating the Docling + LangExtract combination
- See commit `a8065e2` and earlier for baseline implementation details
