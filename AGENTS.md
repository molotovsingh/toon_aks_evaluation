# Repository Guidelines

## Project Structure & Module Organization
Core Docling-to-legal-events pipeline lives under `src/core/`; `extractor_factory.py` registers provider adapters and points to legacy wrappers in `extractors/`. Shared helpers land in `utils/`, `ui/`, and `visualization/` for Streamlit rendering. Launch the main Streamlit UI via top-level `app.py`; focused demos such as `examples/legal_events_app.py` demonstrate alternative flows. CLI automation sits in `src/main.py`, while generated artifacts and logs accumulate in `output/` for manual review. Reference docs, scripts, and tests reside in `docs/`, `scripts/`, and `tests/` respectively.

## Build, Test, and Development Commands
- `uv sync` installs pinned dependencies from `pyproject.toml` and `uv.lock`.
- `uv run python3 -m streamlit run app.py` boots the primary UI; swap in `examples/*.py` to explore demos. **Note:** Use the module syntax (`-m streamlit`) instead of direct command (`streamlit run`) to avoid "Failed to spawn: streamlit" errors with uv.
- `uv run python src/main.py` executes the CLI harness for scripted LangExtract runs.
- `uv run python tests/run_all_tests.py --quick` runs the smoke suite; drop `--quick` for full coverage.
- `uv run pytest tests/test_specific_file.py` runs a single test file; use `::test_function` for specific tests.

## Coding Style & Naming Conventions
Use 4-space indentation, Python type hints, and docstrings aligned with `src/core/interfaces.py`. Prefer `snake_case` for functions, `PascalCase` for classes, and ALL_CAPS for constants. Centralize event schema constants in `src/core/constants.py`, and expose configuration through dataclasses instead of hard-coded literals. Break complex Streamlit callbacks into helpers to keep UI logic legible. Import standard library first, then third-party, then local imports. Handle errors with try/except blocks and log failures; avoid silent failures.

## Testing Guidelines
Write tests under `tests/` with filenames like `test_*.py`, tagging acceptance IDs (for example, `AC-00X`) in logs. Provider-specific suites skip automatically when API keys are missing; mock provider responses for unit coverage. Use pytest fixtures for setup/teardown. Capture verification artifacts with `uv run python tests/run_all_tests.py --report` when preparing evidence.

## Commit & Pull Request Guidelines
Compose present-tense, task-focused commits (for example, "Add Docling adapter metrics") and group related pipeline changes together. Reference acceptance IDs, executed scripts, or test commands in commit bodies. Pull requests should summarize behavioral changes, link tracking issues, and include UI screenshots or terminal captures when relevant. Remove temporary debug assets before requesting review.

## Security & Configuration Tips
Load secrets from environment variables (`EVENT_EXTRACTOR`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, etc.) and avoid committing them to the repo. Tune performance via configuration toggles like `DOCLING_TABLE_MODE` and `LANGEXTRACT_MAX_WORKERS` rather than editing vendor adapters. Review `SECURITY.md` and run `uv run python scripts/generate_adicr_report.py --refresh` before finalizing documentation-heavy changes.
