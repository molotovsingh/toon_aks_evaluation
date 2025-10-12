#!/usr/bin/env python3
"""
Tests for ADICR (Automated Documentation Integrity and Coverage Report)

Validates:
- AST-based provider and env var extraction
- Documentation drift detection
- Report generation (markdown and JSON)
- Edge cases and error handling

Acceptance Criteria:
- AC-ADICR-001: Extract providers from EVENT_PROVIDER_REGISTRY using AST
- AC-ADICR-002: Extract env vars from config dataclasses
- AC-ADICR-003: Detect missing providers in documentation
- AC-ADICR-004: Detect missing env vars in documentation
- AC-ADICR-005: Generate markdown report with severity counts
- AC-ADICR-006: Generate JSON report with machine-readable structure
"""

import json
import tempfile
from pathlib import Path
from textwrap import dedent

import pytest

from src.utils.adicr_helpers import (
    extract_providers_from_registry,
    extract_env_vars_from_config,
    find_text_in_file,
    extract_markdown_section,
    fuzzy_match_heading,
    ProviderInventory,
    EnvVarInventory
)


# ============================================================================
# Fixtures - Test data setup
# ============================================================================

@pytest.fixture
def sample_registry_file(tmp_path):
    """Create a temporary Python file with a sample provider registry"""
    content = dedent('''
    from typing import Dict, Callable

    def create_provider_a():
        pass

    def create_provider_b():
        pass

    EVENT_PROVIDER_REGISTRY: Dict[str, Callable] = {
        "langextract": create_provider_a,
        "openrouter": create_provider_b,
        "deepseek": lambda: None,
    }

    OTHER_REGISTRY = {
        "should_not": "be_extracted"
    }
    ''')

    file_path = tmp_path / "test_registry.py"
    file_path.write_text(content)
    return str(file_path)


@pytest.fixture
def sample_config_file(tmp_path):
    """Create a temporary config.py with sample env var extractions"""
    content = dedent('''
    from dataclasses import dataclass, field

    def env_str(name: str, default: str = "") -> str:
        import os
        return os.getenv(name, default)

    def env_bool(name: str, default: bool = False) -> bool:
        import os
        return os.getenv(name, str(default)).lower() in ("true", "1", "yes")

    def env_int(name: str, default: int = 0) -> int:
        import os
        return int(os.getenv(name, str(default)))

    @dataclass
    class LangExtractConfig:
        api_key: str = field(default_factory=lambda: env_str("GEMINI_API_KEY", ""))
        max_workers: int = field(default_factory=lambda: env_int("LANGEXTRACT_MAX_WORKERS", 4))
        debug_mode: bool = field(default_factory=lambda: env_bool("LANGEXTRACT_DEBUG", False))

    @dataclass
    class OpenRouterConfig:
        api_key: str = field(default_factory=lambda: env_str("OPENROUTER_API_KEY", ""))
        model: str = field(default_factory=lambda: env_str("OPENROUTER_MODEL", "gpt-4"))
    ''')

    file_path = tmp_path / "test_config.py"
    file_path.write_text(content)
    return str(file_path)


@pytest.fixture
def sample_markdown_file(tmp_path):
    """Create a temporary markdown file for documentation testing"""
    content = dedent('''
    # Main Title

    Some introduction text.

    ## Provider Selection

    We support the following providers:
    - langextract
    - openrouter

    ## Environment Variables

    Configure with these variables:
    - GEMINI_API_KEY
    - OPENROUTER_API_KEY

    ## Another Section

    More content here.
    ''')

    file_path = tmp_path / "test_docs.md"
    file_path.write_text(content)
    return str(file_path)


@pytest.fixture
def sample_manifest(tmp_path):
    """Create a sample ADICR manifest"""
    manifest = {
        "version": "1.0",
        "targets": {
            "providers": {
                "source_file": "test_registry.py",
                "source_pattern": "EVENT_PROVIDER_REGISTRY",
                "expected_locations": [
                    {
                        "file": "test_docs.md",
                        "section": "Provider Selection",
                        "description": "Main documentation"
                    }
                ]
            },
            "env_vars": {
                "source_file": "test_config.py",
                "expected_locations": [
                    {
                        "file": "test_docs.md",
                        "section": "Environment Variables",
                        "description": "Env var documentation"
                    }
                ]
            }
        }
    }

    file_path = tmp_path / "test_manifest.json"
    file_path.write_text(json.dumps(manifest, indent=2))
    return str(file_path)


# ============================================================================
# Unit Tests - Helper Functions
# ============================================================================

class TestProviderExtraction:
    """AC-ADICR-001: Extract providers from EVENT_PROVIDER_REGISTRY"""

    def test_extract_providers_basic(self, sample_registry_file):
        """Should extract all provider keys from registry"""
        inventory = extract_providers_from_registry(
            sample_registry_file,
            "EVENT_PROVIDER_REGISTRY"
        )

        assert isinstance(inventory, ProviderInventory)
        assert inventory.providers == {"langextract", "openrouter", "deepseek"}
        assert inventory.registry_name == "EVENT_PROVIDER_REGISTRY"
        assert inventory.line_number > 0

        print("✅ AC-ADICR-001: Provider extraction works correctly")

    def test_extract_providers_empty_registry(self, tmp_path):
        """Should handle empty registry gracefully"""
        content = dedent('''
        EVENT_PROVIDER_REGISTRY = {}
        ''')

        file_path = tmp_path / "empty_registry.py"
        file_path.write_text(content)

        inventory = extract_providers_from_registry(str(file_path), "EVENT_PROVIDER_REGISTRY")
        assert inventory.providers == set()

    def test_extract_providers_missing_file(self):
        """Should raise FileNotFoundError for missing file"""
        with pytest.raises(FileNotFoundError):
            extract_providers_from_registry("/nonexistent/file.py", "EVENT_PROVIDER_REGISTRY")

    def test_extract_providers_wrong_registry_name(self, sample_registry_file):
        """Should return empty set if registry name not found"""
        inventory = extract_providers_from_registry(
            sample_registry_file,
            "NONEXISTENT_REGISTRY"
        )
        assert inventory.providers == set()


class TestEnvVarExtraction:
    """AC-ADICR-002: Extract env vars from config dataclasses"""

    def test_extract_env_vars_basic(self, sample_config_file):
        """Should extract all env var calls from dataclasses"""
        env_vars = extract_env_vars_from_config(sample_config_file)

        assert isinstance(env_vars, list)
        assert len(env_vars) == 5

        # Check specific extractions
        var_names = {var.var_name for var in env_vars}
        assert "GEMINI_API_KEY" in var_names
        assert "LANGEXTRACT_MAX_WORKERS" in var_names
        assert "LANGEXTRACT_DEBUG" in var_names
        assert "OPENROUTER_API_KEY" in var_names
        assert "OPENROUTER_MODEL" in var_names

        # Check types
        types = {var.var_name: var.var_type for var in env_vars}
        assert types["GEMINI_API_KEY"] == "str"
        assert types["LANGEXTRACT_MAX_WORKERS"] == "int"
        assert types["LANGEXTRACT_DEBUG"] == "bool"

        print("✅ AC-ADICR-002: Env var extraction works correctly")

    def test_extract_env_vars_missing_file(self):
        """Should raise FileNotFoundError for missing file"""
        with pytest.raises(FileNotFoundError):
            extract_env_vars_from_config("/nonexistent/config.py")

    def test_extract_env_vars_no_dataclasses(self, tmp_path):
        """Should return empty list if no dataclasses found"""
        content = dedent('''
        # Just a regular Python file
        x = 42
        ''')

        file_path = tmp_path / "no_dataclasses.py"
        file_path.write_text(content)

        env_vars = extract_env_vars_from_config(str(file_path))
        assert env_vars == []


class TestMarkdownHelpers:
    """Test markdown parsing utilities"""

    def test_find_text_case_insensitive(self, sample_markdown_file):
        """Should find text regardless of case"""
        matches = find_text_in_file(sample_markdown_file, "langextract", case_sensitive=False)
        assert len(matches) == 1

        matches_upper = find_text_in_file(sample_markdown_file, "LANGEXTRACT", case_sensitive=False)
        assert len(matches_upper) == 1

    def test_find_text_case_sensitive(self, sample_markdown_file):
        """Should respect case sensitivity"""
        matches = find_text_in_file(sample_markdown_file, "GEMINI_API_KEY", case_sensitive=True)
        assert len(matches) == 1

        matches_wrong_case = find_text_in_file(sample_markdown_file, "gemini_api_key", case_sensitive=True)
        assert len(matches_wrong_case) == 0

    def test_extract_markdown_section(self, sample_markdown_file):
        """Should extract content under specific heading"""
        content = extract_markdown_section(sample_markdown_file, "Provider Selection")

        assert content is not None
        assert "langextract" in content
        assert "openrouter" in content
        assert "Another Section" not in content  # Should stop at next heading

    def test_extract_markdown_section_missing(self, sample_markdown_file):
        """Should return None for missing section"""
        content = extract_markdown_section(sample_markdown_file, "Nonexistent Section")
        assert content is None

    def test_fuzzy_match_heading(self, sample_markdown_file):
        """Should fuzzy match heading with similar text"""
        match = fuzzy_match_heading(sample_markdown_file, "provider selection", confidence_threshold=0.5)
        assert match == "Provider Selection"

        match_partial = fuzzy_match_heading(sample_markdown_file, "provider", confidence_threshold=0.5)
        assert match_partial == "Provider Selection"


# ============================================================================
# Integration Tests - Full ADICR Workflow
# ============================================================================

class TestADICRIntegration:
    """Integration tests for full ADICR workflow"""

    def test_detect_missing_provider_in_docs(self, sample_registry_file, sample_markdown_file, tmp_path):
        """AC-ADICR-003: Should detect providers in code but missing from docs"""
        from scripts.generate_adicr_report import ADICRReport

        # Create manifest that references our test files
        manifest = {
            "version": "1.0",
            "targets": {
                "providers": {
                    "source_file": sample_registry_file,
                    "source_pattern": "EVENT_PROVIDER_REGISTRY",
                    "expected_locations": [
                        {
                            "file": sample_markdown_file,
                            "section": "Provider Selection",
                            "description": "Test docs"
                        }
                    ]
                }
            }
        }

        manifest_path = tmp_path / "test_manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        # Run ADICR
        reporter = ADICRReport(str(manifest_path))
        reporter.check_provider_parity()

        # Should detect 'deepseek' missing from docs
        provider_issues = [d for d in reporter.discrepancies if d['category'] == 'providers']
        assert len(provider_issues) == 1
        assert 'deepseek' in provider_issues[0]['missing']

        print("✅ AC-ADICR-003: Missing provider detection works")

    def test_detect_missing_env_vars_in_docs(self, sample_config_file, sample_markdown_file, tmp_path):
        """AC-ADICR-004: Should detect env vars in code but missing from docs"""
        from scripts.generate_adicr_report import ADICRReport

        manifest = {
            "version": "1.0",
            "targets": {
                "env_vars": {
                    "source_file": sample_config_file,
                    "expected_locations": [
                        {
                            "file": sample_markdown_file,
                            "section": "Environment Variables",
                            "description": "Test docs"
                        }
                    ]
                }
            }
        }

        manifest_path = tmp_path / "test_manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        # Run ADICR
        reporter = ADICRReport(str(manifest_path))
        reporter.check_env_var_coverage()

        # Should detect missing env vars
        env_issues = [d for d in reporter.discrepancies if d['category'] == 'env_vars']
        assert len(env_issues) == 1

        missing_vars = env_issues[0]['missing_vars']
        assert 'LANGEXTRACT_MAX_WORKERS' in missing_vars
        assert 'LANGEXTRACT_DEBUG' in missing_vars
        assert 'OPENROUTER_MODEL' in missing_vars

        print("✅ AC-ADICR-004: Missing env var detection works")

    def test_generate_markdown_report(self, sample_registry_file, sample_markdown_file, tmp_path):
        """AC-ADICR-005: Should generate markdown report with severity counts"""
        from scripts.generate_adicr_report import ADICRReport

        manifest = {
            "version": "1.0",
            "targets": {
                "providers": {
                    "source_file": sample_registry_file,
                    "source_pattern": "EVENT_PROVIDER_REGISTRY",
                    "expected_locations": [
                        {
                            "file": sample_markdown_file,
                            "section": "Provider Selection",
                            "description": "Test docs"
                        }
                    ]
                }
            }
        }

        manifest_path = tmp_path / "test_manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        reporter = ADICRReport(str(manifest_path))
        reporter.check_provider_parity()

        # Generate markdown report
        markdown = reporter.generate_markdown_report()

        assert "# ADICR Report" in markdown
        assert "## Summary" in markdown
        assert "Critical Issues" in markdown
        assert "Provider Parity Check" in markdown
        assert "deepseek" in markdown.lower()

        print("✅ AC-ADICR-005: Markdown report generation works")

    def test_generate_json_report(self, sample_registry_file, sample_markdown_file, tmp_path):
        """AC-ADICR-006: Should generate JSON report with machine-readable structure"""
        from scripts.generate_adicr_report import ADICRReport

        manifest = {
            "version": "1.0",
            "targets": {
                "providers": {
                    "source_file": sample_registry_file,
                    "source_pattern": "EVENT_PROVIDER_REGISTRY",
                    "expected_locations": [
                        {
                            "file": sample_markdown_file,
                            "section": "Provider Selection",
                            "description": "Test docs"
                        }
                    ]
                }
            }
        }

        manifest_path = tmp_path / "test_manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        reporter = ADICRReport(str(manifest_path))
        reporter.check_provider_parity()

        # Generate JSON report
        json_report = reporter.generate_json_report()

        assert 'timestamp' in json_report
        assert 'version' in json_report
        assert 'discrepancies' in json_report
        assert 'summary' in json_report

        summary = json_report['summary']
        assert 'total_critical' in summary
        assert 'total_warnings' in summary
        assert 'total_errors' in summary
        assert 'status' in summary

        assert summary['total_critical'] >= 1  # Should have provider parity issue
        assert summary['status'] == 'drift_detected'

        print("✅ AC-ADICR-006: JSON report generation works")

    def test_all_checks_pass_when_synchronized(self, tmp_path):
        """Should report success when code and docs are in sync"""
        from scripts.generate_adicr_report import ADICRReport

        # Create registry with just one provider
        registry_content = dedent('''
        EVENT_PROVIDER_REGISTRY = {
            "langextract": lambda: None,
        }
        ''')
        registry_path = tmp_path / "sync_registry.py"
        registry_path.write_text(registry_content)

        # Create docs that mention that provider
        docs_content = dedent('''
        # Providers

        We support langextract.
        ''')
        docs_path = tmp_path / "sync_docs.md"
        docs_path.write_text(docs_content)

        # Create manifest
        manifest = {
            "version": "1.0",
            "targets": {
                "providers": {
                    "source_file": str(registry_path),
                    "source_pattern": "EVENT_PROVIDER_REGISTRY",
                    "expected_locations": [
                        {
                            "file": str(docs_path),
                            "section": "Providers",
                            "description": "Test docs"
                        }
                    ]
                }
            }
        }

        manifest_path = tmp_path / "sync_manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        # Run ADICR
        reporter = ADICRReport(str(manifest_path))
        reporter.check_provider_parity()

        # Should have no discrepancies
        assert len(reporter.discrepancies) == 0

        json_report = reporter.generate_json_report()
        assert json_report['summary']['status'] == 'synchronized'

        print("✅ ADICR reports success when code and docs are synchronized")


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_malformed_python_file(self, tmp_path):
        """Should handle syntax errors gracefully"""
        content = "this is not valid python ]["

        file_path = tmp_path / "malformed.py"
        file_path.write_text(content)

        with pytest.raises(SyntaxError):
            extract_providers_from_registry(str(file_path), "EVENT_PROVIDER_REGISTRY")

    def test_missing_manifest_file(self):
        """Should raise FileNotFoundError for missing manifest"""
        from scripts.generate_adicr_report import ADICRReport

        with pytest.raises(FileNotFoundError):
            ADICRReport("/nonexistent/manifest.json")

    def test_invalid_manifest_structure(self, tmp_path):
        """Should raise ValueError for invalid manifest"""
        from scripts.generate_adicr_report import ADICRReport

        manifest = {"invalid": "structure"}

        manifest_path = tmp_path / "invalid_manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        with pytest.raises(ValueError, match="missing 'targets'"):
            ADICRReport(str(manifest_path))

    def test_missing_documentation_file(self, sample_registry_file, tmp_path):
        """Should report all providers as missing if doc file doesn't exist"""
        from scripts.generate_adicr_report import ADICRReport

        manifest = {
            "version": "1.0",
            "targets": {
                "providers": {
                    "source_file": sample_registry_file,
                    "source_pattern": "EVENT_PROVIDER_REGISTRY",
                    "expected_locations": [
                        {
                            "file": "/nonexistent/docs.md",
                            "description": "Missing docs"
                        }
                    ]
                }
            }
        }

        manifest_path = tmp_path / "test_manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        reporter = ADICRReport(str(manifest_path))
        reporter.check_provider_parity()

        # All providers should be reported as missing
        provider_issues = [d for d in reporter.discrepancies if d['category'] == 'providers']
        assert len(provider_issues) == 1
        assert len(provider_issues[0]['missing']) == 3  # All 3 providers


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
