#!/usr/bin/env python3
"""
ADICR - Automated Documentation Integrity and Coverage Report Generator

Detects drift between code reality and documentation promises by comparing:
- Provider registries in code vs documented provider lists
- Environment variables in config.py vs README/env.example
- Doc extractor options in code vs UI documentation

Usage:
    uv run python scripts/generate_adicr_report.py
    uv run python scripts/generate_adicr_report.py --refresh  # Force regeneration
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.adicr_helpers import (
    extract_providers_from_registry,
    extract_env_vars_from_config,
    find_text_in_file,
    fuzzy_match_heading,
    extract_markdown_section
)


class ADICRReport:
    """ADICR report generator"""

    def __init__(self, manifest_path: str):
        self.manifest_path = Path(manifest_path)
        self.manifest = self._load_manifest()
        self.discrepancies: List[Dict[str, Any]] = []
        self.timestamp = datetime.now().isoformat()

    def _load_manifest(self) -> dict:
        """Load and validate ADICR targets manifest"""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        with open(self.manifest_path, 'r') as f:
            manifest = json.load(f)

        # Basic validation
        if 'targets' not in manifest:
            raise ValueError("Manifest missing 'targets' section")

        return manifest

    def check_provider_parity(self):
        """Check if all providers in code appear in documentation"""
        providers_config = self.manifest['targets'].get('providers')
        if not providers_config:
            return

        # Extract providers from code
        source_file = str(project_root / providers_config['source_file'])
        registry_name = providers_config['source_pattern']

        try:
            inventory = extract_providers_from_registry(source_file, registry_name)
            code_providers = inventory.providers
        except Exception as e:
            self.discrepancies.append({
                'category': 'providers',
                'severity': 'error',
                'message': f"Failed to extract providers from {source_file}: {e}",
                'file': source_file
            })
            return

        # Check each documentation location
        for location in providers_config['expected_locations']:
            doc_file = str(project_root / location['file'])
            missing_providers = self._check_providers_in_file(
                doc_file,
                code_providers,
                location
            )

            if missing_providers:
                self.discrepancies.append({
                    'category': 'providers',
                    'severity': 'critical',
                    'file': location['file'],
                    'section': location.get('section', 'N/A'),
                    'expected': sorted(code_providers),
                    'missing': sorted(missing_providers),
                    'description': f"Missing {len(missing_providers)} provider(s) in {location['description']}"
                })

    def _check_providers_in_file(
        self,
        file_path: str,
        providers: Set[str],
        location: dict
    ) -> Set[str]:
        """Check which providers are missing from a documentation file"""
        if not Path(file_path).exists():
            return providers  # All missing if file doesn't exist

        missing = set()

        for provider in providers:
            # Search for provider name (case-insensitive)
            matches = find_text_in_file(file_path, provider, case_sensitive=False)

            if not matches:
                missing.add(provider)

        return missing

    def check_env_var_coverage(self):
        """Check if all environment variables are documented"""
        env_config = self.manifest['targets'].get('env_vars')
        if not env_config:
            return

        # Extract env vars from code
        source_file = str(project_root / env_config['source_file'])

        try:
            env_vars = extract_env_vars_from_config(source_file)
            code_env_vars = {var.var_name for var in env_vars}
        except Exception as e:
            self.discrepancies.append({
                'category': 'env_vars',
                'severity': 'error',
                'message': f"Failed to extract env vars from {source_file}: {e}",
                'file': source_file
            })
            return

        # Check each documentation location
        for location in env_config['expected_locations']:
            doc_file = str(project_root / location['file'])
            missing_vars = self._check_env_vars_in_file(
                doc_file,
                code_env_vars,
                location
            )

            if missing_vars:
                self.discrepancies.append({
                    'category': 'env_vars',
                    'severity': 'critical',
                    'file': location['file'],
                    'section': location.get('section', 'N/A'),
                    'expected_count': len(code_env_vars),
                    'missing_count': len(missing_vars),
                    'missing_vars': sorted(missing_vars)[:10],  # Show first 10
                    'total_missing': len(missing_vars),
                    'description': f"Missing {len(missing_vars)} env var(s) in {location['description']}"
                })

    def _check_env_vars_in_file(
        self,
        file_path: str,
        env_vars: Set[str],
        location: dict
    ) -> Set[str]:
        """Check which environment variables are missing from a documentation file"""
        if not Path(file_path).exists():
            return env_vars  # All missing if file doesn't exist

        missing = set()

        for var in env_vars:
            # Search for env var name (case-sensitive for env vars)
            matches = find_text_in_file(file_path, var, case_sensitive=True)

            if not matches:
                missing.add(var)

        return missing

    def check_doc_extractor_parity(self):
        """Check if all doc extractors in code appear in documentation"""
        doc_config = self.manifest['targets'].get('doc_extractors')
        if not doc_config:
            return

        # Extract doc extractors from code
        source_file = str(project_root / doc_config['source_file'])
        registry_name = doc_config['source_pattern']

        try:
            inventory = extract_providers_from_registry(source_file, registry_name)
            code_extractors = inventory.providers
        except Exception as e:
            self.discrepancies.append({
                'category': 'doc_extractors',
                'severity': 'warning',
                'message': f"Failed to extract doc extractors from {source_file}: {e}",
                'file': source_file
            })
            return

        # Check each documentation location
        for location in doc_config['expected_locations']:
            doc_file = str(project_root / location['file'])
            missing_extractors = self._check_providers_in_file(
                doc_file,
                code_extractors,
                location
            )

            if missing_extractors:
                self.discrepancies.append({
                    'category': 'doc_extractors',
                    'severity': 'warning',
                    'file': location['file'],
                    'section': location.get('section', 'N/A'),
                    'expected': sorted(code_extractors),
                    'missing': sorted(missing_extractors),
                    'description': f"Missing {len(missing_extractors)} doc extractor(s) in {location['description']}"
                })

    def generate_markdown_report(self) -> str:
        """Generate human-readable markdown report"""
        lines = []
        lines.append(f"# ADICR Report - {datetime.now().strftime('%Y-%m-%d')}")
        lines.append("")
        lines.append("**Automated Documentation Integrity and Coverage Report**")
        lines.append("")
        lines.append(f"Generated: {self.timestamp}")
        lines.append(f"Manifest: {self.manifest_path.name}")
        lines.append("")

        # Summary
        critical_count = sum(1 for d in self.discrepancies if d['severity'] == 'critical')
        warning_count = sum(1 for d in self.discrepancies if d['severity'] == 'warning')
        error_count = sum(1 for d in self.discrepancies if d['severity'] == 'error')

        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Critical Issues**: {critical_count}")
        lines.append(f"- **Warnings**: {warning_count}")
        lines.append(f"- **Errors**: {error_count}")
        lines.append(f"- **Total Discrepancies**: {len(self.discrepancies)}")
        lines.append("")

        if not self.discrepancies:
            lines.append("✅ **All checks passed!** Documentation is in sync with code.")
            lines.append("")
        else:
            lines.append("⚠️ **Drift detected** - Documentation updates needed.")
            lines.append("")

        # Provider Parity
        provider_issues = [d for d in self.discrepancies if d['category'] == 'providers']
        if provider_issues:
            lines.append("## Provider Parity Check")
            lines.append("")
            for issue in provider_issues:
                severity_icon = "❌" if issue['severity'] == 'critical' else "⚠️"
                lines.append(f"{severity_icon} **{issue['file']}** - {issue.get('section', 'N/A')}")
                lines.append(f"  - {issue['description']}")
                if 'missing' in issue:
                    lines.append(f"  - Missing: {', '.join(issue['missing'])}")
                lines.append("")

        # Environment Variable Coverage
        env_issues = [d for d in self.discrepancies if d['category'] == 'env_vars']
        if env_issues:
            lines.append("## Environment Variable Coverage")
            lines.append("")
            for issue in env_issues:
                severity_icon = "❌" if issue['severity'] == 'critical' else "⚠️"
                lines.append(f"{severity_icon} **{issue['file']}** - {issue.get('section', 'N/A')}")
                lines.append(f"  - {issue['description']}")
                if 'missing_vars' in issue:
                    lines.append(f"  - Sample missing vars: {', '.join(issue['missing_vars'])}")
                    if issue['total_missing'] > 10:
                        lines.append(f"  - ... and {issue['total_missing'] - 10} more")
                lines.append("")

        # Doc Extractor Parity
        doc_issues = [d for d in self.discrepancies if d['category'] == 'doc_extractors']
        if doc_issues:
            lines.append("## Document Extractor Parity")
            lines.append("")
            for issue in doc_issues:
                severity_icon = "⚠️"  # Always warning
                lines.append(f"{severity_icon} **{issue['file']}** - {issue.get('section', 'N/A')}")
                lines.append(f"  - {issue['description']}")
                if 'missing' in issue:
                    lines.append(f"  - Missing: {', '.join(issue['missing'])}")
                lines.append("")

        # Errors
        error_issues = [d for d in self.discrepancies if d['severity'] == 'error']
        if error_issues:
            lines.append("## Errors")
            lines.append("")
            for issue in error_issues:
                lines.append(f"❌ **{issue['category']}**: {issue['message']}")
                lines.append("")

        # Remediation
        if self.discrepancies:
            lines.append("## Remediation Checklist")
            lines.append("")
            for idx, issue in enumerate(self.discrepancies, 1):
                if issue['severity'] in ('critical', 'warning'):
                    lines.append(f"- [ ] {issue['description']} ({issue['file']})")
            lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("*Generated by ADICR - Automated Documentation Integrity and Coverage Report*")
        lines.append("")

        return '\n'.join(lines)

    def generate_json_report(self) -> dict:
        """Generate machine-readable JSON report"""
        return {
            'timestamp': self.timestamp,
            'version': self.manifest.get('version', '1.0'),
            'manifest_path': str(self.manifest_path),
            'discrepancies': self.discrepancies,
            'summary': {
                'total_critical': sum(1 for d in self.discrepancies if d['severity'] == 'critical'),
                'total_warnings': sum(1 for d in self.discrepancies if d['severity'] == 'warning'),
                'total_errors': sum(1 for d in self.discrepancies if d['severity'] == 'error'),
                'total_discrepancies': len(self.discrepancies),
                'status': 'drift_detected' if self.discrepancies else 'synchronized'
            }
        }

    def run_all_checks(self):
        """Run all ADICR validation checks"""
        print("🔍 Running ADICR checks...")
        print("")

        print("  • Checking provider parity...")
        self.check_provider_parity()

        print("  • Checking environment variable coverage...")
        self.check_env_var_coverage()

        print("  • Checking document extractor parity...")
        self.check_doc_extractor_parity()

        print("")
        print(f"✅ Checks complete: {len(self.discrepancies)} discrepancies found")
        print("")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Generate ADICR (Automated Documentation Integrity and Coverage Report)"
    )
    parser.add_argument(
        '--refresh',
        action='store_true',
        help="Force regeneration of reports"
    )
    parser.add_argument(
        '--manifest',
        default="config/adicr_targets.json",
        help="Path to ADICR manifest file"
    )
    parser.add_argument(
        '--output-dir',
        default="docs/reports",
        help="Output directory for markdown report"
    )
    parser.add_argument(
        '--json-dir',
        default="output/adicr",
        help="Output directory for JSON report"
    )

    args = parser.parse_args()

    # Create output directories
    output_dir = project_root / args.output_dir
    json_dir = project_root / args.json_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    # Initialize reporter
    try:
        reporter = ADICRReport(project_root / args.manifest)
    except Exception as e:
        print(f"❌ Error loading manifest: {e}", file=sys.stderr)
        return 1

    # Run checks
    reporter.run_all_checks()

    # Generate reports
    markdown_report = reporter.generate_markdown_report()
    json_report = reporter.generate_json_report()

    # Save markdown report
    md_path = output_dir / "adicr-latest.md"
    with open(md_path, 'w') as f:
        f.write(markdown_report)
    print(f"📄 Markdown report: {md_path}")

    # Save JSON report
    json_path = json_dir / "adicr_report.json"
    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2)
    print(f"📊 JSON report: {json_path}")

    # Print summary
    print("")
    print("Summary:")
    print(f"  - Critical: {json_report['summary']['total_critical']}")
    print(f"  - Warnings: {json_report['summary']['total_warnings']}")
    print(f"  - Errors: {json_report['summary']['total_errors']}")
    print(f"  - Status: {json_report['summary']['status']}")
    print("")

    # Exit with status code
    if json_report['summary']['total_critical'] > 0:
        print("⚠️ Critical issues found - documentation updates needed")
        return 1
    elif json_report['summary']['total_warnings'] > 0:
        print("⚠️ Warnings found - consider updating documentation")
        return 0
    else:
        print("✅ All documentation in sync!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
