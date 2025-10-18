#!/usr/bin/env python3
"""
List Available Event Extractors

Displays all event extraction providers from the catalog with their metadata.
Useful for discovering available providers and checking their status.

Usage:
    uv run python scripts/list_providers.py
    uv run python scripts/list_providers.py --enabled-only
    uv run python scripts/list_providers.py --recommended-only
    uv run python scripts/list_providers.py --supports-runtime-model
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.event_extractor_catalog import get_event_extractor_catalog


def format_provider_table(providers, show_details=False):
    """Format providers as ASCII table"""
    if not providers:
        return "No providers found matching criteria."

    # Table headers
    headers = ["Provider ID", "Display Name", "Enabled", "Runtime Model", "Recommended"]
    if show_details:
        headers.append("Notes")

    # Calculate column widths
    col_widths = [len(h) for h in headers]

    # Update widths based on data
    for provider in providers:
        col_widths[0] = max(col_widths[0], len(provider.provider_id))
        col_widths[1] = max(col_widths[1], len(provider.display_name))
        col_widths[2] = max(col_widths[2], len(str(provider.enabled)))
        col_widths[3] = max(col_widths[3], len(str(provider.supports_runtime_model)))
        col_widths[4] = max(col_widths[4], len(str(provider.recommended)))
        if show_details:
            # Limit notes to 60 chars for readability
            note_display = provider.notes[:60] + "..." if len(provider.notes) > 60 else provider.notes
            col_widths[5] = max(col_widths[5] if len(col_widths) > 5 else 0, len(note_display))

    # Build table
    lines = []

    # Header row
    header_row = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    lines.append(header_row)
    lines.append("-" * len(header_row))

    # Data rows
    for provider in providers:
        row_data = [
            provider.provider_id.ljust(col_widths[0]),
            provider.display_name.ljust(col_widths[1]),
            ("✓" if provider.enabled else "✗").ljust(col_widths[2]),
            ("✓" if provider.supports_runtime_model else "✗").ljust(col_widths[3]),
            ("✓" if provider.recommended else "").ljust(col_widths[4]),
        ]
        if show_details:
            note_display = provider.notes[:60] + "..." if len(provider.notes) > 60 else provider.notes
            row_data.append(note_display.ljust(col_widths[5]))

        lines.append(" | ".join(row_data))

    return "\n".join(lines)


def check_provider_status(provider_id: str) -> tuple[bool, str]:
    """Check if provider has API key configured"""
    api_key_map = {
        'langextract': ('GEMINI_API_KEY', 'https://aistudio.google.com/app/apikey'),
        'openrouter': ('OPENROUTER_API_KEY', 'https://openrouter.ai/keys'),
        'openai': ('OPENAI_API_KEY', 'https://platform.openai.com/api-keys'),
        'anthropic': ('ANTHROPIC_API_KEY', 'https://console.anthropic.com/'),
        'deepseek': ('DEEPSEEK_API_KEY', 'https://platform.deepseek.com/'),
        'opencode_zen': ('OPENCODEZEN_API_KEY', None)
    }

    if provider_id not in api_key_map:
        return False, "Unknown provider"

    env_var, url = api_key_map[provider_id]
    has_key = bool(os.getenv(env_var))

    if has_key:
        return True, f"✓ {env_var} configured"
    else:
        return False, f"✗ {env_var} not found" + (f" (get key: {url})" if url else "")


def main():
    parser = argparse.ArgumentParser(
        description="List available event extraction providers from catalog",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all providers
  uv run python scripts/list_providers.py

  # List only enabled providers
  uv run python scripts/list_providers.py --enabled-only

  # List recommended providers only
  uv run python scripts/list_providers.py --recommended-only

  # List providers with runtime model support
  uv run python scripts/list_providers.py --supports-runtime-model

  # Show detailed notes
  uv run python scripts/list_providers.py --details
        """
    )

    parser.add_argument(
        "--enabled-only",
        action="store_true",
        help="Show only enabled providers"
    )
    parser.add_argument(
        "--recommended-only",
        action="store_true",
        help="Show only recommended providers"
    )
    parser.add_argument(
        "--supports-runtime-model",
        action="store_true",
        help="Show only providers supporting runtime model selection"
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Show detailed notes for each provider"
    )
    parser.add_argument(
        "--check-credentials",
        action="store_true",
        help="Check API key configuration status for each provider"
    )

    args = parser.parse_args()

    # Get catalog
    catalog = get_event_extractor_catalog()

    # Apply filters
    filters = {}
    if args.enabled_only:
        filters['enabled'] = True
    if args.recommended_only:
        filters['recommended_only'] = True
    if args.supports_runtime_model:
        filters['supports_runtime_model'] = True

    providers = catalog.list_extractors(**filters)

    # Display results
    print("=" * 80)
    print("EVENT EXTRACTION PROVIDERS")
    print("=" * 80)
    print()

    if not providers:
        print("No providers found matching criteria.")
        return

    print(format_provider_table(providers, show_details=args.details))
    print()
    print(f"Total: {len(providers)} provider(s)")

    # Show credential status if requested
    if args.check_credentials:
        print()
        print("=" * 80)
        print("CREDENTIAL STATUS")
        print("=" * 80)
        print()

        for provider in providers:
            configured, status = check_provider_status(provider.provider_id)
            status_icon = "✓" if configured else "✗"
            print(f"{status_icon} {provider.display_name:20} {status}")

    # Show usage hints
    if not args.enabled_only and not args.recommended_only and not args.supports_runtime_model:
        print()
        print("Filters:")
        print("  --enabled-only           Show only enabled providers")
        print("  --recommended-only       Show only recommended providers")
        print("  --supports-runtime-model Show providers with multi-model selection")
        print("  --check-credentials      Check API key configuration")
        print("  --details                Show detailed notes")


if __name__ == "__main__":
    main()
