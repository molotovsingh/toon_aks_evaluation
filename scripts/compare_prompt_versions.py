#!/usr/bin/env python3
"""
Prompt Version Comparison Test
Compares V1 (baseline) vs V2 (enhanced) extraction prompts on the same document
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.docling_adapter import DoclingDocumentExtractor
from src.core.config import load_config
from src.core.constants import LEGAL_EVENTS_PROMPT_V1, LEGAL_EVENTS_PROMPT_V2


class PromptComparison:
    """Compare V1 vs V2 extraction prompts"""

    def __init__(self, test_file: Path, provider: str = "openrouter"):
        self.test_file = test_file
        self.provider = provider
        self.results = {}

        # Results directory
        self.output_dir = Path(__file__).parent.parent / "test_results" / "prompt_comparison"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def extract_document_once(self) -> str:
        """Extract document text once (reuse for both prompts)"""
        print("📄 Extracting document text...")

        # Load config and create extractor
        docling_config, _, _ = load_config()
        extractor = DoclingDocumentExtractor(docling_config)

        # Extract
        start = time.perf_counter()
        result = extractor.extract(self.test_file)
        elapsed = time.perf_counter() - start

        print(f"   ✅ Extracted in {elapsed:.2f}s")
        print(f"   📊 Text length: {len(result.plain_text):,} chars")

        return result.plain_text

    def extract_events_with_prompt(self, text: str, prompt_version: str) -> Tuple[List[Dict], float]:
        """Extract events using specified prompt version via provider"""
        # Import here to avoid circular imports
        if self.provider == "openrouter":
            from src.core.openrouter_adapter import OpenRouterEventExtractor
            from src.core.config import OpenRouterConfig

            # Create config
            config = OpenRouterConfig(
                api_key=os.getenv("OPENROUTER_API_KEY", ""),
                model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
            )

            # Temporarily override the prompt in constants module
            import src.core.constants as constants
            original_prompt = constants.LEGAL_EVENTS_PROMPT

            try:
                # Set the prompt version
                if prompt_version == "v1":
                    constants.LEGAL_EVENTS_PROMPT = LEGAL_EVENTS_PROMPT_V1
                else:
                    constants.LEGAL_EVENTS_PROMPT = LEGAL_EVENTS_PROMPT_V2

                # Reload the adapter to pick up new prompt
                import importlib
                import src.core.openrouter_adapter
                importlib.reload(src.core.openrouter_adapter)
                from src.core.openrouter_adapter import OpenRouterEventExtractor

                # Create extractor
                extractor = OpenRouterEventExtractor(config)

                # Extract events
                start = time.perf_counter()
                event_records = extractor.extract_events(text, {"document_name": self.test_file.name})
                elapsed = time.perf_counter() - start

                # Convert to dicts
                events = [
                    {
                        "number": e.number,
                        "date": e.date,
                        "event_particulars": e.event_particulars,
                        "citation": e.citation,
                        "document_reference": e.document_reference
                    }
                    for e in event_records
                ]

                return events, elapsed

            finally:
                # Restore original prompt
                constants.LEGAL_EVENTS_PROMPT = original_prompt

        elif self.provider == "openai":
            from src.core.openai_adapter import OpenAIEventExtractor
            from src.core.config import OpenAIConfig

            config = OpenAIConfig(
                api_key=os.getenv("OPENAI_API_KEY", ""),
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            )

            import src.core.constants as constants
            original_prompt = constants.LEGAL_EVENTS_PROMPT

            try:
                if prompt_version == "v1":
                    constants.LEGAL_EVENTS_PROMPT = LEGAL_EVENTS_PROMPT_V1
                else:
                    constants.LEGAL_EVENTS_PROMPT = LEGAL_EVENTS_PROMPT_V2

                import importlib
                import src.core.openai_adapter
                importlib.reload(src.core.openai_adapter)
                from src.core.openai_adapter import OpenAIEventExtractor

                extractor = OpenAIEventExtractor(config)

                start = time.perf_counter()
                event_records = extractor.extract_events(text, {"document_name": self.test_file.name})
                elapsed = time.perf_counter() - start

                events = [
                    {
                        "number": e.number,
                        "date": e.date,
                        "event_particulars": e.event_particulars,
                        "citation": e.citation,
                        "document_reference": e.document_reference
                    }
                    for e in event_records
                ]

                return events, elapsed

            finally:
                constants.LEGAL_EVENTS_PROMPT = original_prompt

        else:
            raise ValueError(f"Provider {self.provider} not supported for comparison")

    def analyze_events(self, events: List[Dict]) -> Dict:
        """Analyze event characteristics"""
        if not events:
            return {
                "count": 0,
                "avg_particulars_length": 0,
                "dates_found": 0,
                "citations_found": 0,
                "date_coverage": []
            }

        # Calculate metrics
        particulars_lengths = [len(e["event_particulars"]) for e in events]
        dates_found = sum(1 for e in events if e["date"] and e["date"].strip())
        citations_found = sum(1 for e in events if e["citation"] and e["citation"].strip() and "No citation" not in e["citation"])

        # Get unique dates
        unique_dates = set()
        for e in events:
            if e["date"] and e["date"].strip():
                unique_dates.add(e["date"])

        return {
            "count": len(events),
            "avg_particulars_length": sum(particulars_lengths) / len(particulars_lengths) if particulars_lengths else 0,
            "dates_found": dates_found,
            "citations_found": citations_found,
            "date_coverage": sorted(list(unique_dates))
        }

    def print_comparison_table(self, v1_analysis: Dict, v2_analysis: Dict, v1_time: float, v2_time: float):
        """Print side-by-side comparison table"""
        print("\n" + "=" * 80)
        print("📊 PROMPT COMPARISON RESULTS")
        print("=" * 80)

        print(f"\n{'Metric':<40} {'V1 (Baseline)':<20} {'V2 (Enhanced)':<20}")
        print("-" * 80)

        # Event count
        v1_count = v1_analysis["count"]
        v2_count = v2_analysis["count"]
        count_diff = v2_count - v1_count
        count_symbol = "⬆️" if count_diff > 0 else "⬇️" if count_diff < 0 else "="
        print(f"{'Events Extracted':<40} {v1_count:<20} {v2_count:<20} {count_symbol} {abs(count_diff)}")

        # Extraction time
        time_diff = v2_time - v1_time
        time_symbol = "⬆️" if time_diff > 0 else "⬇️" if time_diff < 0 else "="
        print(f"{'Extraction Time (seconds)':<40} {v1_time:<20.2f} {v2_time:<20.2f} {time_symbol} {abs(time_diff):.2f}s")

        # Avg particulars length
        v1_len = v1_analysis["avg_particulars_length"]
        v2_len = v2_analysis["avg_particulars_length"]
        len_diff = v2_len - v1_len
        len_symbol = "⬆️" if len_diff > 0 else "⬇️" if len_diff < 0 else "="
        print(f"{'Avg Event Detail (chars)':<40} {v1_len:<20.0f} {v2_len:<20.0f} {len_symbol} {abs(len_diff):.0f}")

        # Dates found
        v1_dates = v1_analysis["dates_found"]
        v2_dates = v2_analysis["dates_found"]
        dates_diff = v2_dates - v1_dates
        dates_symbol = "⬆️" if dates_diff > 0 else "⬇️" if dates_diff < 0 else "="
        print(f"{'Events with Dates':<40} {v1_dates:<20} {v2_dates:<20} {dates_symbol} {abs(dates_diff)}")

        # Citations found
        v1_cites = v1_analysis["citations_found"]
        v2_cites = v2_analysis["citations_found"]
        cites_diff = v2_cites - v1_cites
        cites_symbol = "⬆️" if cites_diff > 0 else "⬇️" if cites_diff < 0 else "="
        print(f"{'Events with Citations':<40} {v1_cites:<20} {v2_cites:<20} {cites_symbol} {abs(cites_diff)}")

        print("\n" + "=" * 80)

    def print_sample_events(self, v1_events: List[Dict], v2_events: List[Dict]):
        """Print sample events from each version"""
        print("\n" + "=" * 80)
        print("📋 SAMPLE EVENTS (First 3)")
        print("=" * 80)

        # V1 samples
        print("\n🔵 V1 (Baseline) - Sample Events:")
        print("-" * 80)
        for i, event in enumerate(v1_events[:3], 1):
            print(f"\nEvent {i}:")
            print(f"  Date: {event['date'] or '(none)'}")
            print(f"  Particulars: {event['event_particulars'][:150]}..." if len(event['event_particulars']) > 150 else f"  Particulars: {event['event_particulars']}")
            print(f"  Citation: {event['citation'] or '(none)'}")

        # V2 samples
        print("\n\n🟢 V2 (Enhanced) - Sample Events:")
        print("-" * 80)
        for i, event in enumerate(v2_events[:3], 1):
            print(f"\nEvent {i}:")
            print(f"  Date: {event['date'] or '(none)'}")
            print(f"  Particulars: {event['event_particulars'][:150]}..." if len(event['event_particulars']) > 150 else f"  Particulars: {event['event_particulars']}")
            print(f"  Citation: {event['citation'] or '(none)'}")

        print("\n" + "=" * 80)

    def save_results(self, v1_events: List[Dict], v2_events: List[Dict],
                    v1_analysis: Dict, v2_analysis: Dict,
                    v1_time: float, v2_time: float):
        """Save comparison results to files"""

        # JSON results
        results = {
            "test_file": str(self.test_file),
            "provider": self.provider,
            "timestamp": self.timestamp,
            "v1": {
                "events": v1_events,
                "analysis": v1_analysis,
                "extraction_time": v1_time
            },
            "v2": {
                "events": v2_events,
                "analysis": v2_analysis,
                "extraction_time": v2_time
            },
            "comparison": {
                "event_count_diff": v2_analysis["count"] - v1_analysis["count"],
                "time_diff": v2_time - v1_time,
                "avg_length_diff": v2_analysis["avg_particulars_length"] - v1_analysis["avg_particulars_length"],
                "dates_diff": v2_analysis["dates_found"] - v1_analysis["dates_found"],
                "citations_diff": v2_analysis["citations_found"] - v1_analysis["citations_found"]
            }
        }

        json_file = self.output_dir / f"comparison_{self.timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 JSON results saved to: {json_file}")

        # CSV for Excel analysis
        csv_data = []
        for version, events in [("V1", v1_events), ("V2", v2_events)]:
            for event in events:
                csv_data.append({
                    "Version": version,
                    "Number": event["number"],
                    "Date": event["date"],
                    "Event_Particulars": event["event_particulars"],
                    "Citation": event["citation"],
                    "Particulars_Length": len(event["event_particulars"])
                })

        df = pd.DataFrame(csv_data)
        csv_file = self.output_dir / f"comparison_{self.timestamp}.csv"
        df.to_csv(csv_file, index=False)
        print(f"💾 CSV results saved to: {csv_file}")

        # Summary report
        report = f"""
# Prompt Version Comparison Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Test File: {self.test_file.name}
Provider: {self.provider}

## Summary

| Metric | V1 (Baseline) | V2 (Enhanced) | Difference |
|--------|---------------|---------------|------------|
| Events Extracted | {v1_analysis['count']} | {v2_analysis['count']} | {v2_analysis['count'] - v1_analysis['count']:+d} |
| Extraction Time | {v1_time:.2f}s | {v2_time:.2f}s | {v2_time - v1_time:+.2f}s |
| Avg Event Detail | {v1_analysis['avg_particulars_length']:.0f} chars | {v2_analysis['avg_particulars_length']:.0f} chars | {v2_analysis['avg_particulars_length'] - v1_analysis['avg_particulars_length']:+.0f} |
| Events with Dates | {v1_analysis['dates_found']} | {v2_analysis['dates_found']} | {v2_analysis['dates_found'] - v1_analysis['dates_found']:+d} |
| Events with Citations | {v1_analysis['citations_found']} | {v2_analysis['citations_found']} | {v2_analysis['citations_found'] - v1_analysis['citations_found']:+d} |

## Key Findings

### Event Count
{'✅ V2 extracted MORE events (+' + str(v2_analysis['count'] - v1_analysis['count']) + ')' if v2_analysis['count'] > v1_analysis['count'] else '⚠️ V2 extracted FEWER events (' + str(v2_analysis['count'] - v1_analysis['count']) + ')' if v2_analysis['count'] < v1_analysis['count'] else '= Same number of events'}

### Detail Quality
{'✅ V2 events are more detailed (+' + str(int(v2_analysis['avg_particulars_length'] - v1_analysis['avg_particulars_length'])) + ' chars avg)' if v2_analysis['avg_particulars_length'] > v1_analysis['avg_particulars_length'] else '⚠️ V2 events are less detailed (' + str(int(v2_analysis['avg_particulars_length'] - v1_analysis['avg_particulars_length'])) + ' chars avg)' if v2_analysis['avg_particulars_length'] < v1_analysis['avg_particulars_length'] else '= Similar detail level'}

### Date Coverage
V1 found {v1_analysis['dates_found']} dated events
V2 found {v2_analysis['dates_found']} dated events
{'✅ V2 has better date coverage' if v2_analysis['dates_found'] > v1_analysis['dates_found'] else '⚠️ V1 has better date coverage' if v1_analysis['dates_found'] > v2_analysis['dates_found'] else '= Same date coverage'}

## Conclusion

{'✅ **V2 Enhanced Prompt Shows Improvement**' if v2_analysis['count'] >= v1_analysis['count'] else '⚠️ **V2 May Need Tuning**'}

Full results: {json_file}
CSV data: {csv_file}
"""

        report_file = self.output_dir / f"report_{self.timestamp}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"💾 Report saved to: {report_file}")

    def run_comparison(self):
        """Run the full comparison test"""
        print("\n" + "=" * 80)
        print("🧪 PROMPT VERSION COMPARISON TEST")
        print("=" * 80)
        print(f"Test File: {self.test_file.name}")
        print(f"Provider: {self.provider}")
        print(f"Output: {self.output_dir}")
        print("=" * 80)

        # Step 1: Extract document once
        text = self.extract_document_once()

        # Step 2: Extract events with V1
        print("\n🔵 Testing V1 (Baseline Prompt)...")
        v1_events, v1_time = self.extract_events_with_prompt(text, "v1")
        print(f"   ✅ Extracted {len(v1_events)} events in {v1_time:.2f}s")

        # Step 3: Extract events with V2
        print("\n🟢 Testing V2 (Enhanced Prompt)...")
        v2_events, v2_time = self.extract_events_with_prompt(text, "v2")
        print(f"   ✅ Extracted {len(v2_events)} events in {v2_time:.2f}s")

        # Step 4: Analyze results
        print("\n📊 Analyzing results...")
        v1_analysis = self.analyze_events(v1_events)
        v2_analysis = self.analyze_events(v2_events)

        # Step 5: Display comparison
        self.print_comparison_table(v1_analysis, v2_analysis, v1_time, v2_time)
        self.print_sample_events(v1_events, v2_events)

        # Step 6: Save results
        self.save_results(v1_events, v2_events, v1_analysis, v2_analysis, v1_time, v2_time)

        print("\n✅ Comparison test complete!")


def main():
    # Check for required API keys
    provider = os.getenv("TEST_PROVIDER", "openrouter")

    if provider == "openrouter" and not os.getenv("OPENROUTER_API_KEY"):
        print("❌ Error: OPENROUTER_API_KEY not set")
        print("   Set it in your .env file or use: export OPENROUTER_API_KEY=your_key")
        sys.exit(1)
    elif provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not set")
        print("   Set it in your .env file or use: export OPENAI_API_KEY=your_key")
        sys.exit(1)

    # Use Famas PDF (well-tested, 15 pages, medium complexity)
    test_file = Path("sample_pdf/famas_dispute/Answer to Request for Arbitration.pdf")

    if not test_file.exists():
        print(f"❌ Error: Test file not found: {test_file}")
        sys.exit(1)

    # Run comparison
    comparison = PromptComparison(test_file, provider=provider)
    comparison.run_comparison()


if __name__ == "__main__":
    main()
