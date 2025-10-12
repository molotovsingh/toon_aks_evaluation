#!/usr/bin/env python3
"""
LangExtract V1 vs V2 Prompt Comparison Test
Validates whether V2 prompt improves Gemini event extraction quality
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

from src.core.constants import LEGAL_EVENTS_PROMPT_V1, LEGAL_EVENTS_PROMPT_V2


class LangExtractComparison:
    """Compare V1 vs V2 extraction prompts for LangExtract/Gemini"""

    def __init__(self, text_file: Path):
        self.text_file = text_file
        self.results = {}

        # Results directory
        self.output_dir = Path(__file__).parent.parent / "test_results" / "langextract_comparison"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Read text
        with open(text_file, 'r') as f:
            self.text = f.read()

    def extract_events_with_prompt(self, text: str, prompt_version: str) -> Tuple[List[Dict], float]:
        """Extract events using specified prompt version via LangExtract"""
        from src.core.langextract_adapter import LangExtractEventExtractor
        from src.core.config import LangExtractConfig

        # Check API key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set - required for LangExtract")

        # Create config (model_id is pulled from env by default, but can be overridden)
        config = LangExtractConfig(
            model_id=os.getenv("GEMINI_MODEL_ID", "gemini-2.0-flash")
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
            import src.core.langextract_adapter
            importlib.reload(src.core.langextract_adapter)
            from src.core.langextract_adapter import LangExtractEventExtractor

            # Create extractor
            extractor = LangExtractEventExtractor(config)

            # Extract events
            start = time.perf_counter()
            event_records = extractor.extract_events(text, {"document_name": self.text_file.name})
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
        print("📊 LANGEXTRACT V1 vs V2 COMPARISON")
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

        # Unique dates
        v1_unique = len(v1_analysis["date_coverage"])
        v2_unique = len(v2_analysis["date_coverage"])
        unique_diff = v2_unique - v1_unique
        unique_symbol = "⬆️" if unique_diff > 0 else "⬇️" if unique_diff < 0 else "="
        print(f"{'Unique Dates Covered':<40} {v1_unique:<20} {v2_unique:<20} {unique_symbol} {abs(unique_diff)}")

        # Citations found
        v1_cites = v1_analysis["citations_found"]
        v2_cites = v2_analysis["citations_found"]
        cites_diff = v2_cites - v1_cites
        cites_symbol = "⬆️" if cites_diff > 0 else "⬇️" if cites_diff < 0 else "="
        print(f"{'Events with Citations':<40} {v1_cites:<20} {v2_cites:<20} {cites_symbol} {abs(cites_diff)}")

        print("\n" + "=" * 80)

    def print_date_coverage(self, v1_analysis: Dict, v2_analysis: Dict):
        """Print date coverage comparison"""
        print("\n" + "=" * 80)
        print("📅 DATE COVERAGE COMPARISON")
        print("=" * 80)

        v1_dates = set(v1_analysis["date_coverage"])
        v2_dates = set(v2_analysis["date_coverage"])

        only_v1 = v1_dates - v2_dates
        only_v2 = v2_dates - v1_dates
        both = v1_dates & v2_dates

        print(f"\n✅ Dates found by BOTH versions ({len(both)}):")
        for date in sorted(both):
            print(f"   • {date}")

        if only_v1:
            print(f"\n🔵 Dates found ONLY by V1 ({len(only_v1)}):")
            for date in sorted(only_v1):
                print(f"   • {date}")

        if only_v2:
            print(f"\n🟢 Dates found ONLY by V2 ({len(only_v2)}):")
            for date in sorted(only_v2):
                print(f"   • {date}")

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
            particulars = event['event_particulars'][:200] + "..." if len(event['event_particulars']) > 200 else event['event_particulars']
            print(f"  Particulars: {particulars}")
            print(f"  Citation: {event['citation'] or '(none)'}")

        # V2 samples
        print("\n\n🟢 V2 (Enhanced) - Sample Events:")
        print("-" * 80)
        for i, event in enumerate(v2_events[:3], 1):
            print(f"\nEvent {i}:")
            print(f"  Date: {event['date'] or '(none)'}")
            particulars = event['event_particulars'][:200] + "..." if len(event['event_particulars']) > 200 else event['event_particulars']
            print(f"  Particulars: {particulars}")
            print(f"  Citation: {event['citation'] or '(none)'}")

        print("\n" + "=" * 80)

    def save_results(self, v1_events: List[Dict], v2_events: List[Dict],
                    v1_analysis: Dict, v2_analysis: Dict,
                    v1_time: float, v2_time: float):
        """Save comparison results to files"""

        # JSON results
        results = {
            "test_file": str(self.text_file),
            "provider": "langextract",
            "model": os.getenv("GEMINI_MODEL_ID", "gemini-2.0-flash"),
            "timestamp": self.timestamp,
            "text_length": len(self.text),
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
                "unique_dates_diff": len(v2_analysis["date_coverage"]) - len(v1_analysis["date_coverage"]),
                "citations_diff": v2_analysis["citations_found"] - v1_analysis["citations_found"]
            }
        }

        filename_base = self.text_file.stem
        json_file = self.output_dir / f"{filename_base}_{self.timestamp}.json"
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
        csv_file = self.output_dir / f"{filename_base}_{self.timestamp}.csv"
        df.to_csv(csv_file, index=False)
        print(f"💾 CSV results saved to: {csv_file}")

    def run_comparison(self):
        """Run the full comparison test"""
        print("\n" + "=" * 80)
        print("🧪 LANGEXTRACT V1 vs V2 PROMPT COMPARISON")
        print("=" * 80)
        print(f"Test File: {self.text_file.name}")
        print(f"Text Length: {len(self.text):,} chars")
        print(f"Model: {os.getenv('GEMINI_MODEL_ID', 'gemini-2.0-flash')}")
        print(f"Output: {self.output_dir}")
        print("=" * 80)

        # Step 1: Extract events with V1
        print("\n🔵 Testing V1 (Baseline Prompt)...")
        v1_events, v1_time = self.extract_events_with_prompt(self.text, "v1")
        print(f"   ✅ Extracted {len(v1_events)} events in {v1_time:.2f}s")

        # Step 2: Extract events with V2
        print("\n🟢 Testing V2 (Enhanced Prompt)...")
        v2_events, v2_time = self.extract_events_with_prompt(self.text, "v2")
        print(f"   ✅ Extracted {len(v2_events)} events in {v2_time:.2f}s")

        # Step 3: Analyze results
        print("\n📊 Analyzing results...")
        v1_analysis = self.analyze_events(v1_events)
        v2_analysis = self.analyze_events(v2_events)

        # Step 4: Display comparison
        self.print_comparison_table(v1_analysis, v2_analysis, v1_time, v2_time)
        self.print_date_coverage(v1_analysis, v2_analysis)
        self.print_sample_events(v1_events, v2_events)

        # Step 5: Save results
        self.save_results(v1_events, v2_events, v1_analysis, v2_analysis, v1_time, v2_time)

        # Step 6: Recommendation
        print("\n" + "=" * 80)
        print("💡 RECOMMENDATION")
        print("=" * 80)

        count_improvement = ((v2_analysis["count"] - v1_analysis["count"]) / v1_analysis["count"] * 100) if v1_analysis["count"] > 0 else 0
        date_improvement = len(v2_analysis["date_coverage"]) - len(v1_analysis["date_coverage"])

        if v2_analysis["count"] > v1_analysis["count"] and date_improvement > 0:
            print(f"✅ **V2 shows improvement**: {abs(count_improvement):.1f}% more events, {date_improvement} more unique dates")
            print("   Recommendation: Enable V2 by setting USE_ENHANCED_PROMPT=true in .env")
        elif v2_analysis["count"] < v1_analysis["count"]:
            print(f"⚠️  **V2 extracts fewer events**: {abs(count_improvement):.1f}% reduction")
            print("   This may be noise reduction (good) or under-extraction (bad)")
            print("   Recommendation: Review sample events to assess quality")
        else:
            print("➖ **Similar performance**: Both versions extract similar event counts")
            print("   Recommendation: Review date coverage and event quality to decide")

        print("=" * 80)
        print("\n✅ Comparison test complete!")


def main():
    # Check for required API key
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ Error: GEMINI_API_KEY not set")
        print("   Set it in your .env file or use: export GEMINI_API_KEY=your_key")
        sys.exit(1)

    # Use the contract dispute text file
    test_file = Path("test_documents/abc_xyz_contract_dispute.txt")

    if not test_file.exists():
        print(f"❌ Error: Test file not found: {test_file}")
        sys.exit(1)

    # Run comparison
    comparison = LangExtractComparison(test_file)
    comparison.run_comparison()


if __name__ == "__main__":
    main()
