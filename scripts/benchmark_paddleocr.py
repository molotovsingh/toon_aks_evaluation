#!/usr/bin/env python3
"""
PaddleOCR vs Docling Benchmark Script

Comprehensive side-by-side comparison of PaddleOCR 3.x against Docling
for legal document extraction. Tests quality, performance, and memory usage.

Usage:
    # Quick test (1 PDF)
    uv run python scripts/benchmark_paddleocr.py --quick

    # Medium test (7 PDFs from famas_dispute)
    uv run python scripts/benchmark_paddleocr.py --medium

    # Full stress test (9 PDFs from amrapali_case)
    uv run python scripts/benchmark_paddleocr.py --stress

    # Custom file selection
    uv run python scripts/benchmark_paddleocr.py --files sample_pdf/famas_dispute/*.pdf

    # Compare specific engines
    uv run python scripts/benchmark_paddleocr.py --compare-engines docling,paddleocr --output benchmark_results/comparison.md
"""

import os
import sys
import time
import argparse
import logging
import tracemalloc
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# Check dependencies
try:
    # Only check if docling package exists - actual imports happen in adapter init
    import docling
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    print("⚠️  Docling not available - install with: uv add docling")

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    print("⚠️  PaddleOCR not available - install with: pip install paddleocr")

try:
    import fitz  # PyMuPDF for PDF page counting
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class ExtractionResult:
    """Results from a single document extraction"""
    engine: str
    file_path: Path
    success: bool
    error: Optional[str] = None

    # Quality metrics
    text_length: int = 0
    table_count: int = 0
    ocr_quality_score: float = 0.0  # 0.0-1.0
    has_dates: bool = False
    has_numbers: bool = False

    # Performance metrics
    extraction_time_sec: float = 0.0
    memory_peak_mb: float = 0.0
    pages_processed: int = 0

    # Raw outputs
    extracted_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Aggregate benchmark results across all test files"""
    docling_results: List[ExtractionResult] = field(default_factory=list)
    paddleocr_results: List[ExtractionResult] = field(default_factory=list)

    # Aggregate metrics
    docling_avg_time: float = 0.0
    paddleocr_avg_time: float = 0.0
    docling_avg_memory: float = 0.0
    paddleocr_avg_memory: float = 0.0
    docling_success_rate: float = 0.0
    paddleocr_success_rate: float = 0.0

    # Quality comparison
    text_quality_improvement: float = 0.0  # % improvement PaddleOCR vs Docling
    table_detection_improvement: float = 0.0
    speed_ratio: float = 0.0  # PaddleOCR time / Docling time (>1 = slower)

    # Decision metrics
    recommendation: str = ""  # "INTEGRATE", "CONDITIONAL", "REJECT"
    reasoning: str = ""


# ============================================================================
# ENGINE ADAPTERS
# ============================================================================

class DoclingAdapter:
    """Adapter for Docling document extraction"""

    def __init__(self):
        if not DOCLING_AVAILABLE:
            raise ImportError("Docling not installed")

        # Use existing DoclingDocumentExtractor for consistency with production
        try:
            import sys
            from pathlib import Path
            # Add src directory to path for imports
            project_root = Path(__file__).parent.parent
            sys.path.insert(0, str(project_root))

            from src.core.docling_adapter import DoclingDocumentExtractor
            from src.core.config import DoclingConfig

            # Create production configuration
            config = DoclingConfig()
            self.extractor = DoclingDocumentExtractor(config)

            logging.info("✅ Docling adapter initialized (using production DoclingDocumentExtractor)")
        except Exception as e:
            logging.error(f"Failed to initialize DoclingDocumentExtractor: {e}")
            raise

    def extract(self, file_path: Path) -> ExtractionResult:
        """Extract text using Docling"""
        result = ExtractionResult(
            engine="docling",
            file_path=file_path,
            success=False
        )

        try:
            tracemalloc.start()
            start_time = time.time()

            # Extract using production extractor
            extracted_doc = self.extractor.extract(file_path)

            extraction_time = time.time() - start_time
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Get extracted text (use plain_text for consistency)
            extracted_text = extracted_doc.plain_text

            # Collect metrics
            result.success = True
            result.extracted_text = extracted_text
            result.text_length = len(extracted_text)
            result.extraction_time_sec = extraction_time
            result.memory_peak_mb = peak / (1024 * 1024)
            result.pages_processed = extracted_doc.metadata.get("page_count", 0)
            result.metadata = extracted_doc.metadata

            # Quality indicators
            result.has_dates = bool(len([w for w in extracted_text.split() if any(c.isdigit() for c in w)]) > 5)
            result.has_numbers = any(c.isdigit() for c in extracted_text)

            # Table detection (count table markers in markdown)
            result.table_count = extracted_doc.markdown.count("|")  # Markdown table indicators

            logging.info(f"✅ Docling extracted {result.text_length} chars in {extraction_time:.2f}s")

        except Exception as e:
            result.success = False
            result.error = str(e)
            logging.error(f"❌ Docling extraction failed: {e}")

        return result


class PaddleOCRAdapter:
    """Adapter for PaddleOCR document extraction"""

    def __init__(self):
        if not PADDLEOCR_AVAILABLE:
            raise ImportError("PaddleOCR not installed")

        # Initialize PaddleOCR with optimal settings for English documents
        # Updated API parameters for PaddleOCR 3.x
        self.ocr = PaddleOCR(
            lang='en',  # English language
            use_textline_orientation=True,  # Enable text orientation detection (replaces use_angle_cls)
        )

        logging.info("✅ PaddleOCR adapter initialized")

    def extract(self, file_path: Path) -> ExtractionResult:
        """Extract text using PaddleOCR"""
        result = ExtractionResult(
            engine="paddleocr",
            file_path=file_path,
            success=False
        )

        try:
            tracemalloc.start()
            start_time = time.time()

            # PaddleOCR 3.x - use ocr() method (wraps predict())
            ocr_result = self.ocr.ocr(str(file_path))

            extraction_time = time.time() - start_time
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Parse OCR results
            extracted_lines = []
            table_indicators = 0

            if ocr_result:
                for page_result in ocr_result:
                    if page_result:
                        for line in page_result:
                            if line and len(line) >= 2:
                                text = line[1][0]  # Extract text from detection result
                                extracted_lines.append(text)

                                # Heuristic table detection (lines with multiple spaces/tabs)
                                if text.count("  ") >= 3 or text.count("\t") >= 2:
                                    table_indicators += 1

            extracted_text = "\n".join(extracted_lines)

            # Collect metrics
            result.success = True
            result.extracted_text = extracted_text
            result.text_length = len(extracted_text)
            result.extraction_time_sec = extraction_time
            result.memory_peak_mb = peak / (1024 * 1024)
            result.pages_processed = len(ocr_result) if ocr_result else 0
            result.table_count = table_indicators

            # Quality indicators
            result.has_dates = bool(len([w for w in extracted_text.split() if any(c.isdigit() for c in w)]) > 5)
            result.has_numbers = any(c.isdigit() for c in extracted_text)

            logging.info(f"✅ PaddleOCR extracted {result.text_length} chars in {extraction_time:.2f}s")

        except Exception as e:
            result.success = False
            result.error = str(e)
            logging.error(f"❌ PaddleOCR extraction failed: {e}")

        return result


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

class BenchmarkRunner:
    """Orchestrates benchmark tests across multiple files and engines"""

    def __init__(self, test_files: List[Path], engines: List[str] = None):
        self.test_files = test_files
        self.engines = engines or ["docling", "paddleocr"]
        self.results: Dict[str, List[ExtractionResult]] = {engine: [] for engine in self.engines}

        # Initialize adapters
        self.adapters = {}
        if "docling" in self.engines:
            self.adapters["docling"] = DoclingAdapter()
        if "paddleocr" in self.engines:
            self.adapters["paddleocr"] = PaddleOCRAdapter()

    def run(self) -> BenchmarkReport:
        """Run benchmark on all test files"""
        logging.info(f"🚀 Starting benchmark: {len(self.test_files)} files, {len(self.engines)} engines")

        for i, file_path in enumerate(self.test_files, 1):
            logging.info(f"\n📄 [{i}/{len(self.test_files)}] Testing: {file_path.name}")

            for engine in self.engines:
                adapter = self.adapters.get(engine)
                if not adapter:
                    logging.warning(f"⚠️  Engine '{engine}' not available, skipping")
                    continue

                result = adapter.extract(file_path)
                self.results[engine].append(result)

        # Generate report
        return self._generate_report()

    def _generate_report(self) -> BenchmarkReport:
        """Analyze results and generate recommendation"""
        report = BenchmarkReport(
            docling_results=self.results.get("docling", []),
            paddleocr_results=self.results.get("paddleocr", [])
        )

        # Calculate aggregate metrics for Docling
        if report.docling_results:
            successful_docling = [r for r in report.docling_results if r.success]
            if successful_docling:
                report.docling_avg_time = sum(r.extraction_time_sec for r in successful_docling) / len(successful_docling)
                report.docling_avg_memory = sum(r.memory_peak_mb for r in successful_docling) / len(successful_docling)
                report.docling_success_rate = len(successful_docling) / len(report.docling_results)

        # Calculate aggregate metrics for PaddleOCR
        if report.paddleocr_results:
            successful_paddle = [r for r in report.paddleocr_results if r.success]
            if successful_paddle:
                report.paddleocr_avg_time = sum(r.extraction_time_sec for r in successful_paddle) / len(successful_paddle)
                report.paddleocr_avg_memory = sum(r.memory_peak_mb for r in successful_paddle) / len(successful_paddle)
                report.paddleocr_success_rate = len(successful_paddle) / len(report.paddleocr_results)

        # Calculate quality improvements
        if report.docling_results and report.paddleocr_results:
            docling_avg_text_len = sum(r.text_length for r in report.docling_results if r.success) / max(1, len([r for r in report.docling_results if r.success]))
            paddle_avg_text_len = sum(r.text_length for r in report.paddleocr_results if r.success) / max(1, len([r for r in report.paddleocr_results if r.success]))

            if docling_avg_text_len > 0:
                report.text_quality_improvement = ((paddle_avg_text_len - docling_avg_text_len) / docling_avg_text_len) * 100

            # Speed ratio
            if report.docling_avg_time > 0:
                report.speed_ratio = report.paddleocr_avg_time / report.docling_avg_time

        # Generate recommendation
        report.recommendation, report.reasoning = self._make_recommendation(report)

        return report

    def _make_recommendation(self, report: BenchmarkReport) -> Tuple[str, str]:
        """
        Decision criteria (from CLAUDE.md):
        - ✅ INTEGRATE: ≥20% quality improvement, <30s per page, proven on user's PDFs
        - ⚠️ CONDITIONAL: 10-20% improvement, test on Hindi/multilingual first
        - ❌ REJECT: <10% improvement, >2x slower, >4GB memory, frequent errors
        """
        reasons = []

        # Check if PaddleOCR results exist
        if not report.paddleocr_results or len([r for r in report.paddleocr_results if r.success]) == 0:
            reasons.append("❌ PaddleOCR: No successful extractions (missing dependency or initialization failed)")
            reasons.append("\n🚫 RECOMMENDATION: Install PaddlePaddle first - pip install paddlepaddle paddleocr")
            return "REJECT", "\n".join(reasons)

        # Quality check
        if report.text_quality_improvement >= 20:
            reasons.append(f"✅ Quality: {report.text_quality_improvement:.1f}% improvement (≥20% threshold)")
            quality_pass = True
        elif report.text_quality_improvement >= 10:
            reasons.append(f"⚠️  Quality: {report.text_quality_improvement:.1f}% improvement (10-20% range)")
            quality_pass = "conditional"
        else:
            reasons.append(f"❌ Quality: {report.text_quality_improvement:.1f}% improvement (<10% threshold)")
            quality_pass = False

        # Speed check (per page)
        successful_paddle = [r for r in report.paddleocr_results if r.success]
        total_pages = sum(r.pages_processed for r in successful_paddle)
        avg_time_per_page = report.paddleocr_avg_time / max(1, total_pages / len(successful_paddle)) if total_pages > 0 else 0

        if avg_time_per_page > 0 and avg_time_per_page < 30:
            reasons.append(f"✅ Speed: {avg_time_per_page:.1f}s per page (<30s threshold)")
            speed_pass = True
        elif avg_time_per_page == 0:
            reasons.append(f"⚠️  Speed: Cannot calculate (no page count metadata)")
            speed_pass = True  # Don't fail on missing metadata
        else:
            reasons.append(f"❌ Speed: {avg_time_per_page:.1f}s per page (>30s threshold)")
            speed_pass = False

        # Memory check
        if report.paddleocr_avg_memory < 4096:
            reasons.append(f"✅ Memory: {report.paddleocr_avg_memory:.0f}MB peak (<4GB threshold)")
            memory_pass = True
        else:
            reasons.append(f"❌ Memory: {report.paddleocr_avg_memory:.0f}MB peak (>4GB threshold)")
            memory_pass = False

        # Success rate check
        if report.paddleocr_success_rate >= 0.9:
            reasons.append(f"✅ Reliability: {report.paddleocr_success_rate*100:.0f}% success rate")
            reliability_pass = True
        else:
            reasons.append(f"❌ Reliability: {report.paddleocr_success_rate*100:.0f}% success rate (<90%)")
            reliability_pass = False

        # Make final decision
        if quality_pass is True and speed_pass and memory_pass and reliability_pass:
            recommendation = "INTEGRATE"
            reasons.append("\n🎯 RECOMMENDATION: Proceed with PaddleOCR integration")
        elif quality_pass == "conditional" and speed_pass and memory_pass:
            recommendation = "CONDITIONAL"
            reasons.append("\n⚠️  RECOMMENDATION: Test on Hindi/multilingual documents before integration")
        else:
            recommendation = "REJECT"
            reasons.append("\n🚫 RECOMMENDATION: Do not integrate - Docling remains superior")

        return recommendation, "\n".join(reasons)


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_markdown_report(report: BenchmarkReport, output_path: Path) -> None:
    """Generate comprehensive markdown benchmark report"""

    md_lines = [
        "# PaddleOCR vs Docling Benchmark Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Test Files:** {len(report.docling_results)} documents",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        f"**Recommendation:** `{report.recommendation}`",
        "",
        "### Reasoning",
        "",
        report.reasoning,
        "",
        "---",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Docling | PaddleOCR | Winner |",
        "|--------|---------|-----------|--------|",
        f"| Avg Extraction Time | {report.docling_avg_time:.2f}s | {report.paddleocr_avg_time:.2f}s | {'🏆 Docling' if report.docling_avg_time < report.paddleocr_avg_time else '🏆 PaddleOCR'} |",
        f"| Avg Memory Usage | {report.docling_avg_memory:.0f}MB | {report.paddleocr_avg_memory:.0f}MB | {'🏆 Docling' if report.docling_avg_memory < report.paddleocr_avg_memory else '🏆 PaddleOCR'} |",
        f"| Success Rate | {report.docling_success_rate*100:.0f}% | {report.paddleocr_success_rate*100:.0f}% | {'🏆 Docling' if report.docling_success_rate > report.paddleocr_success_rate else '🏆 PaddleOCR'} |",
        f"| Text Quality Improvement | - | {report.text_quality_improvement:+.1f}% | {'🏆 PaddleOCR' if report.text_quality_improvement > 0 else '🏆 Docling'} |",
        f"| Speed Ratio | 1.0x | {report.speed_ratio:.2f}x | {'🏆 Docling' if report.speed_ratio > 1 else '🏆 PaddleOCR'} |",
        "",
        "---",
        "",
        "## Per-Document Results",
        "",
    ]

    # Docling results table
    md_lines.extend([
        "### Docling Extractions",
        "",
        "| Document | Status | Text Length | Time | Memory | Pages |",
        "|----------|--------|-------------|------|--------|-------|",
    ])

    for r in report.docling_results:
        status = "✅" if r.success else f"❌ {r.error[:30]}"
        md_lines.append(
            f"| {r.file_path.name} | {status} | {r.text_length:,} chars | {r.extraction_time_sec:.2f}s | {r.memory_peak_mb:.0f}MB | {r.pages_processed} |"
        )

    md_lines.extend(["", ""])

    # PaddleOCR results table
    md_lines.extend([
        "### PaddleOCR Extractions",
        "",
        "| Document | Status | Text Length | Time | Memory | Pages |",
        "|----------|--------|-------------|------|--------|-------|",
    ])

    for r in report.paddleocr_results:
        status = "✅" if r.success else f"❌ {r.error[:30]}"
        md_lines.append(
            f"| {r.file_path.name} | {status} | {r.text_length:,} chars | {r.extraction_time_sec:.2f}s | {r.memory_peak_mb:.0f}MB | {r.pages_processed} |"
        )

    md_lines.extend(["", "---", "", "## Next Steps", ""])

    if report.recommendation == "INTEGRATE":
        md_lines.extend([
            "1. ✅ Create `src/core/paddleocr_doc_adapter.py` implementing `DocumentExtractor` protocol",
            "2. ✅ Register in `src/core/extractor_factory.py` (`DOC_PROVIDER_REGISTRY['paddleocr']`)",
            "3. ✅ Add `PaddleOCRConfig` to `src/core/config.py`",
            "4. ✅ Update UI selector in `app.py`",
            "5. ✅ Add to `document_extractor_catalog.py` with pricing metadata",
        ])
    elif report.recommendation == "CONDITIONAL":
        md_lines.extend([
            "1. ⚠️  Test on Hindi/Devanagari documents first",
            "2. ⚠️  Compare table extraction quality on complex legal tables",
            "3. ⚠️  Re-run benchmark with multilingual test set",
            "4. ⚠️  If multilingual performance >20% better, proceed to integration",
        ])
    else:
        md_lines.extend([
            "1. 🚫 Do not integrate PaddleOCR",
            "2. 📝 Document findings in `docs/benchmark_reports/paddleocr_evaluation.md`",
            "3. ✅ Continue using Docling as primary document extractor",
        ])

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(md_lines))
    logging.info(f"📊 Report saved to: {output_path}")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def get_test_files(args) -> List[Path]:
    """Determine which test files to use based on CLI arguments"""
    project_root = Path(__file__).parent.parent

    if args.files:
        # Custom file selection
        return [Path(f) for f in args.files if Path(f).exists()]

    elif args.quick:
        # Quick test - 1 PDF from famas_dispute
        test_dir = project_root / "sample_pdf" / "famas_dispute"
        pdf_files = list(test_dir.glob("*.pdf"))
        return pdf_files[:1] if pdf_files else []

    elif args.medium:
        # Medium test - all PDFs from famas_dispute
        test_dir = project_root / "sample_pdf" / "famas_dispute"
        return list(test_dir.glob("*.pdf"))

    elif args.stress:
        # Stress test - all PDFs from amrapali_case
        test_dir = project_root / "sample_pdf" / "amrapali_case"
        return list(test_dir.glob("*.pdf"))

    else:
        # Default - medium test
        test_dir = project_root / "sample_pdf" / "famas_dispute"
        return list(test_dir.glob("*.pdf"))


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PaddleOCR vs Docling for legal document extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (1 PDF)
  uv run python scripts/benchmark_paddleocr.py --quick

  # Medium test (7 PDFs)
  uv run python scripts/benchmark_paddleocr.py --medium

  # Stress test (9 PDFs)
  uv run python scripts/benchmark_paddleocr.py --stress

  # Custom files
  uv run python scripts/benchmark_paddleocr.py --files sample_pdf/famas_dispute/*.pdf
        """
    )

    # Test selection
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument("--quick", action="store_true", help="Quick test (1 PDF)")
    test_group.add_argument("--medium", action="store_true", help="Medium test (7 PDFs from famas_dispute)")
    test_group.add_argument("--stress", action="store_true", help="Stress test (9 PDFs from amrapali_case)")
    test_group.add_argument("--files", nargs="+", help="Custom file paths")

    # Engine selection
    parser.add_argument(
        "--compare-engines",
        default="docling,paddleocr",
        help="Comma-separated list of engines to compare (default: docling,paddleocr)"
    )

    # Output
    parser.add_argument(
        "--output",
        default="benchmark_results/paddleocr_comparison.md",
        help="Output markdown report path (default: benchmark_results/paddleocr_comparison.md)"
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s"
    )

    # Check dependencies
    if not DOCLING_AVAILABLE:
        print("❌ Docling not installed. Install with: uv add docling")
        return 1

    if not PADDLEOCR_AVAILABLE:
        print("❌ PaddleOCR not installed. Install with: pip install paddleocr")
        return 1

    # Get test files
    test_files = get_test_files(args)
    if not test_files:
        print("❌ No test files found. Check file paths or use --quick/--medium/--stress")
        return 1

    print(f"\n📋 Benchmark Configuration:")
    print(f"   Files: {len(test_files)} PDFs")
    print(f"   Engines: {args.compare_engines}")
    print(f"   Output: {args.output}")
    print()

    # Run benchmark
    engines = [e.strip() for e in args.compare_engines.split(",")]
    runner = BenchmarkRunner(test_files, engines)
    report = runner.run()

    # Generate report
    output_path = Path(args.output)
    generate_markdown_report(report, output_path)

    # Print summary
    print(f"\n{'='*70}")
    print(f"📊 BENCHMARK COMPLETE")
    print(f"{'='*70}")
    print(f"\n{report.reasoning}")
    print(f"\n📄 Full report: {output_path}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
