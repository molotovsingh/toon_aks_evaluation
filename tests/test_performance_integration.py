#!/usr/bin/env python3
"""
Performance and Integration Test Suite
Tests end-to-end functionality, performance, and data integrity for 5-column table

Run with: GOOGLE_API_KEY=your_key uv run python tests/test_performance_integration.py
          or: pytest tests/test_performance_integration.py (skips if no API key)
"""

import os
import sys
import time
import pandas as pd
import logging
import pytest
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.legal_pipeline_refactored import LegalEventsPipeline
from src.core.constants import FIVE_COLUMN_HEADERS, DEFAULT_NO_DATE
from src.ui.streamlit_common import get_pipeline
from src.core.table_formatter import TableFormatter

logger = logging.getLogger(__name__)

# Skip all integration tests if API key not available
pytestmark = pytest.mark.skipif(
    not (os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')),
    reason="Integration tests require GOOGLE_API_KEY or GEMINI_API_KEY environment variable"
)


class PerformanceIntegrationTests:
    """Performance and integration test suite for 5-column legal events system"""

    def __init__(self):
        self.test_results = []
        self.pipeline = None
        self.test_documents_dir = Path(__file__).parent / "test_documents"
        self.setup_logging()

    def setup_logging(self):
        """Configure logging for test output"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def log_performance_result(self, test_name: str, execution_time: float,
                             success: bool, details: Dict[str, Any] = None):
        """Log performance test results"""
        status = "✅ PASS" if success else "❌ FAIL"
        result = {
            "test_name": test_name,
            "execution_time": execution_time,
            "success": success,
            "details": details or {},
            "timestamp": time.time()
        }
        self.test_results.append(result)
        print(f"{status} {test_name}: {execution_time:.2f}s")
        if details:
            for key, value in details.items():
                print(f"   {key}: {value}")

    def create_mock_uploaded_file(self, file_path: Path):
        """Create mock uploaded file from test document"""
        class MockUploadedFile:
            def __init__(self, name, content):
                self.name = name
                self._content = content

            def getbuffer(self):
                return self._content.encode() if isinstance(self._content, str) else self._content

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return MockUploadedFile(file_path.name, content)

    def test_single_document_performance(self):
        """Test performance with single document processing"""
        test_name = "Single Document Processing Performance"

        try:
            # Use clear dates document for consistent results
            doc_path = self.test_documents_dir / "clear_dates_document.html"
            if not doc_path.exists():
                self.log_performance_result(test_name, 0.0, False,
                                          {"error": "Test document not found"})
                return

            mock_file = self.create_mock_uploaded_file(doc_path)

            # Measure processing time
            start_time = time.time()
            df, warning = self.pipeline.process_documents_for_legal_events([mock_file])
            end_time = time.time()

            execution_time = end_time - start_time

            # Performance criteria: should complete within 30 seconds
            success = execution_time <= 30.0 and df is not None and not df.empty

            details = {
                "events_extracted": len(df) if df is not None else 0,
                "columns": list(df.columns) if df is not None else [],
                "format_valid": self.pipeline.validate_five_column_format(df) if df is not None else False,
                "warning": warning
            }

            self.log_performance_result(test_name, execution_time, success, details)

        except Exception as e:
            self.log_performance_result(test_name, 0.0, False, {"error": str(e)})

    def test_multiple_documents_performance(self):
        """Test performance with multiple document processing"""
        test_name = "Multiple Documents Processing Performance"

        try:
            # Process all available test documents
            test_files = []
            for doc_file in self.test_documents_dir.glob("*.html"):
                test_files.append(self.create_mock_uploaded_file(doc_file))

            if not test_files:
                self.log_performance_result(test_name, 0.0, False,
                                          {"error": "No test documents found"})
                return

            # Measure processing time
            start_time = time.time()
            df, warning = self.pipeline.process_documents_for_legal_events(test_files)
            end_time = time.time()

            execution_time = end_time - start_time

            # Performance criteria: should complete within 60 seconds for multiple docs
            success = execution_time <= 60.0 and df is not None and not df.empty

            details = {
                "documents_processed": len(test_files),
                "total_events": len(df) if df is not None else 0,
                "avg_time_per_doc": execution_time / len(test_files) if test_files else 0,
                "format_valid": self.pipeline.validate_five_column_format(df) if df is not None else False,
                "unique_documents": df[FIVE_COLUMN_HEADERS[4]].nunique() if df is not None else 0
            }

            self.log_performance_result(test_name, execution_time, success, details)

        except Exception as e:
            self.log_performance_result(test_name, 0.0, False, {"error": str(e)})

    def test_date_extraction_accuracy(self):
        """Test accuracy of date extraction across different document types"""
        test_name = "Date Extraction Accuracy Test"

        try:
            # Test with clear dates document
            doc_path = self.test_documents_dir / "clear_dates_document.html"
            if not doc_path.exists():
                self.log_performance_result(test_name, 0.0, False,
                                          {"error": "Clear dates test document not found"})
                return

            mock_file = self.create_mock_uploaded_file(doc_path)

            start_time = time.time()
            df, warning = self.pipeline.process_documents_for_legal_events([mock_file])
            end_time = time.time()

            execution_time = end_time - start_time

            if df is not None and not df.empty:
                # Analyze date extraction results
                date_column = df[FIVE_COLUMN_HEADERS[1]]  # Date column
                total_events = len(df)
                valid_dates = len(date_column[date_column != DEFAULT_NO_DATE])
                extraction_rate = (valid_dates / total_events) * 100 if total_events > 0 else 0

                # Success if >50% of events have valid dates (clear dates doc should have high rate)
                success = extraction_rate >= 50.0

                details = {
                    "total_events": total_events,
                    "valid_dates": valid_dates,
                    "extraction_rate_percent": round(extraction_rate, 1),
                    "sample_dates": list(date_column[date_column != DEFAULT_NO_DATE].head(3)),
                    "fallback_count": len(date_column[date_column == DEFAULT_NO_DATE])
                }

                self.log_performance_result(test_name, execution_time, success, details)
            else:
                self.log_performance_result(test_name, execution_time, False,
                                          {"error": "No data generated"})

        except Exception as e:
            self.log_performance_result(test_name, 0.0, False, {"error": str(e)})

    def test_mixed_date_formats_handling(self):
        """Test handling of various date formats"""
        test_name = "Mixed Date Formats Handling Test"

        try:
            # Test with mixed formats document
            doc_path = self.test_documents_dir / "mixed_date_formats_document.html"
            if not doc_path.exists():
                self.log_performance_result(test_name, 0.0, False,
                                          {"error": "Mixed date formats test document not found"})
                return

            mock_file = self.create_mock_uploaded_file(doc_path)

            start_time = time.time()
            df, warning = self.pipeline.process_documents_for_legal_events([mock_file])
            end_time = time.time()

            execution_time = end_time - start_time

            if df is not None and not df.empty:
                date_column = df[FIVE_COLUMN_HEADERS[1]]
                valid_dates = date_column[date_column != DEFAULT_NO_DATE]

                # Analyze date format diversity
                format_patterns = {
                    "iso_format": len([d for d in valid_dates if "-" in str(d) and len(str(d)) == 10]),
                    "us_format": len([d for d in valid_dates if "/" in str(d)]),
                    "other_formats": len([d for d in valid_dates if d not in [DEFAULT_NO_DATE]])
                }

                success = len(valid_dates) > 0  # Should extract some dates from mixed format doc

                details = {
                    "total_events": len(df),
                    "valid_dates_extracted": len(valid_dates),
                    "format_patterns": format_patterns,
                    "sample_extracted_dates": list(valid_dates.head(5))
                }

                self.log_performance_result(test_name, execution_time, success, details)
            else:
                self.log_performance_result(test_name, execution_time, False,
                                          {"error": "No data generated"})

        except Exception as e:
            self.log_performance_result(test_name, 0.0, False, {"error": str(e)})

    def test_export_functionality(self):
        """Test export functionality with 5-column format"""
        test_name = "Export Functionality Test"

        try:
            # Generate test data
            doc_path = self.test_documents_dir / "clear_dates_document.html"
            if not doc_path.exists():
                self.log_performance_result(test_name, 0.0, False,
                                          {"error": "Test document not found"})
                return

            mock_file = self.create_mock_uploaded_file(doc_path)
            df, warning = self.pipeline.process_documents_for_legal_events([mock_file])

            if df is None or df.empty:
                self.log_performance_result(test_name, 0.0, False,
                                          {"error": "No data to export"})
                return

            start_time = time.time()

            # Test Excel export
            excel_data = self.pipeline.export_dataframe(df, "xlsx")

            # Test CSV export
            csv_data = self.pipeline.export_dataframe(df, "csv")

            end_time = time.time()
            execution_time = end_time - start_time

            # Validate exports
            success = (excel_data is not None and len(excel_data) > 0 and
                      csv_data is not None and len(csv_data) > 0)

            details = {
                "excel_size_bytes": len(excel_data) if excel_data else 0,
                "csv_size_bytes": len(csv_data) if csv_data else 0,
                "source_dataframe_shape": df.shape,
                "columns_exported": list(df.columns)
            }

            self.log_performance_result(test_name, execution_time, success, details)

        except Exception as e:
            self.log_performance_result(test_name, 0.0, False, {"error": str(e)})

    def test_data_consistency(self):
        """Test data consistency across multiple runs"""
        test_name = "Data Consistency Test"

        try:
            doc_path = self.test_documents_dir / "clear_dates_document.html"
            if not doc_path.exists():
                self.log_performance_result(test_name, 0.0, False,
                                          {"error": "Test document not found"})
                return

            # Run processing multiple times
            results = []
            total_time = 0

            for run in range(3):  # Test 3 runs for consistency
                mock_file = self.create_mock_uploaded_file(doc_path)

                start_time = time.time()
                df, warning = self.pipeline.process_documents_for_legal_events([mock_file])
                end_time = time.time()

                total_time += (end_time - start_time)

                if df is not None:
                    results.append({
                        "run": run + 1,
                        "event_count": len(df),
                        "date_extractions": len(df[df[FIVE_COLUMN_HEADERS[1]] != DEFAULT_NO_DATE]),
                        "columns": list(df.columns)
                    })

            avg_time = total_time / len(results) if results else 0

            # Check consistency
            if len(results) >= 2:
                event_counts = [r["event_count"] for r in results]
                date_counts = [r["date_extractions"] for r in results]
                column_sets = [tuple(r["columns"]) for r in results]

                # Consistent if all runs produce same structure
                consistent_events = len(set(event_counts)) == 1
                consistent_dates = len(set(date_counts)) == 1
                consistent_columns = len(set(column_sets)) == 1

                success = consistent_events and consistent_columns

                details = {
                    "runs_completed": len(results),
                    "event_counts": event_counts,
                    "date_extraction_counts": date_counts,
                    "consistent_events": consistent_events,
                    "consistent_dates": consistent_dates,
                    "consistent_columns": consistent_columns
                }

                self.log_performance_result(test_name, avg_time, success, details)
            else:
                self.log_performance_result(test_name, avg_time, False,
                                          {"error": "Insufficient runs completed"})

        except Exception as e:
            self.log_performance_result(test_name, 0.0, False, {"error": str(e)})

    def test_memory_usage(self):
        """Test memory usage during processing"""
        test_name = "Memory Usage Test"

        try:
            import psutil
            import os

            # Get initial memory usage
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Process multiple documents
            test_files = []
            for doc_file in self.test_documents_dir.glob("*.html"):
                test_files.append(self.create_mock_uploaded_file(doc_file))

            start_time = time.time()
            df, warning = self.pipeline.process_documents_for_legal_events(test_files)
            end_time = time.time()

            # Get peak memory usage
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - initial_memory

            execution_time = end_time - start_time

            # Success if memory increase is reasonable (< 500MB for test docs)
            success = memory_increase < 500.0 and df is not None

            details = {
                "initial_memory_mb": round(initial_memory, 1),
                "peak_memory_mb": round(peak_memory, 1),
                "memory_increase_mb": round(memory_increase, 1),
                "documents_processed": len(test_files),
                "events_generated": len(df) if df is not None else 0
            }

            self.log_performance_result(test_name, execution_time, success, details)

        except ImportError:
            self.log_performance_result(test_name, 0.0, False,
                                      {"error": "psutil not available for memory testing"})
        except Exception as e:
            self.log_performance_result(test_name, 0.0, False, {"error": str(e)})

    def run_all_tests(self):
        """Execute all performance and integration tests"""
        print("🚀 Starting Performance and Integration Test Suite")
        print("=" * 70)

        # Initialize pipeline
        try:
            self.pipeline = LegalEventsPipeline()
            print("✅ Pipeline initialized successfully")
        except Exception as e:
            print(f"❌ CRITICAL: Could not initialize pipeline: {e}")
            return False

        # Run tests
        print("\n📊 PERFORMANCE TESTS")
        print("-" * 50)
        self.test_single_document_performance()
        self.test_multiple_documents_performance()
        self.test_memory_usage()

        print("\n🔍 FUNCTIONALITY TESTS")
        print("-" * 50)
        self.test_date_extraction_accuracy()
        self.test_mixed_date_formats_handling()
        self.test_export_functionality()
        self.test_data_consistency()

        # Generate report
        self.generate_performance_report()
        return True

    def generate_performance_report(self):
        """Generate comprehensive performance test report"""
        print("\n" + "=" * 70)
        print("🏁 PERFORMANCE TEST RESULTS SUMMARY")
        print("=" * 70)

        if not self.test_results:
            print("❌ No test results to report")
            return

        # Overall statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r["success"])
        failed_tests = total_tests - passed_tests

        print(f"\n📈 OVERALL RESULTS:")
        print(f"   Total Tests: {total_tests}")
        print(f"   ✅ Passed: {passed_tests}")
        print(f"   ❌ Failed: {failed_tests}")
        print(f"   Success Rate: {(passed_tests/total_tests)*100:.1f}%")

        # Performance statistics
        successful_times = [r["execution_time"] for r in self.test_results if r["success"]]
        if successful_times:
            print(f"\n⏱️  PERFORMANCE METRICS:")
            print(f"   Average Execution Time: {sum(successful_times)/len(successful_times):.2f}s")
            print(f"   Fastest Test: {min(successful_times):.2f}s")
            print(f"   Slowest Test: {max(successful_times):.2f}s")

        # Failed tests details
        if failed_tests > 0:
            print(f"\n❌ FAILED TESTS:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"   • {result['test_name']}")
                    if "error" in result["details"]:
                        print(f"     Error: {result['details']['error']}")

        print("\n" + "=" * 70)


def main():
    """Main test execution"""
    # Check environment
    api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GOOGLE_API_KEY or GEMINI_API_KEY required for testing")
        print("Set environment variable and try again")
        return 1

    # Run tests
    test_suite = PerformanceIntegrationTests()
    success = test_suite.run_all_tests()

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)