"""
Unit Tests for DuckDB Metadata Ingestion

Tests the ingestion script's core functionality including schema creation,
single file ingestion, idempotency, NULL handling, and validation.

Usage:
    uv run python -m pytest tests/test_duckdb_ingestion.py -v
    uv run python -m pytest tests/test_duckdb_ingestion.py::TestDuckDBIngestion::test_schema_creation -v

Order: duckdb-ingestion-001
"""

import unittest
import tempfile
import json
from pathlib import Path
from datetime import datetime

try:
    import duckdb
except ImportError:
    raise ImportError("DuckDB not installed. Run: uv sync")

# Import from ingestion script
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.ingest_metadata_to_duckdb import (
    SCHEMA_DDL,
    validate_metadata,
    prepare_row,
    ingest_metadata_file
)


class TestDuckDBIngestion(unittest.TestCase):
    """Test suite for DuckDB metadata ingestion functionality"""

    def setUp(self):
        """Create temporary directory and sample metadata for each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.duckdb"

        # Sample metadata (based on real export format)
        self.sample_metadata = {
            "run_id": "DL2-OA2-TS1-F-20251017231723",
            "timestamp": "2025-10-17T23:17:23.382567",
            "parser_name": "docling",
            "parser_version": None,
            "provider_name": "openai",
            "provider_model": "gpt-4o-mini",
            "ocr_engine": "tesseract",
            "table_mode": "FAST",
            "environment": "test-machine",
            "session_label": None,
            "input_filename": "test_document.pdf",
            "input_size_bytes": 166033,
            "input_pages": None,
            "output_path": None,
            "docling_seconds": 1.5,
            "extractor_seconds": 5.2,
            "total_seconds": 6.7,
            "events_extracted": 10,
            "citations_found": 8,
            "avg_detail_length": 150.5,
            "status": "success",
            "error_message": None,
            "config_snapshot": {
                "parser": "docling",
                "parser_version": None,
                "provider": "openai",
                "provider_model": "gpt-4o-mini",
                "ocr_engine": "tesseract",
                "table_mode": "FAST",
                "environment": "test-machine",
                "session_label": None
            },
            "cost_usd": None,
            "tokens_input": None,
            "tokens_output": None
        }

    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_schema_creation(self):
        """Test that schema is created correctly with all expected columns"""
        conn = duckdb.connect(str(self.db_path))

        # Create schema
        conn.execute(SCHEMA_DDL)

        # Verify table exists
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [t[0] for t in tables]
        self.assertIn("pipeline_runs", table_names)

        # Verify columns exist
        columns = conn.execute("DESCRIBE pipeline_runs").fetchall()
        column_names = [c[0] for c in columns]

        expected_columns = [
            'run_id', 'timestamp', 'parser_name', 'parser_version', 'provider_name',
            'provider_model', 'ocr_engine', 'table_mode', 'environment', 'session_label',
            'input_filename', 'input_size_bytes', 'input_pages', 'output_path',
            'docling_seconds', 'extractor_seconds', 'total_seconds',
            'events_extracted', 'citations_found', 'avg_detail_length',
            'status', 'error_message', 'config_snapshot',
            'cost_usd', 'tokens_input', 'tokens_output'
        ]

        for col in expected_columns:
            self.assertIn(col, column_names, f"Column '{col}' missing from schema")

        # Verify indexes exist
        indexes = conn.execute("SELECT * FROM duckdb_indexes()").fetchall()
        index_names = [idx[2] for idx in indexes]  # index_name is 3rd column

        # DuckDB auto-creates primary key index, check for our custom indexes
        self.assertTrue(any('idx_provider_name' in idx for idx in index_names))
        self.assertTrue(any('idx_provider_model' in idx for idx in index_names))
        self.assertTrue(any('idx_timestamp' in idx for idx in index_names))
        self.assertTrue(any('idx_status' in idx for idx in index_names))

        conn.close()

    def test_validate_metadata_valid(self):
        """Test validation accepts valid metadata"""
        is_valid, error_msg = validate_metadata(self.sample_metadata, Path("test.json"))

        self.assertTrue(is_valid)
        self.assertEqual(error_msg, "")

    def test_validate_metadata_missing_run_id(self):
        """Test validation rejects metadata without run_id"""
        invalid_metadata = self.sample_metadata.copy()
        del invalid_metadata['run_id']

        is_valid, error_msg = validate_metadata(invalid_metadata, Path("test.json"))

        self.assertFalse(is_valid)
        self.assertIn("run_id", error_msg)

    def test_validate_metadata_missing_timestamp(self):
        """Test validation rejects metadata without timestamp"""
        invalid_metadata = self.sample_metadata.copy()
        del invalid_metadata['timestamp']

        is_valid, error_msg = validate_metadata(invalid_metadata, Path("test.json"))

        self.assertFalse(is_valid)
        self.assertIn("timestamp", error_msg)

    def test_validate_metadata_invalid_timestamp(self):
        """Test validation rejects invalid timestamp format"""
        invalid_metadata = self.sample_metadata.copy()
        invalid_metadata['timestamp'] = "not-a-valid-timestamp"

        is_valid, error_msg = validate_metadata(invalid_metadata, Path("test.json"))

        self.assertFalse(is_valid)
        self.assertIn("timestamp", error_msg.lower())

    def test_prepare_row_complete(self):
        """Test prepare_row handles complete metadata correctly"""
        row = prepare_row(self.sample_metadata)

        # Verify all fields present
        self.assertEqual(row['run_id'], "DL2-OA2-TS1-F-20251017231723")
        self.assertIsInstance(row['timestamp'], datetime)
        self.assertEqual(row['provider_name'], "openai")
        self.assertEqual(row['provider_model'], "gpt-4o-mini")
        self.assertEqual(row['events_extracted'], 10)
        self.assertEqual(row['docling_seconds'], 1.5)

        # Verify config_snapshot is JSON string
        self.assertIsInstance(row['config_snapshot'], str)
        config = json.loads(row['config_snapshot'])
        self.assertEqual(config['provider'], "openai")

    def test_prepare_row_null_handling(self):
        """Test prepare_row handles NULL values correctly"""
        minimal_metadata = {
            "run_id": "TEST-123",
            "timestamp": "2025-10-17T12:00:00",
            # All other fields missing
        }

        row = prepare_row(minimal_metadata)

        # Verify required fields
        self.assertEqual(row['run_id'], "TEST-123")
        self.assertIsInstance(row['timestamp'], datetime)

        # Verify optional fields are None
        self.assertIsNone(row['provider_name'])
        self.assertIsNone(row['cost_usd'])
        self.assertIsNone(row['events_extracted'])
        self.assertIsNone(row['config_snapshot'])

    def test_single_file_ingestion(self):
        """Test ingesting a single metadata file"""
        # Create metadata JSON file
        metadata_file = Path(self.temp_dir) / "test_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.sample_metadata, f, indent=2)

        # Create database and schema
        conn = duckdb.connect(str(self.db_path))
        conn.execute(SCHEMA_DDL)

        # Ingest file
        success, message = ingest_metadata_file(conn, metadata_file, dry_run=False)

        self.assertTrue(success, f"Ingestion failed: {message}")
        self.assertIn("run_id", message)

        # Verify row was inserted
        result = conn.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()
        self.assertEqual(result[0], 1)

        # Verify data integrity
        row = conn.execute("SELECT * FROM pipeline_runs").fetchone()
        self.assertEqual(row[0], "DL2-OA2-TS1-F-20251017231723")  # run_id

        conn.close()

    def test_idempotency(self):
        """Test re-ingesting same file doesn't create duplicates"""
        # Create metadata JSON file
        metadata_file = Path(self.temp_dir) / "test_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.sample_metadata, f, indent=2)

        # Create database and schema
        conn = duckdb.connect(str(self.db_path))
        conn.execute(SCHEMA_DDL)

        # Ingest file first time
        success1, msg1 = ingest_metadata_file(conn, metadata_file, dry_run=False)
        self.assertTrue(success1)

        # Verify one row
        count1 = conn.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0]
        self.assertEqual(count1, 1)

        # Ingest same file again
        success2, msg2 = ingest_metadata_file(conn, metadata_file, dry_run=False)
        self.assertTrue(success2)

        # Verify still only one row (idempotent - replaced, not duplicated)
        count2 = conn.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0]
        self.assertEqual(count2, 1, "Re-ingestion created duplicate rows")

        conn.close()

    def test_dry_run_mode(self):
        """Test dry-run mode validates without inserting"""
        # Create metadata JSON file
        metadata_file = Path(self.temp_dir) / "test_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.sample_metadata, f, indent=2)

        # Create database and schema
        conn = duckdb.connect(str(self.db_path))
        conn.execute(SCHEMA_DDL)

        # Ingest in dry-run mode
        success, message = ingest_metadata_file(conn, metadata_file, dry_run=True)

        self.assertTrue(success)
        self.assertIn("DRY RUN", message)

        # Verify no rows inserted
        count = conn.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0]
        self.assertEqual(count, 0, "Dry-run mode should not insert rows")

        conn.close()

    def test_invalid_json_handling(self):
        """Test handling of malformed JSON files"""
        # Create invalid JSON file
        invalid_file = Path(self.temp_dir) / "invalid.json"
        with open(invalid_file, 'w', encoding='utf-8') as f:
            f.write("{ this is not valid JSON }")

        # Create database and schema
        conn = duckdb.connect(str(self.db_path))
        conn.execute(SCHEMA_DDL)

        # Attempt ingestion
        success, message = ingest_metadata_file(conn, invalid_file, dry_run=False)

        self.assertFalse(success)
        self.assertIn("JSON", message)

        # Verify no rows inserted
        count = conn.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0]
        self.assertEqual(count, 0)

        conn.close()

    def test_json_column_querying(self):
        """Test JSON column can be queried correctly"""
        # Create metadata JSON file
        metadata_file = Path(self.temp_dir) / "test_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.sample_metadata, f, indent=2)

        # Create database, schema, and ingest
        conn = duckdb.connect(str(self.db_path))
        conn.execute(SCHEMA_DDL)
        ingest_metadata_file(conn, metadata_file, dry_run=False)

        # Query JSON column
        result = conn.execute("""
            SELECT
                config_snapshot->>'provider' as config_provider,
                config_snapshot->>'provider_model' as config_model
            FROM pipeline_runs
        """).fetchone()

        self.assertEqual(result[0], "openai")
        self.assertEqual(result[1], "gpt-4o-mini")

        conn.close()


def run_tests():
    """Run test suite"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDuckDBIngestion)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
