#!/usr/bin/env python3
"""
DuckDB Metadata Ingestion Script

Scans for pipeline metadata JSON files and loads them into a DuckDB database
for queryable analytics. Supports idempotent re-ingestion and batch processing.

Usage:
    python scripts/ingest_metadata_to_duckdb.py --db runs.duckdb --glob "output/**/*_metadata.json"
    python scripts/ingest_metadata_to_duckdb.py --help
    python scripts/ingest_metadata_to_duckdb.py --db runs.duckdb --glob "output/**/*_metadata.json" --dry-run
    python scripts/ingest_metadata_to_duckdb.py --db runs.duckdb --glob "output/**/*_metadata.json" --replace

Order: duckdb-ingestion-001
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

try:
    import duckdb
except ImportError:
    logger.error("❌ DuckDB not installed. Run: uv pip install duckdb")
    sys.exit(1)


# === SCHEMA DDL ===

SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS pipeline_runs (
    -- Primary Key
    run_id VARCHAR PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,

    -- Configuration
    parser_name VARCHAR,
    parser_version VARCHAR,
    provider_name VARCHAR,
    provider_model VARCHAR,
    ocr_engine VARCHAR,
    table_mode VARCHAR,

    -- Environment Tracking
    environment VARCHAR,
    session_label VARCHAR,

    -- Input Metadata
    input_filename VARCHAR,
    input_size_bytes BIGINT,
    input_pages INTEGER,
    output_path VARCHAR,

    -- Performance Metrics
    docling_seconds DOUBLE,
    extractor_seconds DOUBLE,
    total_seconds DOUBLE,

    -- Quality Metrics
    events_extracted INTEGER,
    citations_found INTEGER,
    avg_detail_length DOUBLE,

    -- Status Tracking
    status VARCHAR,
    error_message VARCHAR,

    -- Nested Configuration (JSON)
    config_snapshot JSON,

    -- Cost Tracking
    cost_usd DOUBLE,
    tokens_input INTEGER,
    tokens_output INTEGER
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_provider_name ON pipeline_runs(provider_name);
CREATE INDEX IF NOT EXISTS idx_provider_model ON pipeline_runs(provider_model);
CREATE INDEX IF NOT EXISTS idx_timestamp ON pipeline_runs(timestamp);
CREATE INDEX IF NOT EXISTS idx_status ON pipeline_runs(status);
"""


# === INGESTION LOGIC ===

def discover_metadata_files(glob_pattern: str) -> List[Path]:
    """
    Discover metadata JSON files matching the glob pattern.

    Args:
        glob_pattern: Glob pattern for file discovery (e.g., "output/**/*_metadata.json")

    Returns:
        List of Path objects for discovered files
    """
    # Use Path.cwd() to resolve relative paths
    base_path = Path.cwd()

    # Extract directory part from glob pattern
    if "**" in glob_pattern:
        # Pattern like "output/**/*_metadata.json"
        parts = glob_pattern.split("**")
        search_dir = base_path / parts[0].strip("/")
        pattern = "**/" + parts[1].strip("/")
    else:
        # Simple pattern like "*.json"
        search_dir = base_path
        pattern = glob_pattern

    logger.info(f"🔍 Searching in: {search_dir}")
    logger.info(f"📋 Pattern: {pattern}")

    # Discover files
    files = list(search_dir.glob(pattern))

    # Filter out non-metadata files
    metadata_files = [f for f in files if f.name.endswith('_metadata.json')]

    logger.info(f"✅ Discovered {len(metadata_files)} metadata file(s)")

    return sorted(metadata_files)


def validate_metadata(metadata: Dict[str, Any], file_path: Path) -> Tuple[bool, str]:
    """
    Validate metadata structure for required fields.

    Args:
        metadata: Parsed JSON metadata
        file_path: Source file path (for error reporting)

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check required fields
    if 'run_id' not in metadata or not metadata['run_id']:
        return False, f"Missing required field: run_id"

    if 'timestamp' not in metadata or not metadata['timestamp']:
        return False, f"Missing required field: timestamp"

    # Validate timestamp format
    try:
        datetime.fromisoformat(metadata['timestamp'].replace('Z', '+00:00'))
    except (ValueError, AttributeError) as e:
        return False, f"Invalid timestamp format: {e}"

    return True, ""


def prepare_row(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare metadata for database insertion.

    Handles type conversions, NULL values, and JSON serialization.

    Args:
        metadata: Parsed JSON metadata

    Returns:
        Dict ready for database insertion
    """
    # Parse timestamp
    timestamp_str = metadata.get('timestamp', '')
    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

    # Serialize config_snapshot as JSON string
    config_snapshot = metadata.get('config_snapshot')
    config_json = json.dumps(config_snapshot) if config_snapshot else None

    # Build row with NULL safety (.get() returns None for missing keys)
    row = {
        'run_id': metadata.get('run_id'),
        'timestamp': timestamp,
        'parser_name': metadata.get('parser_name'),
        'parser_version': metadata.get('parser_version'),
        'provider_name': metadata.get('provider_name'),
        'provider_model': metadata.get('provider_model'),
        'ocr_engine': metadata.get('ocr_engine'),
        'table_mode': metadata.get('table_mode'),
        'environment': metadata.get('environment'),
        'session_label': metadata.get('session_label'),
        'input_filename': metadata.get('input_filename'),
        'input_size_bytes': metadata.get('input_size_bytes'),
        'input_pages': metadata.get('input_pages'),
        'output_path': metadata.get('output_path'),
        'docling_seconds': metadata.get('docling_seconds'),
        'extractor_seconds': metadata.get('extractor_seconds'),
        'total_seconds': metadata.get('total_seconds'),
        'events_extracted': metadata.get('events_extracted'),
        'citations_found': metadata.get('citations_found'),
        'avg_detail_length': metadata.get('avg_detail_length'),
        'status': metadata.get('status'),
        'error_message': metadata.get('error_message'),
        'config_snapshot': config_json,
        'cost_usd': metadata.get('cost_usd'),
        'tokens_input': metadata.get('tokens_input'),
        'tokens_output': metadata.get('tokens_output')
    }

    return row


def ingest_metadata_file(
    conn: duckdb.DuckDBPyConnection,
    file_path: Path,
    dry_run: bool = False
) -> Tuple[bool, str]:
    """
    Ingest a single metadata file into the database.

    Args:
        conn: DuckDB connection
        file_path: Path to metadata JSON file
        dry_run: If True, validate but don't insert

    Returns:
        Tuple of (success, message)
    """
    try:
        # Read and parse JSON
        with open(file_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Validate
        is_valid, error_msg = validate_metadata(metadata, file_path)
        if not is_valid:
            return False, f"Validation failed: {error_msg}"

        # Prepare row
        row = prepare_row(metadata)

        if dry_run:
            return True, f"[DRY RUN] Would insert: run_id={row['run_id']}"

        # Insert or replace (idempotent)
        conn.execute("""
            INSERT OR REPLACE INTO pipeline_runs (
                run_id, timestamp, parser_name, parser_version, provider_name, provider_model,
                ocr_engine, table_mode, environment, session_label, input_filename,
                input_size_bytes, input_pages, output_path, docling_seconds, extractor_seconds,
                total_seconds, events_extracted, citations_found, avg_detail_length, status,
                error_message, config_snapshot, cost_usd, tokens_input, tokens_output
            ) VALUES (
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?
            )
        """, (
            row['run_id'], row['timestamp'], row['parser_name'], row['parser_version'],
            row['provider_name'], row['provider_model'], row['ocr_engine'], row['table_mode'],
            row['environment'], row['session_label'], row['input_filename'],
            row['input_size_bytes'], row['input_pages'], row['output_path'],
            row['docling_seconds'], row['extractor_seconds'], row['total_seconds'],
            row['events_extracted'], row['citations_found'], row['avg_detail_length'],
            row['status'], row['error_message'], row['config_snapshot'],
            row['cost_usd'], row['tokens_input'], row['tokens_output']
        ))

        return True, f"Inserted run_id={row['run_id']}"

    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except Exception as e:
        return False, f"Unexpected error: {e}"


def batch_ingest(
    db_path: str,
    glob_pattern: str,
    dry_run: bool = False,
    replace: bool = False
) -> None:
    """
    Batch ingest metadata files into DuckDB.

    Args:
        db_path: Path to DuckDB database file
        glob_pattern: Glob pattern for file discovery
        dry_run: If True, validate files but don't insert
        replace: If True, drop and recreate table (ignored if dry_run=True)
    """
    logger.info("🚀 Starting DuckDB metadata ingestion")
    logger.info(f"📊 Database: {db_path}")
    logger.info(f"📂 Pattern: {glob_pattern}")
    logger.info(f"🧪 Dry run: {dry_run}")
    logger.info(f"🔄 Replace mode: {replace}")

    # Discover files
    files = discover_metadata_files(glob_pattern)

    if not files:
        logger.warning("⚠️  No metadata files found. Nothing to ingest.")
        return

    # Connect to database
    logger.info(f"📡 Connecting to database: {db_path}")
    conn = duckdb.connect(db_path)

    try:
        # Create schema (or recreate if replace mode)
        if replace and not dry_run:
            logger.info("🗑️  Dropping existing table (replace mode)")
            conn.execute("DROP TABLE IF EXISTS pipeline_runs")

        logger.info("📋 Creating schema (if not exists)")
        conn.execute(SCHEMA_DDL)

        # Ingest files
        success_count = 0
        error_count = 0
        skipped_count = 0

        logger.info(f"📥 Processing {len(files)} file(s)...")

        for idx, file_path in enumerate(files, 1):
            # Progress indicator
            if idx % 10 == 0 or idx == len(files):
                logger.info(f"Progress: {idx}/{len(files)} files processed")

            # Ingest file
            success, message = ingest_metadata_file(conn, file_path, dry_run=dry_run)

            if success:
                success_count += 1
                logger.debug(f"✅ {file_path.name}: {message}")
            else:
                error_count += 1
                logger.warning(f"❌ {file_path.name}: {message}")

        # Summary statistics
        logger.info("")
        logger.info("=" * 60)
        logger.info("📊 INGESTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total files discovered: {len(files)}")
        logger.info(f"✅ Successfully processed: {success_count}")
        logger.info(f"❌ Errors: {error_count}")
        logger.info(f"⏭️  Skipped: {skipped_count}")
        logger.info("=" * 60)

        if not dry_run:
            # Query database stats
            total_rows = conn.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0]
            unique_providers = conn.execute(
                "SELECT COUNT(DISTINCT provider_name) FROM pipeline_runs"
            ).fetchone()[0]
            unique_models = conn.execute(
                "SELECT COUNT(DISTINCT provider_model) FROM pipeline_runs"
            ).fetchone()[0]

            logger.info("")
            logger.info("📈 DATABASE STATISTICS")
            logger.info("=" * 60)
            logger.info(f"Total runs in database: {total_rows}")
            logger.info(f"Unique providers: {unique_providers}")
            logger.info(f"Unique models: {unique_models}")
            logger.info("=" * 60)

            logger.info("")
            logger.info("✨ Ingestion complete! Query your data:")
            logger.info(f"   uv run python -c \"import duckdb; conn = duckdb.connect('{db_path}'); print(conn.execute('SELECT * FROM pipeline_runs LIMIT 5').df())\"")
            logger.info(f"   uv run python scripts/query_duckdb.py --db {db_path}")
        else:
            logger.info("")
            logger.info("✨ Dry run complete! Remove --dry-run flag to perform actual ingestion.")

    finally:
        conn.close()
        logger.info("🔌 Database connection closed")


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Ingest pipeline metadata JSON files into DuckDB for analytics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic ingestion
  python scripts/ingest_metadata_to_duckdb.py --db runs.duckdb --glob "output/**/*_metadata.json"

  # Dry run (validate without inserting)
  python scripts/ingest_metadata_to_duckdb.py --db runs.duckdb --glob "output/**/*_metadata.json" --dry-run

  # Replace mode (drop and recreate table)
  python scripts/ingest_metadata_to_duckdb.py --db runs.duckdb --glob "output/**/*_metadata.json" --replace

  # Verbose logging
  python scripts/ingest_metadata_to_duckdb.py --db runs.duckdb --glob "output/**/*_metadata.json" --verbose

For more information, see: docs/reports/duckdb-ingestion-plan.md
        """
    )

    parser.add_argument(
        '--db',
        type=str,
        default='runs.duckdb',
        help='Path to DuckDB database file (default: runs.duckdb)'
    )

    parser.add_argument(
        '--glob',
        type=str,
        required=True,
        help='Glob pattern for metadata files (e.g., "output/**/*_metadata.json")'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate files without inserting into database'
    )

    parser.add_argument(
        '--replace',
        action='store_true',
        help='Drop and recreate table before ingestion (destructive!)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose (DEBUG) logging'
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run ingestion
    try:
        batch_ingest(
            db_path=args.db,
            glob_pattern=args.glob,
            dry_run=args.dry_run,
            replace=args.replace
        )
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
