#!/usr/bin/env python3
"""
DuckDB Query Examples - Pipeline Metadata Analytics

Demonstrates common queries for analyzing pipeline runs stored in DuckDB.
Use these examples as templates for custom analytics.

Usage:
    python scripts/query_duckdb.py --db runs.duckdb
    python scripts/query_duckdb.py --db runs.duckdb --query avg-time
    python scripts/query_duckdb.py --help

For more query examples, see: docs/reports/duckdb-queries.sql

Order: duckdb-ingestion-001
"""

import argparse
import sys
from pathlib import Path

try:
    import duckdb
    import pandas as pd
except ImportError:
    print("❌ Required packages not installed. Run: uv sync")
    sys.exit(1)


def connect_db(db_path: str) -> duckdb.DuckDBPyConnection:
    """
    Connect to DuckDB database.

    Args:
        db_path: Path to DuckDB file

    Returns:
        DuckDB connection object
    """
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        print(f"💡 Run ingestion first: python scripts/ingest_metadata_to_duckdb.py --db {db_path} --glob \"output/**/*_metadata.json\"")
        sys.exit(1)

    return duckdb.connect(db_path, read_only=True)


# ============================================================================
# EXAMPLE QUERIES
# ============================================================================

def query_database_stats(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Get overall database statistics.

    Returns summary counts for runs, providers, models, and environments.
    """
    query = """
    SELECT
        COUNT(*) as total_runs,
        COUNT(DISTINCT provider_name) as unique_providers,
        COUNT(DISTINCT provider_model) as unique_models,
        COUNT(DISTINCT environment) as unique_environments,
        MIN(timestamp) as first_run,
        MAX(timestamp) as last_run
    FROM pipeline_runs;
    """
    return conn.execute(query).df()


def query_avg_time_by_model(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Average extractor time by model.

    Shows performance comparison across different models.
    Useful for identifying fastest models for production use.
    """
    query = """
    SELECT
        provider_model,
        COUNT(*) as total_runs,
        ROUND(AVG(extractor_seconds), 3) as avg_extractor_time,
        ROUND(MIN(extractor_seconds), 3) as min_time,
        ROUND(MAX(extractor_seconds), 3) as max_time,
        ROUND(AVG(total_seconds), 3) as avg_total_time
    FROM pipeline_runs
    WHERE status = 'success'
        AND extractor_seconds IS NOT NULL
    GROUP BY provider_model
    ORDER BY avg_extractor_time ASC;
    """
    return conn.execute(query).df()


def query_cost_by_provider(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Total cost by provider.

    Tracks spending across different providers (when cost_usd is available).
    Useful for budget analysis and cost optimization.
    """
    query = """
    SELECT
        provider_name,
        COUNT(*) as total_runs,
        ROUND(SUM(cost_usd), 4) as total_cost_usd,
        ROUND(AVG(cost_usd), 4) as avg_cost_per_run,
        ROUND(MIN(cost_usd), 4) as min_cost,
        ROUND(MAX(cost_usd), 4) as max_cost
    FROM pipeline_runs
    WHERE cost_usd IS NOT NULL
    GROUP BY provider_name
    ORDER BY total_cost_usd DESC;
    """
    df = conn.execute(query).df()
    if df.empty:
        print("💡 No cost data available (cost_usd is NULL for all runs)")
    return df


def query_success_rate_by_provider(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Success rate by provider.

    Monitors reliability of each provider.
    Useful for identifying providers with high failure rates.
    """
    query = """
    SELECT
        provider_name,
        COUNT(*) as total_runs,
        SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful_runs,
        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_runs,
        ROUND(100.0 * SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) / COUNT(*), 2) as success_rate_pct
    FROM pipeline_runs
    GROUP BY provider_name
    ORDER BY success_rate_pct DESC;
    """
    return conn.execute(query).df()


def query_recent_runs(conn: duckdb.DuckDBPyConnection, days: int = 7) -> pd.DataFrame:
    """
    Recent runs (last N days).

    Shows recent activity and outcomes.
    Useful for monitoring current work and debugging recent failures.

    Args:
        days: Number of days to look back (default: 7)
    """
    query = f"""
    SELECT
        run_id,
        timestamp,
        provider_model,
        events_extracted,
        ROUND(total_seconds, 2) as total_seconds,
        status,
        input_filename
    FROM pipeline_runs
    WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '{days} days'
    ORDER BY timestamp DESC;
    """
    return conn.execute(query).df()


def query_slowest_documents(conn: duckdb.DuckDBPyConnection, limit: int = 20) -> pd.DataFrame:
    """
    Slowest documents.

    Identifies performance bottlenecks and large files.
    Useful for optimization and capacity planning.

    Args:
        limit: Number of results to return (default: 20)
    """
    query = f"""
    SELECT
        input_filename,
        provider_model,
        ROUND(total_seconds, 2) as total_seconds,
        ROUND(input_size_bytes / 1024.0 / 1024.0, 2) as size_mb,
        events_extracted,
        status
    FROM pipeline_runs
    WHERE total_seconds IS NOT NULL
    ORDER BY total_seconds DESC
    LIMIT {limit};
    """
    return conn.execute(query).df()


def query_events_by_provider(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Average events extracted by provider.

    Compares extraction thoroughness across providers.
    Useful for quality assessment and provider selection.
    """
    query = """
    SELECT
        provider_name,
        provider_model,
        COUNT(*) as total_runs,
        ROUND(AVG(events_extracted), 1) as avg_events,
        MIN(events_extracted) as min_events,
        MAX(events_extracted) as max_events
    FROM pipeline_runs
    WHERE status = 'success'
        AND events_extracted IS NOT NULL
    GROUP BY provider_name, provider_model
    ORDER BY avg_events DESC;
    """
    return conn.execute(query).df()


def query_config_validation(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Config validation (JSON column querying).

    Demonstrates JSON column queries and validates config consistency.
    Useful for debugging configuration mismatches.
    """
    query = """
    SELECT
        run_id,
        provider_name,
        provider_model,
        config_snapshot->>'provider' as config_provider,
        config_snapshot->>'provider_model' as config_model,
        CASE
            WHEN provider_name != config_snapshot->>'provider' THEN 'MISMATCH'
            WHEN provider_model != config_snapshot->>'provider_model' THEN 'MISMATCH'
            ELSE 'OK'
        END as validation_status
    FROM pipeline_runs
    WHERE config_snapshot IS NOT NULL
    LIMIT 10;
    """
    return conn.execute(query).df()


# ============================================================================
# CLI INTERFACE
# ============================================================================

QUERY_MAP = {
    'stats': ('Database Statistics', query_database_stats),
    'avg-time': ('Average Time by Model', query_avg_time_by_model),
    'cost': ('Cost by Provider', query_cost_by_provider),
    'success-rate': ('Success Rate by Provider', query_success_rate_by_provider),
    'recent': ('Recent Runs (Last 7 Days)', query_recent_runs),
    'slowest': ('Slowest Documents', query_slowest_documents),
    'events': ('Events by Provider', query_events_by_provider),
    'config': ('Config Validation', query_config_validation),
}


def run_query(conn: duckdb.DuckDBPyConnection, query_name: str) -> None:
    """
    Run a named query and display results.

    Args:
        conn: DuckDB connection
        query_name: Name of query to run (from QUERY_MAP)
    """
    if query_name not in QUERY_MAP:
        print(f"❌ Unknown query: {query_name}")
        print(f"💡 Available queries: {', '.join(QUERY_MAP.keys())}")
        sys.exit(1)

    title, query_func = QUERY_MAP[query_name]

    print(f"\n{'='*70}")
    print(f"📊 {title}")
    print(f"{'='*70}\n")

    df = query_func(conn)

    if df.empty:
        print("(No results)")
    else:
        # Use pandas display options for better formatting
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 50)
        print(df.to_string(index=False))

    print()


def run_all_queries(conn: duckdb.DuckDBPyConnection) -> None:
    """Run all example queries and display results."""
    for query_name in QUERY_MAP.keys():
        run_query(conn, query_name)


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Query DuckDB pipeline metadata with example analytics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available Queries:
{chr(10).join(f"  {name:15} - {title}" for name, (title, _) in QUERY_MAP.items())}

Examples:
  # Run all queries
  python scripts/query_duckdb.py --db runs.duckdb

  # Run specific query
  python scripts/query_duckdb.py --db runs.duckdb --query avg-time

  # Custom query (interactive)
  python -c "import duckdb; conn = duckdb.connect('runs.duckdb'); print(conn.execute('SELECT * FROM pipeline_runs LIMIT 5').df())"

For more query examples, see: docs/reports/duckdb-queries.sql
        """
    )

    parser.add_argument(
        '--db',
        type=str,
        default='runs.duckdb',
        help='Path to DuckDB database file (default: runs.duckdb)'
    )

    parser.add_argument(
        '--query',
        type=str,
        choices=list(QUERY_MAP.keys()),
        help='Specific query to run (default: run all)'
    )

    args = parser.parse_args()

    # Connect to database
    try:
        conn = connect_db(args.db)
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        sys.exit(1)

    try:
        # Run query or all queries
        if args.query:
            run_query(conn, args.query)
        else:
            print("\n📈 Running all example queries...")
            run_all_queries(conn)

        print("✨ Done! For more query examples, see: docs/reports/duckdb-queries.sql")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
