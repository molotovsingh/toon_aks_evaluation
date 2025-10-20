"""
DuckDB Read-Only Connection Manager

Provides thread-safe, read-only access to the pipeline_runs database.

Order: fastapi-duckdb-api-001
"""

import duckdb
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

from src.api.config import settings


class DatabaseManager:
    """Manages read-only DuckDB connections"""

    def __init__(self):
        self._conn: Optional[duckdb.DuckDBPyConnection] = None

    def connect(self) -> duckdb.DuckDBPyConnection:
        """
        Get or create read-only DuckDB connection.

        Returns:
            duckdb.DuckDBPyConnection: Read-only connection

        Raises:
            FileNotFoundError: If database file doesn't exist
            duckdb.Error: If database is invalid or connection fails
        """
        if self._conn is None:
            db_path = Path(settings.runs_db_path)

            # Check if database file exists
            if not db_path.exists():
                raise FileNotFoundError(
                    f"Database file not found: {db_path}. "
                    "Run ingestion script to create the database:\n"
                    f"  uv run python scripts/ingest_metadata_to_duckdb.py "
                    f"--db {db_path} --glob 'output/**/*_metadata.json'"
                )

            # Create read-only connection
            self._conn = duckdb.connect(str(db_path), read_only=True)

        return self._conn

    def close(self):
        """Close the database connection"""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def is_connected(self) -> bool:
        """Check if database is currently connected"""
        return self._conn is not None

    def health_check(self) -> bool:
        """
        Verify database is accessible and has expected schema.

        Returns:
            bool: True if database is healthy, False otherwise
        """
        try:
            conn = self.connect()
            # Check if pipeline_runs table exists
            result = conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'pipeline_runs'"
            ).fetchone()
            return result[0] == 1 if result else False
        except Exception:
            return False


# Global database manager instance
db_manager = DatabaseManager()


@contextmanager
def get_db():
    """
    Dependency injection for FastAPI routes.

    Yields:
        duckdb.DuckDBPyConnection: Read-only database connection

    Usage:
        @router.get("/example")
        def example_route(db: duckdb.DuckDBPyConnection = Depends(get_db)):
            result = db.execute("SELECT * FROM pipeline_runs LIMIT 1").fetchone()
            return result
    """
    conn = None
    try:
        conn = db_manager.connect()
        yield conn
    finally:
        # Connection is reused, don't close it
        pass


def shutdown_db():
    """Cleanup function to close database on app shutdown"""
    db_manager.close()
