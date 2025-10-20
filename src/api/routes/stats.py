"""
Statistics Aggregation Endpoints

Provides aggregated statistics by model and provider.

Order: fastapi-duckdb-api-001
"""

from typing import Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
import duckdb

from src.api.models import (
    StatsItem, ListResponse, ResponseMeta, ResponseLinks
)
from src.api.database import get_db

router = APIRouter(prefix="/api/v1/stats", tags=["Statistics"])


@router.get(
    "/by-model",
    response_model=ListResponse[StatsItem],
    summary="Statistics by Model",
    description=(
        "Aggregate performance statistics grouped by model. "
        "Returns counts and avg/min/max for extraction times."
    )
)
def stats_by_model(
    request: Request,
    date_from: Optional[datetime] = Query(None, description="Filter runs after this timestamp (UTC)"),
    date_to: Optional[datetime] = Query(None, description="Filter runs before this timestamp (UTC)"),
    status: Optional[str] = Query(None, description="Filter by status (e.g., 'success')"),
    db: duckdb.DuckDBPyConnection = Depends(get_db)
):
    """
    Aggregate statistics by model.

    Based on SQL query #2 from docs/reports/duckdb-queries.sql
    (Average Extractor Time by Model).

    Query Parameters:
    - date_from/date_to: Date range filter (ISO 8601 UTC)
    - status: Filter by status (recommended: 'success' for performance analysis)

    Returns:
    - List of StatsItem with performance aggregations per model
    """
    try:
        # Build WHERE clause
        where_clauses = []
        params = []

        # Date range filters
        if date_from:
            where_clauses.append("timestamp >= ?")
            params.append(date_from)
        if date_to:
            where_clauses.append("timestamp <= ?")
            params.append(date_to)

        # Status filter
        if status:
            where_clauses.append("status = ?")
            params.append(status)
        else:
            # Default: only successful runs for performance metrics
            where_clauses.append("status = 'success'")

        # Combine WHERE clauses
        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

        # Build query
        # Mirrors query #2 from duckdb-queries.sql
        query = f"""
            SELECT
                provider_model as key_name,
                COUNT(*) as total_runs,
                ROUND(AVG(extractor_seconds), 3) as avg_extractor_seconds,
                ROUND(MIN(extractor_seconds), 3) as min_extractor_seconds,
                ROUND(MAX(extractor_seconds), 3) as max_extractor_seconds,
                ROUND(AVG(total_seconds), 3) as avg_total_seconds,
                ROUND(MIN(total_seconds), 3) as min_total_seconds,
                ROUND(MAX(total_seconds), 3) as max_total_seconds
            FROM pipeline_runs
            WHERE {where_sql}
                AND extractor_seconds IS NOT NULL
                AND total_seconds IS NOT NULL
            GROUP BY provider_model
            ORDER BY avg_extractor_seconds ASC
        """

        # Execute query
        rows = db.execute(query, params).fetchall()

        # Convert to StatsItem models
        stats = []
        for row in rows:
            stats.append(StatsItem(
                key_name=row[0],
                total_runs=row[1],
                avg_extractor_seconds=row[2],
                min_extractor_seconds=row[3],
                max_extractor_seconds=row[4],
                avg_total_seconds=row[5],
                min_total_seconds=row[6],
                max_total_seconds=row[7]
            ))

        # Build response envelope
        response = ListResponse(
            data=stats,
            meta=ResponseMeta(
                returned=len(stats),
                total=len(stats)  # No pagination for stats
            ),
            links=ResponseLinks(
                self=str(request.url)
            )
        )

        return response

    except Exception as e:
        print(f"Error computing stats by model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "INTERNAL_ERROR", "message": str(e)}
        )


@router.get(
    "/by-provider",
    response_model=ListResponse[StatsItem],
    summary="Statistics by Provider",
    description=(
        "Aggregate performance statistics grouped by provider. "
        "Returns counts and avg/min/max for extraction times."
    )
)
def stats_by_provider(
    request: Request,
    date_from: Optional[datetime] = Query(None, description="Filter runs after this timestamp (UTC)"),
    date_to: Optional[datetime] = Query(None, description="Filter runs before this timestamp (UTC)"),
    status: Optional[str] = Query(None, description="Filter by status (e.g., 'success')"),
    db: duckdb.DuckDBPyConnection = Depends(get_db)
):
    """
    Aggregate statistics by provider.

    Similar to by-model endpoint but groups by provider_name.

    Query Parameters:
    - date_from/date_to: Date range filter (ISO 8601 UTC)
    - status: Filter by status (recommended: 'success' for performance analysis)

    Returns:
    - List of StatsItem with performance aggregations per provider
    """
    try:
        # Build WHERE clause
        where_clauses = []
        params = []

        # Date range filters
        if date_from:
            where_clauses.append("timestamp >= ?")
            params.append(date_from)
        if date_to:
            where_clauses.append("timestamp <= ?")
            params.append(date_to)

        # Status filter
        if status:
            where_clauses.append("status = ?")
            params.append(status)
        else:
            # Default: only successful runs for performance metrics
            where_clauses.append("status = 'success'")

        # Combine WHERE clauses
        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

        # Build query
        # Groups by provider_name instead of provider_model
        query = f"""
            SELECT
                provider_name as key_name,
                COUNT(*) as total_runs,
                ROUND(AVG(extractor_seconds), 3) as avg_extractor_seconds,
                ROUND(MIN(extractor_seconds), 3) as min_extractor_seconds,
                ROUND(MAX(extractor_seconds), 3) as max_extractor_seconds,
                ROUND(AVG(total_seconds), 3) as avg_total_seconds,
                ROUND(MIN(total_seconds), 3) as min_total_seconds,
                ROUND(MAX(total_seconds), 3) as max_total_seconds
            FROM pipeline_runs
            WHERE {where_sql}
                AND extractor_seconds IS NOT NULL
                AND total_seconds IS NOT NULL
            GROUP BY provider_name
            ORDER BY avg_extractor_seconds ASC
        """

        # Execute query
        rows = db.execute(query, params).fetchall()

        # Convert to StatsItem models
        stats = []
        for row in rows:
            stats.append(StatsItem(
                key_name=row[0],
                total_runs=row[1],
                avg_extractor_seconds=row[2],
                min_extractor_seconds=row[3],
                max_extractor_seconds=row[4],
                avg_total_seconds=row[5],
                min_total_seconds=row[6],
                max_total_seconds=row[7]
            ))

        # Build response envelope
        response = ListResponse(
            data=stats,
            meta=ResponseMeta(
                returned=len(stats),
                total=len(stats)  # No pagination for stats
            ),
            links=ResponseLinks(
                self=str(request.url)
            )
        )

        return response

    except Exception as e:
        print(f"Error computing stats by provider: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "INTERNAL_ERROR", "message": str(e)}
        )
