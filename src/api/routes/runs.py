"""
Pipeline Runs Endpoints

Provides list and detail endpoints for querying pipeline execution runs.

Order: fastapi-duckdb-api-001
"""

import json
from typing import Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
import duckdb

from src.api.models import (
    RunSummary, RunDetail, ListResponse, DetailResponse,
    ResponseMeta, ResponseLinks, ErrorDetail, ErrorResponse
)
from src.api.database import get_db
from src.api.utils import (
    encode_cursor, decode_cursor,
    sanitize_config_snapshot, parse_sort_param
)

router = APIRouter(prefix="/api/v1", tags=["Runs"])

# Sort field whitelist (SQL injection prevention)
ALLOWED_SORT_FIELDS = {
    'timestamp', 'provider_name', 'provider_model',
    'total_seconds', 'extractor_seconds'
}


@router.get(
    "/runs",
    response_model=ListResponse[RunSummary],
    summary="List Pipeline Runs",
    description=(
        "List pipeline execution runs with filtering, sorting, and cursor-based pagination. "
        "Returns summary view (excludes config_snapshot for performance)."
    )
)
def list_runs(
    request: Request,
    limit: int = Query(50, ge=1, le=500, description="Maximum results per page"),
    cursor: Optional[str] = Query(None, description="Pagination cursor (base64-encoded)"),
    date_from: Optional[datetime] = Query(None, description="Filter runs after this timestamp (UTC)"),
    date_to: Optional[datetime] = Query(None, description="Filter runs before this timestamp (UTC)"),
    provider: Optional[str] = Query(None, description="Filter by provider name (exact match)"),
    model: Optional[str] = Query(None, description="Filter by model identifier (exact match)"),
    status: Optional[str] = Query(None, description="Filter by status (exact match)"),
    filename_contains: Optional[str] = Query(None, description="Filter by filename substring"),
    sort: str = Query("timestamp:desc", description="Sort field:direction (e.g., 'timestamp:desc')"),
    db: duckdb.DuckDBPyConnection = Depends(get_db)
):
    """
    List pipeline runs with filtering and pagination.

    Query Parameters:
    - limit: Results per page (1-500, default 50)
    - cursor: Pagination cursor from previous response
    - date_from/date_to: Date range filter (ISO 8601 UTC)
    - provider/model/status: Exact match filters
    - filename_contains: Case-insensitive substring search
    - sort: Field and direction (whitelist: timestamp, provider_name, provider_model, total_seconds, extractor_seconds)

    Returns:
    - Paginated list of runs with meta.total count and next_cursor
    """
    try:
        # Parse and validate sort parameter
        sort_field, sort_direction = parse_sort_param(sort, ALLOWED_SORT_FIELDS)

        # Build WHERE clause dynamically
        where_clauses = []
        params = []

        # Cursor pagination (keyset pagination for efficiency)
        cursor_timestamp = None
        cursor_run_id = None
        if cursor:
            try:
                cursor_timestamp, cursor_run_id = decode_cursor(cursor)
                # For DESC sort, we want rows "less than" the cursor
                # For ASC sort, we want rows "greater than" the cursor
                if sort_direction == 'desc':
                    where_clauses.append(
                        f"((timestamp < ?) OR (timestamp = ? AND run_id < ?))"
                    )
                    params.extend([cursor_timestamp, cursor_timestamp, cursor_run_id])
                else:
                    where_clauses.append(
                        f"((timestamp > ?) OR (timestamp = ? AND run_id > ?))"
                    )
                    params.extend([cursor_timestamp, cursor_timestamp, cursor_run_id])
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"code": "INVALID_CURSOR", "message": str(e)}
                )

        # Date range filters
        if date_from:
            where_clauses.append("timestamp >= ?")
            params.append(date_from)
        if date_to:
            where_clauses.append("timestamp <= ?")
            params.append(date_to)

        # Exact match filters
        if provider:
            where_clauses.append("provider_name = ?")
            params.append(provider)
        if model:
            where_clauses.append("provider_model = ?")
            params.append(model)
        if status:
            where_clauses.append("status = ?")
            params.append(status)

        # Substring search (case-insensitive)
        if filename_contains:
            where_clauses.append("LOWER(input_filename) LIKE LOWER(?)")
            params.append(f"%{filename_contains}%")

        # Combine WHERE clauses
        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

        # Compute total count (always, as per user requirement)
        count_query = f"SELECT COUNT(*) FROM pipeline_runs WHERE {where_sql}"
        total_count = db.execute(count_query, params).fetchone()[0]

        # Build main query
        # Select only fields needed for RunSummary (excludes config_snapshot)
        select_fields = [
            "run_id", "timestamp",
            "parser_name", "provider_name", "provider_model",
            "input_filename", "input_size_bytes",
            "total_seconds", "extractor_seconds",
            "events_extracted", "status"
        ]

        query = f"""
            SELECT {', '.join(select_fields)}
            FROM pipeline_runs
            WHERE {where_sql}
            ORDER BY {sort_field} {sort_direction.upper()}, run_id {sort_direction.upper()}
            LIMIT ?
        """
        params.append(limit)

        # Execute query
        rows = db.execute(query, params).fetchall()

        # Convert rows to RunSummary models
        runs = []
        for row in rows:
            run_dict = {
                field: value for field, value in zip(select_fields, row)
            }
            runs.append(RunSummary(**run_dict))

        # Generate next cursor if there are more results
        next_cursor = None
        if len(runs) == limit:
            last_run = runs[-1]
            next_cursor = encode_cursor(last_run.timestamp, last_run.run_id)

        # Build response envelope
        base_url = str(request.url).split('?')[0]
        response = ListResponse(
            data=runs,
            meta=ResponseMeta(
                returned=len(runs),
                total=total_count,
                next_cursor=next_cursor
            ),
            links=ResponseLinks(
                self=str(request.url),
                next=(
                    f"{base_url}?limit={limit}&cursor={next_cursor}"
                    if next_cursor else None
                )
            )
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        # Log error for debugging
        print(f"Error listing runs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "INTERNAL_ERROR", "message": str(e)}
        )


@router.get(
    "/runs/{run_id}",
    response_model=DetailResponse[RunDetail],
    summary="Get Run by ID",
    description="Fetch detailed information for a single pipeline run, including sanitized config_snapshot.",
    responses={
        200: {"description": "Run found"},
        404: {"description": "Run not found", "model": ErrorResponse}
    }
)
def get_run(
    run_id: str,
    request: Request,
    db: duckdb.DuckDBPyConnection = Depends(get_db)
):
    """
    Get detailed information for a single run.

    Path Parameters:
    - run_id: Unique pipeline execution identifier

    Returns:
    - Full run details with sanitized config_snapshot (sensitive keys removed)

    Raises:
    - 404: Run with specified ID not found
    """
    try:
        # Query for all fields
        query = """
            SELECT
                run_id, timestamp,
                parser_name, parser_version, provider_name, provider_model,
                ocr_engine, table_mode,
                environment, session_label,
                input_filename, input_size_bytes, input_pages, output_path,
                docling_seconds, extractor_seconds, total_seconds,
                events_extracted, citations_found, avg_detail_length,
                status, error_message,
                config_snapshot,
                cost_usd, tokens_input, tokens_output
            FROM pipeline_runs
            WHERE run_id = ?
        """

        row = db.execute(query, [run_id]).fetchone()

        if not row:
            # Return 404 with structured error
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "NOT_FOUND",
                    "message": f"Run with id '{run_id}' not found"
                }
            )

        # Map row to dictionary
        field_names = [
            "run_id", "timestamp",
            "parser_name", "parser_version", "provider_name", "provider_model",
            "ocr_engine", "table_mode",
            "environment", "session_label",
            "input_filename", "input_size_bytes", "input_pages", "output_path",
            "docling_seconds", "extractor_seconds", "total_seconds",
            "events_extracted", "citations_found", "avg_detail_length",
            "status", "error_message",
            "config_snapshot",
            "cost_usd", "tokens_input", "tokens_output"
        ]

        run_dict = {field: value for field, value in zip(field_names, row)}

        # Sanitize config_snapshot (remove sensitive keys)
        if run_dict['config_snapshot']:
            # Parse JSON string if needed
            config = run_dict['config_snapshot']
            if isinstance(config, str):
                config = json.loads(config)
            run_dict['config_snapshot'] = sanitize_config_snapshot(config)

        # Create RunDetail model
        run = RunDetail(**run_dict)

        # Build response envelope
        response = DetailResponse(
            data=run,
            links=ResponseLinks(
                self=str(request.url)
            )
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error fetching run {run_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "INTERNAL_ERROR", "message": str(e)}
        )
