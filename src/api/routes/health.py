"""
Health Check Endpoints

Provides service health, readiness, and version information.

Order: fastapi-duckdb-api-001
"""

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from pathlib import Path

from src.api.config import settings
from src.api.database import db_manager

router = APIRouter(tags=["Health"])


@router.get(
    "/healthz",
    summary="Service Health Check",
    description="Returns 200 if the service is running. Does not check database connectivity.",
    status_code=status.HTTP_200_OK
)
def healthz():
    """
    Service health check endpoint.

    Always returns 200 if the app is up, regardless of database state.
    Use /readyz to check database readiness.

    Returns:
        dict: {"status": "healthy"}
    """
    return {
        "status": "healthy",
        "service": "pipeline-analytics-api"
    }


@router.get(
    "/readyz",
    summary="Service Readiness Check",
    description="Returns 200 if service and database are ready, 503 if database is unavailable.",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Service and database are ready"},
        503: {"description": "Database is unavailable"}
    }
)
def readyz():
    """
    Service readiness check endpoint.

    Verifies database file exists and schema is valid.
    Returns 503 if database is missing or unhealthy.

    Returns:
        dict: {"status": "ready"} or {"status": "not_ready", "reason": "..."}
    """
    # Check if database file exists
    db_path = Path(settings.runs_db_path)
    if not db_path.exists():
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "not_ready",
                "reason": f"Database file not found: {db_path}",
                "hint": (
                    "Run ingestion script to create the database:\n"
                    f"  uv run python scripts/ingest_metadata_to_duckdb.py "
                    f"--db {db_path} --glob 'output/**/*_metadata.json'"
                )
            }
        )

    # Check if database schema is valid
    if not db_manager.health_check():
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "not_ready",
                "reason": "Database schema validation failed",
                "hint": "Database exists but pipeline_runs table not found"
            }
        )

    return {
        "status": "ready",
        "database": str(db_path),
        "schema_valid": True
    }


@router.get(
    "/version",
    summary="API Version",
    description="Returns API version and configuration information.",
    status_code=status.HTTP_200_OK
)
def version():
    """
    API version endpoint.

    Returns:
        dict: Version and configuration metadata
    """
    return {
        "api_version": settings.api_version,
        "api_title": settings.api_title,
        "database_path": settings.runs_db_path,
        "cors_enabled": bool(settings.cors_origins_list)
    }
