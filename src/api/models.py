"""
Pydantic Models for API Responses

Defines request/response schemas for all API endpoints.

Order: fastapi-duckdb-api-001
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Generic, TypeVar
from pydantic import BaseModel, Field


# ============================================================================
# Run Models
# ============================================================================

class RunSummary(BaseModel):
    """
    Summary view of a pipeline run (for list endpoints).

    Excludes config_snapshot for performance - use RunDetail to get full config.
    """
    run_id: str = Field(..., description="Unique pipeline execution identifier")
    timestamp: datetime = Field(..., description="Pipeline execution start time (UTC)")

    # Configuration
    parser_name: str = Field(..., description="Document parser name")
    provider_name: str = Field(..., description="Event extraction provider")
    provider_model: Optional[str] = Field(None, description="Model identifier")

    # Input
    input_filename: str = Field(..., description="Original filename")
    input_size_bytes: Optional[int] = Field(None, description="File size in bytes")

    # Performance
    total_seconds: Optional[float] = Field(None, description="Total pipeline duration")
    extractor_seconds: Optional[float] = Field(None, description="Event extraction duration")

    # Quality
    events_extracted: Optional[int] = Field(None, description="Number of events extracted")
    status: str = Field(..., description="Run status (success, failed, partial)")

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "run_id": "DL2-OA2-TS1-F-20251018120000",
                "timestamp": "2025-10-18T12:00:00",
                "parser_name": "docling",
                "provider_name": "openai",
                "provider_model": "gpt-4o-mini",
                "input_filename": "contract.pdf",
                "input_size_bytes": 166033,
                "total_seconds": 6.7,
                "extractor_seconds": 5.2,
                "events_extracted": 10,
                "status": "success"
            }
        }


class RunDetail(BaseModel):
    """
    Detailed view of a single pipeline run.

    Includes all fields and sanitized config_snapshot.
    """
    # Primary key
    run_id: str = Field(..., description="Unique pipeline execution identifier")
    timestamp: datetime = Field(..., description="Pipeline execution start time (UTC)")

    # Configuration
    parser_name: str = Field(..., description="Document parser name")
    parser_version: Optional[str] = Field(None, description="Parser version")
    provider_name: str = Field(..., description="Event extraction provider")
    provider_model: Optional[str] = Field(None, description="Model identifier")
    ocr_engine: Optional[str] = Field(None, description="OCR engine used")
    table_mode: Optional[str] = Field(None, description="Table extraction mode")

    # Environment
    environment: Optional[str] = Field(None, description="Execution environment hostname")
    session_label: Optional[str] = Field(None, description="Batch processing label")

    # Input metadata
    input_filename: str = Field(..., description="Original filename")
    input_size_bytes: Optional[int] = Field(None, description="File size in bytes")
    input_pages: Optional[int] = Field(None, description="Number of pages")
    output_path: Optional[str] = Field(None, description="Output file path")

    # Performance metrics
    docling_seconds: Optional[float] = Field(None, description="Document parsing duration")
    extractor_seconds: Optional[float] = Field(None, description="Event extraction duration")
    total_seconds: Optional[float] = Field(None, description="Total pipeline duration")

    # Quality metrics
    events_extracted: Optional[int] = Field(None, description="Number of events extracted")
    citations_found: Optional[int] = Field(None, description="Number of citations found")
    avg_detail_length: Optional[float] = Field(None, description="Average event description length")

    # Status
    status: str = Field(..., description="Run status (success, failed, partial)")
    error_message: Optional[str] = Field(None, description="Error details if failed")

    # Config snapshot (sanitized - sensitive keys removed)
    config_snapshot: Optional[Dict[str, Any]] = Field(None, description="Sanitized configuration snapshot")

    # Cost tracking
    cost_usd: Optional[float] = Field(None, description="Estimated cost in USD")
    tokens_input: Optional[int] = Field(None, description="Input tokens consumed")
    tokens_output: Optional[int] = Field(None, description="Output tokens generated")

    class Config:
        from_attributes = True


# ============================================================================
# Stats Models
# ============================================================================

class StatsItem(BaseModel):
    """
    Aggregation statistics for a provider or model.
    """
    # Identifier
    key_name: str = Field(..., description="Provider or model name")

    # Counts
    total_runs: int = Field(..., description="Total number of runs")

    # Performance metrics (extractor)
    avg_extractor_seconds: Optional[float] = Field(None, description="Average extraction time")
    min_extractor_seconds: Optional[float] = Field(None, description="Minimum extraction time")
    max_extractor_seconds: Optional[float] = Field(None, description="Maximum extraction time")

    # Performance metrics (total)
    avg_total_seconds: Optional[float] = Field(None, description="Average total time")
    min_total_seconds: Optional[float] = Field(None, description="Minimum total time")
    max_total_seconds: Optional[float] = Field(None, description="Maximum total time")

    class Config:
        json_schema_extra = {
            "example": {
                "key_name": "openai/gpt-4o-mini",
                "total_runs": 23,
                "avg_extractor_seconds": 6.1,
                "min_extractor_seconds": 2.9,
                "max_extractor_seconds": 11.5,
                "avg_total_seconds": 6.1,
                "min_total_seconds": 3.0,
                "max_total_seconds": 11.6
            }
        }


# ============================================================================
# Response Envelopes
# ============================================================================

T = TypeVar('T')


class ResponseMeta(BaseModel):
    """Metadata about the response"""
    returned: int = Field(..., description="Number of items in this response")
    total: Optional[int] = Field(None, description="Total matching items (if computed)")
    next_cursor: Optional[str] = Field(None, description="Cursor for next page")


class ResponseLinks(BaseModel):
    """Hypermedia links for navigation"""
    self: str = Field(..., description="Current request URL")
    next: Optional[str] = Field(None, description="Next page URL")


class ListResponse(BaseModel, Generic[T]):
    """Generic list response envelope"""
    data: List[T] = Field(..., description="List of items")
    meta: ResponseMeta = Field(..., description="Response metadata")
    links: ResponseLinks = Field(..., description="Navigation links")


class DetailResponse(BaseModel, Generic[T]):
    """Generic detail response envelope"""
    data: T = Field(..., description="Single item")
    links: ResponseLinks = Field(..., description="Navigation links")


# ============================================================================
# Error Models
# ============================================================================

class ErrorDetail(BaseModel):
    """Error response detail"""
    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error context")

    class Config:
        json_schema_extra = {
            "example": {
                "code": "NOT_FOUND",
                "message": "Run with id 'XYZ' not found",
                "details": {"run_id": "XYZ"}
            }
        }


class ErrorResponse(BaseModel):
    """Error response envelope"""
    error: ErrorDetail = Field(..., description="Error information")


# ============================================================================
# Request Query Parameter Models
# ============================================================================

class RunFilters(BaseModel):
    """Query parameters for filtering runs"""
    limit: int = Field(50, ge=1, le=500, description="Maximum results per page")
    cursor: Optional[str] = Field(None, description="Pagination cursor")
    date_from: Optional[datetime] = Field(None, description="Filter runs after this timestamp (UTC)")
    date_to: Optional[datetime] = Field(None, description="Filter runs before this timestamp (UTC)")
    provider: Optional[str] = Field(None, description="Filter by provider name")
    model: Optional[str] = Field(None, description="Filter by model identifier")
    status: Optional[str] = Field(None, description="Filter by status")
    filename_contains: Optional[str] = Field(None, description="Filter by filename substring")
    sort: str = Field("timestamp:desc", description="Sort field and direction (e.g., 'timestamp:desc')")


class StatsFilters(BaseModel):
    """Query parameters for stats endpoints"""
    date_from: Optional[datetime] = Field(None, description="Filter runs after this timestamp (UTC)")
    date_to: Optional[datetime] = Field(None, description="Filter runs before this timestamp (UTC)")
    status: Optional[str] = Field(None, description="Filter by status")
