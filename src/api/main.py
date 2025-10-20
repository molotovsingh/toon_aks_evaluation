"""
FastAPI Analytics API - Main Application

Read-only REST API for querying pipeline execution metadata from DuckDB.

Order: fastapi-duckdb-api-001
"""

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.config import settings
from src.api.database import shutdown_db
from src.api.routes import health, runs, stats

# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
if settings.cors_origins_list:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["GET"],  # Read-only API
        allow_headers=["*"],
    )

# Register routers
app.include_router(health.router)
app.include_router(runs.router)
app.include_router(stats.router)


# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle 422 validation errors and convert to 400 Bad Request.

    Returns structured error response with details about validation failures.
    """
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Invalid request parameters",
                "details": exc.errors()
            }
        }
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """
    Handle HTTP exceptions and ensure consistent error response format.

    Catches HTTPException raised in routes and returns structured error.
    """
    # If detail is already a dict (structured error), use it directly
    if isinstance(exc.detail, dict):
        error_content = {"error": exc.detail}
    else:
        # Convert string detail to structured format
        error_content = {
            "error": {
                "code": f"HTTP_{exc.status_code}",
                "message": exc.detail or "An error occurred"
            }
        }

    return JSONResponse(
        status_code=exc.status_code,
        content=error_content
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Catch-all handler for unexpected exceptions.

    Returns 500 Internal Server Error with generic message
    (don't expose internal details to clients).
    """
    # Log the full exception for debugging
    import traceback
    print(f"Unhandled exception: {exc}")
    print(traceback.format_exc())

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An internal server error occurred",
                "details": {"type": type(exc).__name__}
            }
        }
    )


# ============================================================================
# Lifecycle Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Application startup event handler.

    Performs initialization tasks when the app starts.
    """
    print(f"🚀 Starting {settings.api_title} v{settings.api_version}")
    print(f"📊 Database: {settings.runs_db_path}")
    if settings.cors_origins_list:
        print(f"🔓 CORS enabled for: {', '.join(settings.cors_origins_list)}")
    else:
        print("🔒 CORS disabled (local access only)")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Application shutdown event handler.

    Cleans up resources when the app shuts down.
    """
    print("🛑 Shutting down API...")
    shutdown_db()
    print("✅ Database connection closed")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level="info"
    )
