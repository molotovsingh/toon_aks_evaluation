"""
API Configuration via Environment Variables

Uses pydantic-settings for type-safe configuration management.
All settings can be overridden via environment variables.

Order: fastapi-duckdb-api-001
"""

from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """API Configuration Settings"""

    # Database
    runs_db_path: str  # REQUIRED - Path to DuckDB file (e.g., "runs.duckdb")

    # Server
    api_port: int = 8000
    api_host: str = "0.0.0.0"

    # CORS
    cors_allow_origins: str = ""  # Comma-separated list of allowed origins

    # API Metadata
    api_title: str = "Pipeline Analytics API"
    api_version: str = "0.1.0"
    api_description: str = """
    Read-only REST API for querying pipeline execution metadata stored in DuckDB.

    Provides endpoints for listing runs, fetching run details, and computing
    model/provider statistics.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS_ALLOW_ORIGINS comma-separated string into list"""
        if not self.cors_allow_origins:
            return []
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]


# Global settings instance
settings = Settings()
