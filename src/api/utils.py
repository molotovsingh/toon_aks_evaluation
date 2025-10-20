"""
Utility Functions for API

Cursor encoding/decoding and config sanitization helpers.

Order: fastapi-duckdb-api-001
"""

import base64
from datetime import datetime
from typing import Dict, Any, Tuple


def encode_cursor(timestamp: datetime, run_id: str) -> str:
    """
    Encode cursor for pagination.

    Format: base64("timestamp_iso|run_id")

    Args:
        timestamp: Timestamp of the row
        run_id: Run ID of the row

    Returns:
        Base64-encoded cursor string

    Example:
        >>> encode_cursor(datetime(2025, 10, 18, 12, 0, 0), "DL2-OA2-TS1-F-20251018120000")
        'MjAyNS0xMC0xOFQxMjowMDowMHxETDItT0EyLVRTMS1GLTIwMjUxMDE4MTIwMDAw'
    """
    payload = f"{timestamp.isoformat()}|{run_id}"
    return base64.urlsafe_b64encode(payload.encode()).decode()


def decode_cursor(cursor: str) -> Tuple[datetime, str]:
    """
    Decode pagination cursor.

    Args:
        cursor: Base64-encoded cursor string

    Returns:
        Tuple of (timestamp, run_id)

    Raises:
        ValueError: If cursor is invalid or malformed

    Example:
        >>> decode_cursor('MjAyNS0xMC0xOFQxMjowMDowMHxETDItT0EyLVRTMS1GLTIwMjUxMDE4MTIwMDAw')
        (datetime.datetime(2025, 10, 18, 12, 0), 'DL2-OA2-TS1-F-20251018120000')
    """
    try:
        payload = base64.urlsafe_b64decode(cursor.encode()).decode()
        timestamp_str, run_id = payload.split('|', 1)
        timestamp = datetime.fromisoformat(timestamp_str)
        return timestamp, run_id
    except (ValueError, UnicodeDecodeError, AttributeError) as e:
        raise ValueError(f"Invalid cursor format: {e}")


def sanitize_config_snapshot(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove sensitive keys from config_snapshot.

    Redacts fields matching patterns (case-insensitive):
    - api_key
    - token
    - secret
    - password
    - key (standalone)
    - bearer

    Args:
        config: Raw config_snapshot dictionary

    Returns:
        Sanitized config dictionary with sensitive keys removed

    Example:
        >>> config = {"provider": "openai", "api_key": "sk-xxx", "model": "gpt-4"}
        >>> sanitize_config_snapshot(config)
        {'provider': 'openai', 'model': 'gpt-4'}
    """
    if not config:
        return {}

    sensitive_patterns = ['api_key', 'token', 'secret', 'password', 'bearer']

    sanitized = {}
    for key, value in config.items():
        key_lower = key.lower()

        # Check if key contains sensitive patterns
        is_sensitive = any(pattern in key_lower for pattern in sensitive_patterns)

        # Special case: standalone "key" but not in compound words like "session_key_id"
        if not is_sensitive and key_lower == 'key':
            is_sensitive = True

        if not is_sensitive:
            sanitized[key] = value

    return sanitized


def parse_sort_param(sort: str, allowed_fields: set) -> Tuple[str, str]:
    """
    Parse and validate sort parameter.

    Format: "field:direction" or "field" (defaults to desc)
    Validates field against whitelist.

    Args:
        sort: Sort parameter string (e.g., "timestamp:asc", "total_seconds")
        allowed_fields: Set of allowed field names

    Returns:
        Tuple of (field_name, direction)

    Raises:
        ValueError: If field not in whitelist or direction invalid

    Example:
        >>> parse_sort_param("timestamp:desc", {"timestamp", "provider_name"})
        ('timestamp', 'desc')
        >>> parse_sort_param("total_seconds", {"timestamp", "total_seconds"})
        ('total_seconds', 'desc')
    """
    # Parse format
    if ':' in sort:
        field, direction = sort.split(':', 1)
    else:
        field, direction = sort, 'desc'

    # Validate field
    if field not in allowed_fields:
        raise ValueError(
            f"Invalid sort field '{field}'. "
            f"Allowed: {', '.join(sorted(allowed_fields))}"
        )

    # Validate direction
    direction = direction.lower()
    if direction not in ['asc', 'desc']:
        raise ValueError(
            f"Invalid sort direction '{direction}'. "
            f"Allowed: asc, desc"
        )

    return field, direction
