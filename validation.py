"""
Input validation for code-memory tools.

Provides validation functions for all tool parameters with
clear error messages and protection against common attacks.
"""

from __future__ import annotations

import re
from pathlib import Path

from errors import ValidationError


def validate_directory(path: str, must_exist: bool = True) -> Path:
    """Validate that path exists and is a directory.

    Args:
        path: Directory path to validate
        must_exist: If True, directory must exist

    Returns:
        Resolved Path object

    Raises:
        ValidationError: If path is invalid or not a directory
    """
    if not path or not path.strip():
        raise ValidationError("Directory path cannot be empty")

    try:
        resolved = Path(path).resolve()
    except Exception as e:
        raise ValidationError(f"Invalid directory path: {path}", {"exception": str(e)})

    if must_exist and not resolved.exists():
        raise ValidationError(f"Directory not found: {path}")

    if must_exist and not resolved.is_dir():
        raise ValidationError(f"Path is not a directory: {path}")

    return resolved


def validate_file(path: str, must_exist: bool = True) -> Path:
    """Validate that path exists and is a file.

    Args:
        path: File path to validate
        must_exist: If True, file must exist

    Returns:
        Resolved Path object

    Raises:
        ValidationError: If path is invalid or not a file
    """
    if not path or not path.strip():
        raise ValidationError("File path cannot be empty")

    try:
        resolved = Path(path).resolve()
    except Exception as e:
        raise ValidationError(f"Invalid file path: {path}", {"exception": str(e)})

    if must_exist and not resolved.exists():
        raise ValidationError(f"File not found: {path}")

    if must_exist and not resolved.is_file():
        raise ValidationError(f"Path is not a file: {path}")

    return resolved


def validate_query(query: str, min_length: int = 1, max_length: int = 1000) -> str:
    """Validate and sanitize query string.

    Args:
        query: Query string to validate
        min_length: Minimum allowed length
        max_length: Maximum allowed length

    Returns:
        Sanitized query string (stripped whitespace)

    Raises:
        ValidationError: If query is invalid
    """
    if query is None:
        raise ValidationError("Query cannot be None")

    # Strip whitespace
    sanitized = query.strip()

    if len(sanitized) < min_length:
        raise ValidationError(
            f"Query too short (minimum {min_length} characters)",
            {"length": len(sanitized), "minimum": min_length}
        )

    if len(sanitized) > max_length:
        raise ValidationError(
            f"Query too long (maximum {max_length} characters)",
            {"length": len(sanitized), "maximum": max_length}
        )

    return sanitized


def validate_search_type(search_type: str, allowed: list[str]) -> str:
    """Validate search_type is in allowed values.

    Args:
        search_type: Search type to validate
        allowed: List of allowed values

    Returns:
        Validated search_type

    Raises:
        ValidationError: If search_type is not allowed
    """
    if not search_type:
        raise ValidationError(
            "Search type is required",
            {"allowed_values": allowed}
        )

    if search_type not in allowed:
        raise ValidationError(
            f"Invalid search type: '{search_type}'",
            {"allowed_values": allowed, "provided": search_type}
        )

    return search_type


def validate_line_number(value: int | None, name: str, min_val: int = 1) -> int | None:
    """Validate a line number parameter.

    Args:
        value: Line number to validate (None is allowed)
        name: Parameter name for error messages
        min_val: Minimum allowed value

    Returns:
        Validated line number or None

    Raises:
        ValidationError: If line number is invalid
    """
    if value is None:
        return None

    if not isinstance(value, int):
        raise ValidationError(
            f"{name} must be an integer",
            {"provided_type": type(value).__name__}
        )

    if value < min_val:
        raise ValidationError(
            f"{name} must be >= {min_val}",
            {"provided": value, "minimum": min_val}
        )

    return value


def validate_line_range(line_start: int | None, line_end: int | None) -> tuple[int | None, int | None]:
    """Validate a line range (start must be <= end if both provided).

    Args:
        line_start: Start line number
        line_end: End line number

    Returns:
        Tuple of (validated_line_start, validated_line_end)

    Raises:
        ValidationError: If range is invalid
    """
    start = validate_line_number(line_start, "line_start")
    end = validate_line_number(line_end, "line_end")

    if start is not None and end is not None and start > end:
        raise ValidationError(
            "line_start cannot be greater than line_end",
            {"line_start": start, "line_end": end}
        )

    return start, end


def validate_top_k(value: int, min_val: int = 1, max_val: int = 100, default: int = 10) -> int:
    """Validate top_k parameter with default.

    Args:
        value: top_k value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        default: Default value if not provided (0 or None)

    Returns:
        Validated top_k value

    Raises:
        ValidationError: If value is out of range
    """
    if value is None or value == 0:
        return default

    if not isinstance(value, int):
        raise ValidationError(
            "top_k must be an integer",
            {"provided_type": type(value).__name__}
        )

    if value < min_val:
        raise ValidationError(
            f"top_k must be >= {min_val}",
            {"provided": value, "minimum": min_val}
        )

    if value > max_val:
        raise ValidationError(
            f"top_k must be <= {max_val}",
            {"provided": value, "maximum": max_val}
        )

    return value


def validate_path_in_directory(path: str, base_dir: str) -> Path:
    """Validate that a path is within a base directory (prevent path traversal).

    Args:
        path: Path to validate
        base_dir: Base directory that path must be within

    Returns:
        Resolved Path object

    Raises:
        ValidationError: If path escapes base directory
    """
    if not path:
        raise ValidationError("Path cannot be empty")

    try:
        resolved_path = Path(path).resolve()
        resolved_base = Path(base_dir).resolve()
    except Exception as e:
        raise ValidationError(f"Invalid path: {path}", {"exception": str(e)})

    try:
        resolved_path.relative_to(resolved_base)
    except ValueError:
        raise ValidationError(
            f"Path escapes base directory: {path}",
            {"base_directory": str(resolved_base)}
        )

    return resolved_path


def sanitize_fts_query(query: str) -> str:
    """Sanitize a query for FTS5 MATCH.

    Escapes special FTS5 characters to prevent query errors.

    Args:
        query: Query string to sanitize

    Returns:
        Sanitized query safe for FTS5 MATCH
    """
    # FTS5 special characters that need escaping
    # In FTS5, double-quotes are escaped by doubling them
    sanitized = query.replace('"', '""')

    # Wrap in quotes if contains special characters
    special_chars = {'-', '*', '^', '(', ')', ':', '"'}
    if any(c in query for c in special_chars):
        sanitized = f'"{sanitized}"'

    return sanitized


def validate_commit_hash(hash_str: str) -> str:
    """Validate a git commit hash.

    Args:
        hash_str: Commit hash to validate

    Returns:
        Validated commit hash

    Raises:
        ValidationError: If hash format is invalid
    """
    if not hash_str:
        raise ValidationError("Commit hash cannot be empty")

    # Git hashes are 40 hex characters (full) or 7+ (short)
    sanitized = hash_str.strip().lower()

    if not re.match(r'^[0-9a-f]{7,40}$', sanitized):
        raise ValidationError(
            f"Invalid commit hash format: {hash_str}",
            {"expected": "7-40 hexadecimal characters"}
        )

    return sanitized
