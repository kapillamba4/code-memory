"""
Custom exception hierarchy for code-memory.

All exceptions inherit from CodeMemoryError for easy catching.
Each exception type maps to a specific error category for
structured error responses to MCP clients.
"""

from __future__ import annotations


class CodeMemoryError(Exception):
    """Base exception for all code-memory errors.

    All custom exceptions should inherit from this class.
    Provides a consistent interface for error handling.
    """

    def __init__(self, message: str, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> dict:
        """Convert exception to structured error response dict."""
        return {
            "error": True,
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details if self.details else None,
        }


class DatabaseError(CodeMemoryError):
    """Database operation failed.

    Raised when:
    - Database file not found or corrupted
    - SQL execution fails
    - sqlite-vec operations fail
    - Schema migration errors
    """
    pass


class IndexingError(CodeMemoryError):
    """Code or documentation indexing failed.

    Raised when:
    - File parsing fails
    - Embedding generation fails
    - File read errors
    - Unsupported file type errors
    """
    pass


class GitError(CodeMemoryError):
    """Git operation failed.

    Raised when:
    - Not a git repository
    - Git command fails
    - Invalid commit hash
    - File not in git history
    """
    pass


class ValidationError(CodeMemoryError):
    """Input validation failed.

    Raised when:
    - Empty or invalid query
    - Invalid search_type
    - Path traversal attempt
    - Invalid line numbers
    - File/directory not found
    """
    pass


class EmbeddingError(CodeMemoryError):
    """Embedding model operation failed.

    Raised when:
    - Model fails to load
    - Embedding generation fails
    - Vector dimension mismatch
    """
    pass


def format_error(error: Exception) -> dict:
    """Format any exception as a structured error response.

    Args:
        error: Any exception (CodeMemoryError or built-in)

    Returns:
        Structured error dict suitable for MCP response
    """
    if isinstance(error, CodeMemoryError):
        return error.to_dict()

    # Handle common built-in exceptions
    error_type = error.__class__.__name__
    message = str(error) or f"An error of type {error_type} occurred"

    return {
        "error": True,
        "error_type": error_type,
        "message": message,
        "details": None,
    }
