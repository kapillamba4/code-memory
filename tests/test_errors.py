"""Tests for error handling module."""

from __future__ import annotations

from errors import (
    CodeMemoryError,
    DatabaseError,
    EmbeddingError,
    GitError,
    IndexingError,
    ValidationError,
    format_error,
)


class TestCodeMemoryError:
    """Tests for base CodeMemoryError class."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = CodeMemoryError("Test error")
        assert error.message == "Test error"
        assert str(error) == "Test error"

    def test_error_with_details(self):
        """Test error with additional details."""
        error = CodeMemoryError("Test error", {"key": "value"})
        assert error.details == {"key": "value"}

    def test_to_dict(self):
        """Test conversion to dict."""
        error = CodeMemoryError("Test error")
        result = error.to_dict()
        assert result["error"] is True
        assert result["error_type"] == "CodeMemoryError"
        assert result["message"] == "Test error"

    def test_to_dict_with_details(self):
        """Test conversion to dict with details."""
        error = CodeMemoryError("Test error", {"key": "value"})
        result = error.to_dict()
        assert result["details"] == {"key": "value"}

    def test_to_dict_without_details(self):
        """Test that None details are preserved."""
        error = CodeMemoryError("Test error", {})
        result = error.to_dict()
        assert result["details"] is None


class TestSpecializedErrors:
    """Tests for specialized error classes."""

    def test_database_error(self):
        """Test DatabaseError."""
        error = DatabaseError("Connection failed")
        assert isinstance(error, CodeMemoryError)
        assert error.to_dict()["error_type"] == "DatabaseError"

    def test_indexing_error(self):
        """Test IndexingError."""
        error = IndexingError("Parse failed")
        assert isinstance(error, CodeMemoryError)
        assert error.to_dict()["error_type"] == "IndexingError"

    def test_git_error(self):
        """Test GitError."""
        error = GitError("Not a git repo")
        assert isinstance(error, CodeMemoryError)
        assert error.to_dict()["error_type"] == "GitError"

    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError("Invalid input")
        assert isinstance(error, CodeMemoryError)
        assert error.to_dict()["error_type"] == "ValidationError"

    def test_embedding_error(self):
        """Test EmbeddingError."""
        error = EmbeddingError("Model load failed")
        assert isinstance(error, CodeMemoryError)
        assert error.to_dict()["error_type"] == "EmbeddingError"


class TestFormatError:
    """Tests for format_error function."""

    def test_format_code_memory_error(self):
        """Test formatting CodeMemoryError."""
        error = ValidationError("Invalid input", {"field": "query"})
        result = format_error(error)
        assert result["error"] is True
        assert result["error_type"] == "ValidationError"
        assert result["message"] == "Invalid input"
        assert result["details"] == {"field": "query"}

    def test_format_builtin_exception(self):
        """Test formatting built-in exceptions."""
        error = ValueError("Something went wrong")
        result = format_error(error)
        assert result["error"] is True
        assert result["error_type"] == "ValueError"
        assert result["message"] == "Something went wrong"

    def test_format_exception_without_message(self):
        """Test formatting exception with empty message."""
        error = RuntimeError()
        result = format_error(error)
        assert result["error"] is True
        assert "RuntimeError" in result["message"]
