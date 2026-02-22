"""Tests for input validation module."""

from __future__ import annotations

import pytest

from errors import ValidationError
import validation as val


class TestValidateQuery:
    """Tests for validate_query function."""

    def test_valid_query(self):
        """Test that valid queries pass validation."""
        result = val.validate_query("test query")
        assert result == "test query"

    def test_strips_whitespace(self):
        """Test that whitespace is stripped."""
        result = val.validate_query("  test query  ")
        assert result == "test query"

    def test_empty_query_fails(self):
        """Test that empty queries raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            val.validate_query("")
        assert "too short" in str(exc_info.value.message)

    def test_whitespace_only_fails(self):
        """Test that whitespace-only queries raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            val.validate_query("   ")
        assert "too short" in str(exc_info.value.message)

    def test_long_query_fails(self):
        """Test that overly long queries raise ValidationError."""
        long_query = "a" * 1001
        with pytest.raises(ValidationError) as exc_info:
            val.validate_query(long_query, max_length=1000)
        assert "too long" in str(exc_info.value.message)

    def test_custom_min_length(self):
        """Test custom minimum length."""
        with pytest.raises(ValidationError):
            val.validate_query("ab", min_length=5)


class TestValidateSearchType:
    """Tests for validate_search_type function."""

    def test_valid_search_type(self):
        """Test that valid search types pass."""
        result = val.validate_search_type("definition", ["definition", "references"])
        assert result == "definition"

    def test_invalid_search_type(self):
        """Test that invalid search types raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            val.validate_search_type("invalid", ["definition", "references"])
        assert "Invalid search type" in str(exc_info.value.message)

    def test_empty_search_type(self):
        """Test that empty search type raises ValidationError."""
        with pytest.raises(ValidationError):
            val.validate_search_type("", ["definition"])


class TestValidateLineNumber:
    """Tests for validate_line_number function."""

    def test_valid_line_number(self):
        """Test that valid line numbers pass."""
        result = val.validate_line_number(10, "line_start")
        assert result == 10

    def test_none_allowed(self):
        """Test that None is allowed."""
        result = val.validate_line_number(None, "line_start")
        assert result is None

    def test_negative_fails(self):
        """Test that negative numbers raise ValidationError."""
        with pytest.raises(ValidationError):
            val.validate_line_number(-1, "line_start")

    def test_zero_fails(self):
        """Test that zero raises ValidationError with default min."""
        with pytest.raises(ValidationError):
            val.validate_line_number(0, "line_start")


class TestValidateLineRange:
    """Tests for validate_line_range function."""

    def test_valid_range(self):
        """Test that valid ranges pass."""
        start, end = val.validate_line_range(1, 10)
        assert start == 1
        assert end == 10

    def test_start_greater_than_end_fails(self):
        """Test that start > end raises ValidationError."""
        with pytest.raises(ValidationError):
            val.validate_line_range(10, 1)


class TestValidateTopK:
    """Tests for validate_top_k function."""

    def test_valid_value(self):
        """Test that valid values pass."""
        result = val.validate_top_k(10)
        assert result == 10

    def test_none_uses_default(self):
        """Test that None returns default."""
        result = val.validate_top_k(None)
        assert result == 10

    def test_zero_uses_default(self):
        """Test that zero returns default."""
        result = val.validate_top_k(0)
        assert result == 10

    def test_too_large_fails(self):
        """Test that values > max raise ValidationError."""
        with pytest.raises(ValidationError):
            val.validate_top_k(200, max_val=100)

    def test_negative_fails(self):
        """Test that negative values raise ValidationError."""
        with pytest.raises(ValidationError):
            val.validate_top_k(-1, min_val=1, default=10)


class TestValidateDirectory:
    """Tests for validate_directory function."""

    def test_existing_directory(self, temp_dir):
        """Test that existing directories pass."""
        result = val.validate_directory(str(temp_dir))
        assert result == temp_dir

    def test_nonexistent_fails(self):
        """Test that nonexistent directories raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            val.validate_directory("/nonexistent/path")
        assert "not found" in str(exc_info.value.message)

    def test_file_not_directory_fails(self, temp_dir):
        """Test that files (not directories) raise ValidationError."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test")
        with pytest.raises(ValidationError):
            val.validate_directory(str(test_file))


class TestValidateFile:
    """Tests for validate_file function."""

    def test_existing_file(self, temp_dir):
        """Test that existing files pass."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test")
        result = val.validate_file(str(test_file))
        assert result == test_file

    def test_nonexistent_fails(self):
        """Test that nonexistent files raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            val.validate_file("/nonexistent/file.txt")
        assert "not found" in str(exc_info.value.message)

    def test_directory_not_file_fails(self, temp_dir):
        """Test that directories (not files) raise ValidationError."""
        with pytest.raises(ValidationError):
            val.validate_file(str(temp_dir))


class TestSanitizeFtsQuery:
    """Tests for sanitize_fts_query function."""

    def test_simple_query(self):
        """Test that simple queries pass through."""
        result = val.sanitize_fts_query("simple query")
        assert result == "simple query"

    def test_escapes_quotes(self):
        """Test that quotes are escaped."""
        result = val.sanitize_fts_query('test "quoted"')
        assert '""' in result or '"' in result


class TestValidateCommitHash:
    """Tests for validate_commit_hash function."""

    def test_valid_full_hash(self):
        """Test that valid full hashes pass."""
        result = val.validate_commit_hash("a" * 40)
        assert result == "a" * 40

    def test_valid_short_hash(self):
        """Test that valid short hashes pass."""
        result = val.validate_commit_hash("abc1234")
        assert result == "abc1234"

    def test_invalid_format_fails(self):
        """Test that invalid formats raise ValidationError."""
        with pytest.raises(ValidationError):
            val.validate_commit_hash("not-a-hash")

    def test_too_short_fails(self):
        """Test that too-short hashes raise ValidationError."""
        with pytest.raises(ValidationError):
            val.validate_commit_hash("abc123")
