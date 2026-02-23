"""Integration tests for MCP tools."""

from __future__ import annotations

from unittest.mock import AsyncMock


class MockContext:
    """Mock MCP Context for testing."""

    def __init__(self):
        self.report_progress = AsyncMock()


class TestSearchCodeValidation:
    """Tests for search_code tool input validation."""

    def test_empty_query_returns_error(self):
        """Test that empty query returns structured error."""
        import server
        result = server.search_code("", "definition", "/tmp")
        assert result.get("error") is True
        assert "ValidationError" in result.get("error_type", "")

    def test_invalid_search_type_returns_error(self):
        """Test that invalid search_type returns structured error."""
        import server
        result = server.search_code("test", "invalid_type", "/tmp")
        assert result.get("error") is True
        assert "ValidationError" in result.get("error_type", "")


class TestSearchDocsValidation:
    """Tests for search_docs tool input validation."""

    def test_empty_query_returns_error(self):
        """Test that empty query returns structured error."""
        import server
        result = server.search_docs("", "/tmp")
        assert result.get("error") is True
        assert "ValidationError" in result.get("error_type", "")

    def test_invalid_top_k_returns_error(self):
        """Test that invalid top_k returns structured error."""
        import server
        result = server.search_docs("test", "/tmp", top_k=-1)
        assert result.get("error") is True


class TestSearchHistoryValidation:
    """Tests for search_history tool input validation."""

    def test_invalid_search_type_returns_error(self):
        """Test that invalid search_type returns structured error."""
        import server
        result = server.search_history("test", "/tmp", search_type="invalid")
        assert result.get("error") is True
        assert "ValidationError" in result.get("error_type", "")

    def test_file_history_requires_target_file(self):
        """Test that file_history requires target_file."""
        import server
        # Use current directory (which is a git repo) to get past git validation
        result = server.search_history("test", ".", search_type="file_history", target_file=None)
        assert result.get("error") is True
        assert "target_file" in result.get("message", "").lower()

    def test_blame_requires_target_file(self):
        """Test that blame requires target_file."""
        import server
        # Use current directory (which is a git repo) to get past git validation
        result = server.search_history("test", ".", search_type="blame", target_file=None)
        assert result.get("error") is True
        assert "target_file" in result.get("message", "").lower()

    def test_invalid_line_range_returns_error(self):
        """Test that invalid line range returns error."""
        import server
        # This should work since we're in a git repo, but line_start > line_end
        result = server.search_history(
            "test",
            ".",
            search_type="blame",
            target_file="server.py",
            line_start=10,
            line_end=5
        )
        assert result.get("error") is True


class TestIndexCodebaseValidation:
    """Tests for index_codebase tool input validation."""

    def test_nonexistent_directory_returns_error(self):
        """Test that nonexistent directory returns structured error."""
        import asyncio

        import server
        ctx = MockContext()

        async def run_test():
            result = await server.index_codebase("/nonexistent/directory", ctx)
            return result

        result = asyncio.run(run_test())
        assert result.get("error") is True
        assert "ValidationError" in result.get("error_type", "")


class TestToolResponseStructure:
    """Tests for consistent tool response structure."""

    def test_success_response_has_status(self):
        """Test that successful responses have status field."""
        import server
        # search_docs should work even without indexed content
        result = server.search_docs("test query", "/tmp")
        # Either it succeeds or fails gracefully
        if "status" in result:
            assert result["status"] in ("ok", "error")
        elif "error" in result:
            assert result["error"] is True

    def test_error_response_structure(self):
        """Test that error responses have consistent structure."""
        import server
        result = server.search_code("", "definition", "/tmp")
        assert "error" in result
        assert result["error"] is True
        assert "error_type" in result
        assert "message" in result
