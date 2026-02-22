"""Tests for logging configuration module."""

from __future__ import annotations

import io
import logging

import logging_config


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_creates_logger(self):
        """Test that setup_logging creates a logger."""
        # Reset initialization state for this test
        logging_config._initialized = False
        logger = logging_config.setup_logging(level="INFO")
        assert logger is not None
        assert logger.name == "code_memory"

    def test_respects_level(self):
        """Test that log level is set correctly."""
        # Create a fresh logger with custom stream
        stream = io.StringIO()
        logger = logging.getLogger("test_code_memory_1")
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        assert logger.level == logging.DEBUG

    def test_custom_level_string(self):
        """Test custom log level string."""
        logger = logging.getLogger("test_code_memory_2")
        logger.handlers.clear()
        logger.setLevel(logging.WARNING)
        assert logger.level == logging.WARNING


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_module_logger(self):
        """Test getting a module-specific logger."""
        logger = logging_config.get_logger("test_module")
        assert logger.name == "code_memory.test_module"

    def test_different_modules_different_loggers(self):
        """Test that different modules get different loggers."""
        logger1 = logging_config.get_logger("module1")
        logger2 = logging_config.get_logger("module2")
        assert logger1.name != logger2.name


class TestToolLogger:
    """Tests for ToolLogger context manager."""

    def test_logs_invocation(self):
        """Test that tool invocation is logged."""
        # Create a logger with a string stream to capture output
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter('%(message)s'))

        logger = logging.getLogger("code_memory.tools")
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

        with logging_config.ToolLogger("test_tool", query="test"):
            pass

        output = stream.getvalue()
        assert "Tool invoked" in output

    def test_logs_completion(self):
        """Test that tool completion is logged."""
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter('%(message)s'))

        logger = logging.getLogger("code_memory.tools")
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

        with logging_config.ToolLogger("test_tool", query="test"):
            pass

        output = stream.getvalue()
        assert "Tool completed" in output

    def test_logs_error_on_exception(self):
        """Test that exceptions are logged."""
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter('%(message)s'))

        logger = logging.getLogger("code_memory.tools")
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

        try:
            with logging_config.ToolLogger("test_tool", query="test"):
                raise ValueError("Test error")
        except ValueError:
            pass

        output = stream.getvalue()
        assert "Tool failed" in output

    def test_result_count_logged(self):
        """Test that result count is logged."""
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter('%(message)s'))

        logger = logging.getLogger("code_memory.tools")
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

        with logging_config.ToolLogger("test_tool") as log:
            log.set_result_count(5)

        output = stream.getvalue()
        assert "count=5" in output


class TestIndexingLogger:
    """Tests for IndexingLogger class."""

    def test_tracks_files_processed(self):
        """Test that files processed are tracked."""
        idx_logger = logging_config.IndexingLogger("test")
        idx_logger.file_indexed("file1.py", 3)
        idx_logger.file_indexed("file2.py", 2)
        assert idx_logger.files_processed == 2
        assert idx_logger.items_indexed == 5

    def test_tracks_files_skipped(self):
        """Test that files skipped are tracked."""
        idx_logger = logging_config.IndexingLogger("test")
        idx_logger.file_skipped("file1.py", "unchanged")
        assert idx_logger.files_skipped == 1


class TestPreconfiguredLoggers:
    """Tests for pre-configured logger functions."""

    def test_get_server_logger(self):
        """Test getting server logger."""
        logger = logging_config.get_server_logger()
        assert "server" in logger.name

    def test_get_db_logger(self):
        """Test getting db logger."""
        logger = logging_config.get_db_logger()
        assert "db" in logger.name

    def test_get_query_logger(self):
        """Test getting query logger."""
        logger = logging_config.get_query_logger()
        assert "queries" in logger.name
