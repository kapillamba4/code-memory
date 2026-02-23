"""
Structured logging configuration for code-memory.

Provides configurable logging with environment variable control.
Log level can be set via CODE_MEMORY_LOG_LEVEL environment variable.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from typing import TextIO

# Default log level from environment or INFO
LOG_LEVEL = os.environ.get("CODE_MEMORY_LOG_LEVEL", "INFO").upper()

# Log format with timestamp, module, level, and message
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Track if logging has been initialized
_initialized = False


@contextmanager
def log_timing(operation_name: str, logger: logging.Logger):
    """Context manager to log operation timing.

    Args:
        operation_name: Name of the operation being timed.
        logger: Logger instance to use for logging.

    Example:
        with log_timing("Indexing myfile.py", logger):
            # ... indexing code ...
    """
    start = time.perf_counter()
    logger.debug(f"{operation_name} started")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info(f"{operation_name} completed in {elapsed:.2f}s")


def setup_logging(level: str = LOG_LEVEL, stream: TextIO = sys.stderr) -> logging.Logger:
    """Configure structured logging for code-memory.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        stream: Output stream for logs (default: stderr)

    Returns:
        Configured root logger for code_memory
    """
    global _initialized

    logger = logging.getLogger("code_memory")

    # Avoid adding duplicate handlers
    if _initialized and logger.handlers:
        return logger

    # Parse log level
    level_value = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level_value)

    # Create handler with formatter
    handler = logging.StreamHandler(stream)
    handler.setLevel(level_value)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    handler.setFormatter(formatter)

    # Clear existing handlers and add new one
    logger.handlers.clear()
    logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False

    _initialized = True
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.

    Args:
        name: Module name (e.g., "server", "db", "parser")

    Returns:
        Logger instance for the module
    """
    # Ensure logging is initialized
    if not _initialized:
        setup_logging()

    return logging.getLogger(f"code_memory.{name}")


class ToolLogger:
    """Context manager for logging tool invocations with timing.

    Usage:
        with ToolLogger("search_code", query="test", search_type="definition") as log:
            result = perform_search()
            log.set_result_count(len(result))
    """

    def __init__(self, tool_name: str, **params):
        self.tool_name = tool_name
        self.params = params
        self.logger = get_logger("tools")
        self.start_time: datetime | None = None
        self.result_count: int | None = None
        self.error: str | None = None

    def __enter__(self) -> ToolLogger:
        self.start_time = datetime.now()
        # Sanitize params for logging (don't log sensitive data)
        safe_params = {k: v for k, v in self.params.items() if v is not None}
        self.logger.info(f"Tool invoked: {self.tool_name} params={safe_params}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (datetime.now() - self.start_time).total_seconds() * 1000 if self.start_time else 0

        if exc_type is not None:
            self.error = str(exc_val)
            self.logger.error(
                f"Tool failed: {self.tool_name} error={self.error} duration={duration_ms:.1f}ms"
            )
        else:
            count_str = f" count={self.result_count}" if self.result_count is not None else ""
            self.logger.info(
                f"Tool completed: {self.tool_name}{count_str} duration={duration_ms:.1f}ms"
            )

        return False  # Don't suppress exceptions

    def set_result_count(self, count: int) -> None:
        """Set the number of results returned by the tool."""
        self.result_count = count


class IndexingLogger:
    """Logger for indexing operations with progress tracking."""

    def __init__(self, indexer_type: str):
        self.indexer_type = indexer_type
        self.logger = get_logger("indexing")
        self.files_processed = 0
        self.items_indexed = 0
        self.files_unchanged = 0
        self.start_time: datetime | None = None

    def start(self, directory: str) -> None:
        """Log the start of an indexing operation."""
        self.start_time = datetime.now()
        self.logger.info(f"Starting {self.indexer_type} indexing: directory={directory}")

    def file_indexed(self, filepath: str, items: int = 1) -> None:
        """Log successful file indexing."""
        self.files_processed += 1
        self.items_indexed += items
        self.logger.debug(f"Indexed {self.indexer_type}: {filepath} ({items} items)")

    def file_skipped(self, filepath: str, reason: str) -> None:
        """Log skipped file."""
        self.files_unchanged += 1
        self.logger.debug(f"Skipped {self.indexer_type}: {filepath} ({reason})")

    def complete(self) -> None:
        """Log completion of indexing."""
        duration_ms = (datetime.now() - self.start_time).total_seconds() * 1000 if self.start_time else 0
        self.logger.info(
            f"Completed {self.indexer_type} indexing: "
            f"files={self.files_processed} items={self.items_indexed} "
            f"unchanged={self.files_unchanged} duration={duration_ms:.1f}ms"
        )

    def error(self, filepath: str, error_msg: str) -> None:
        """Log indexing error."""
        self.logger.warning(f"Error indexing {filepath}: {error_msg}")


# Pre-configured loggers for common modules
def get_server_logger() -> logging.Logger:
    """Get logger for server module."""
    return get_logger("server")


def get_db_logger() -> logging.Logger:
    """Get logger for database module."""
    return get_logger("db")


def get_parser_logger() -> logging.Logger:
    """Get logger for parser module."""
    return get_logger("parser")


def get_query_logger() -> logging.Logger:
    """Get logger for query module."""
    return get_logger("queries")


def get_git_logger() -> logging.Logger:
    """Get logger for git search module."""
    return get_logger("git")
