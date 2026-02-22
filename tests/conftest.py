"""
Shared test fixtures for code-memory tests.
"""

from __future__ import annotations

import os
import sys
import sqlite3
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


@pytest.fixture
def temp_db():
    """Provide a temporary in-memory database for tests."""
    # Use a temporary file for sqlite-vec compatibility
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    db = sqlite3.connect(db_path)

    # Load sqlite-vec
    try:
        import sqlite_vec
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
    except ImportError:
        pass  # sqlite-vec not available, skip vector tests

    yield db

    db.close()
    os.unlink(db_path)


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for file tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_python_file(temp_dir):
    """Create a sample Python file for parsing tests."""
    code = '''
"""Module docstring for testing."""

import os
from typing import Optional

class SampleClass:
    """A sample class for testing."""

    def __init__(self, name: str):
        """Initialize the sample class."""
        self.name = name

    def get_name(self) -> str:
        """Return the name."""
        return self.name

    def process_data(self, data: Optional[dict] = None) -> dict:
        """Process some data."""
        if data is None:
            data = {}
        return {"name": self.name, **data}


def standalone_function(x: int, y: int) -> int:
    """A standalone function that adds two numbers."""
    return x + y


def another_function(text: str) -> str:
    """Another function for testing."""
    return text.upper()
'''
    filepath = temp_dir / "sample.py"
    filepath.write_text(code)
    return filepath


@pytest.fixture
def sample_markdown_file(temp_dir):
    """Create a sample markdown file for documentation tests."""
    content = """# Sample Documentation

This is a sample documentation file for testing.

## Installation

To install, run:

```bash
pip install code-memory
```

## Usage

Here's how to use the tool:

1. Index your codebase
2. Search for symbols

## Architecture

The system uses a Progressive Disclosure architecture.

### Components

- search_code: Find definitions
- search_docs: Find documentation
- search_history: Search git history
"""
    filepath = temp_dir / "README.md"
    filepath.write_text(content)
    return filepath


@pytest.fixture
def temp_git_repo(temp_dir):
    """Provide a temporary git repository for tests."""
    import subprocess

    # Initialize git repo
    subprocess.run(["git", "init"], cwd=temp_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=temp_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=temp_dir, check=True, capture_output=True)

    # Create and commit a file
    test_file = temp_dir / "test.py"
    test_file.write_text("# Test file\nprint('hello')\n")
    subprocess.run(["git", "add", "."], cwd=temp_dir, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=temp_dir, check=True, capture_output=True)

    yield temp_dir


@pytest.fixture
def sample_symbols_db(temp_db):
    """Provide a database with sample symbols for search tests."""
    # Create minimal schema
    temp_db.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY,
            path TEXT UNIQUE NOT NULL,
            last_modified REAL NOT NULL,
            file_hash TEXT NOT NULL
        )
    """)

    temp_db.execute("""
        CREATE TABLE IF NOT EXISTS symbols (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            kind TEXT NOT NULL,
            file_id INTEGER NOT NULL,
            line_start INTEGER NOT NULL,
            line_end INTEGER NOT NULL,
            parent_symbol_id INTEGER,
            source_text TEXT NOT NULL
        )
    """)

    # Insert sample data
    temp_db.execute("INSERT INTO files (path, last_modified, file_hash) VALUES (?, ?, ?)",
                    ("/test/sample.py", 0.0, "abc123"))
    file_id = temp_db.lastrowid

    symbols = [
        ("SampleClass", "class", file_id, 5, 20, None, "class SampleClass: ..."),
        ("__init__", "method", file_id, 8, 10, 1, "def __init__(self, name): ..."),
        ("get_name", "method", file_id, 12, 14, 1, "def get_name(self): ..."),
        ("standalone_function", "function", file_id, 22, 24, None, "def standalone_function(x, y): ..."),
    ]

    for name, kind, fid, line_start, line_end, parent, source in symbols:
        temp_db.execute(
            "INSERT INTO symbols (name, kind, file_id, line_start, line_end, parent_symbol_id, source_text) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (name, kind, fid, line_start, line_end, parent, source)
        )

    temp_db.commit()
    return temp_db
