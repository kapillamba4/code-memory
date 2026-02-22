"""
Documentation parser for code-memory.

Parses markdown documentation files, chunks them into semantic units,
and indexes them for hybrid retrieval (BM25 + vector search).
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from markdown_it import MarkdownIt

import db as db_mod

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SKIP_DIRS = {
    ".venv",
    "venv",
    "__pycache__",
    ".git",
    "node_modules",
    "build",
    "dist",
    "target",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}

DOC_EXTENSIONS = {".md", ".markdown"}
README_PATTERN = re.compile(r"^readme(\.md|\.markdown|\.txt)?$", re.IGNORECASE)

DEFAULT_MAX_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 100
DEFAULT_MIN_CHUNK_SIZE = 50


# ---------------------------------------------------------------------------
# Markdown parsing
# ---------------------------------------------------------------------------


def parse_markdown_sections(filepath: str) -> list[dict]:
    """Parse markdown file into sections based on heading hierarchy.

    Args:
        filepath: Path to the markdown file.

    Returns:
        List of section dicts with keys:
        - section_title: The heading text (or None for preamble)
        - content: Full text including heading
        - line_start: Starting line number (1-indexed)
        - line_end: Ending line number (1-indexed)
        - level: Heading level (1-6, or 0 for preamble)
    """
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    md = MarkdownIt()
    tokens = md.parse("".join(lines))

    sections = []
    current_section = {"section_title": None, "content": [], "line_start": 1, "level": 0}

    # Build a map from token to line number
    line_map = _build_line_map(tokens, lines)

    for i, token in enumerate(tokens):
        if token.type == "heading_open":
            # Save previous section if it has content
            if current_section["content"]:
                sections.append(_finalize_section(current_section, line_map, i - 1))

            level = int(token.tag[1])  # h1 -> 1, h2 -> 2, etc.
            current_section = {
                "section_title": None,
                "content": [],
                "line_start": line_map.get(i, 1),
                "level": level,
            }

        elif token.type == "heading_close":
            # Heading content collected, continue to next section
            pass

        elif token.type == "inline":
            # This is content (heading text or paragraph)
            current_section["content"].append(token.content)
            if current_section["section_title"] is None and current_section["level"] > 0:
                current_section["section_title"] = token.content

    # Finalize last section
    if current_section["content"]:
        sections.append(_finalize_section(current_section, line_map, len(tokens) - 1))

    # Calculate line numbers from actual line content
    return _calculate_line_numbers(sections, lines)


def _build_line_map(tokens, lines) -> dict[int, int]:
    """Build a map from token index to line number."""
    # Simplified: use map from token content to line
    return {}  # We'll calculate lines differently


def _finalize_section(section: dict, line_map: dict, end_token_idx: int) -> dict:
    """Finalize a section dict."""
    return {
        "section_title": section["section_title"],
        "content": "\n".join(section["content"]),
        "line_start": section["line_start"],
        "line_end": section["line_start"],  # Will be updated
        "level": section["level"],
    }


def _calculate_line_numbers(sections: list[dict], lines: list[str]) -> list[dict]:
    """Calculate accurate line numbers by matching content to source lines."""
    if not sections:
        return sections

    result = []
    line_idx = 0

    for section in sections:
        content_lines = section["content"].split("\n")
        if not content_lines or not content_lines[0]:
            continue

        # Find the starting line by looking for the section title or content
        start_line = line_idx + 1
        first_content = content_lines[0].strip()

        # Search for the content in remaining lines
        for i in range(line_idx, len(lines)):
            if first_content in lines[i]:
                start_line = i + 1
                line_idx = i
                break

        # Find the end line (next heading or end of file)
        end_line = len(lines)
        level = section["level"]

        if level > 0:
            # Look for next heading of same or higher level
            for i in range(line_idx + 1, len(lines)):
                if re.match(r"^#{1," + str(level) + r"}\s", lines[i]):
                    end_line = i
                    break
        else:
            # Preamble ends at first heading
            for i in range(line_idx, len(lines)):
                if re.match(r"^#{1,6}\s", lines[i]):
                    end_line = i
                    break

        result.append({
            "section_title": section["section_title"],
            "content": section["content"],
            "line_start": start_line,
            "line_end": end_line,
            "level": level,
        })

        line_idx = end_line

    return result


def chunk_content(content: str, max_size: int = DEFAULT_MAX_CHUNK_SIZE,
                  overlap: int = DEFAULT_OVERLAP) -> list[str]:
    """Split content into overlapping chunks if it exceeds max_size.

    Attempts to split on sentence boundaries when possible.

    Args:
        content: The text content to chunk.
        max_size: Maximum chunk size in characters.
        overlap: Number of characters to overlap between chunks.

    Returns:
        List of chunk strings.
    """
    if len(content) <= max_size:
        return [content]

    chunks = []
    start = 0

    while start < len(content):
        end = start + max_size

        if end < len(content):
            # Try to find a sentence boundary
            boundary = content.rfind(". ", start, end)
            if boundary > start + max_size // 2:
                end = boundary + 1  # Include the period
            else:
                # Try newline
                boundary = content.rfind("\n", start, end)
                if boundary > start + max_size // 2:
                    end = boundary

        chunk = content[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap if end < len(content) else len(content)

    return chunks


# ---------------------------------------------------------------------------
# File indexing
# ---------------------------------------------------------------------------


def _get_doc_type(filepath: str) -> str:
    """Determine documentation type from filepath."""
    filename = os.path.basename(filepath).lower()
    if README_PATTERN.match(filename):
        return "readme"
    return "markdown"


def index_doc_file(
    filepath: str,
    db,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    min_chunk_size: int = DEFAULT_MIN_CHUNK_SIZE,
) -> dict:
    """Index a documentation file.

    Args:
        filepath: Path to the documentation file.
        db: Database connection.
        max_chunk_size: Maximum chunk size in characters.
        overlap: Overlap between chunks.
        min_chunk_size: Minimum chunk size (smaller chunks are discarded).

    Returns:
        Summary dict with file, doc_type, chunks_indexed, etc.
    """
    abs_path = os.path.abspath(filepath)

    if not os.path.isfile(abs_path):
        return {"file": filepath, "error": "File not found", "chunks_indexed": 0}

    # Check if file has changed
    stat = os.stat(abs_path)
    last_modified = stat.st_mtime
    fhash = db_mod.file_hash(abs_path)

    existing = db.execute(
        "SELECT id, file_hash FROM doc_files WHERE path = ?", (abs_path,)
    ).fetchone()

    if existing and existing[1] == fhash:
        return {
            "file": filepath,
            "doc_type": _get_doc_type(abs_path),
            "chunks_indexed": 0,
            "skipped": True,
            "reason": "Unchanged",
        }

    # Delete old data if re-indexing
    if existing:
        db_mod.delete_doc_file_data(db, existing[0])

    # Upsert file record
    doc_type = _get_doc_type(abs_path)
    doc_file_id = db_mod.upsert_doc_file(db, abs_path, last_modified, fhash, doc_type)

    # Parse and chunk
    sections = parse_markdown_sections(abs_path)

    chunks_indexed = 0
    chunk_index = 0

    for section in sections:
        content = section["content"]
        if len(content) < min_chunk_size:
            continue

        # Split large sections into smaller chunks
        sub_chunks = chunk_content(content, max_chunk_size, overlap)

        for sub_content in sub_chunks:
            if len(sub_content) < min_chunk_size:
                continue

            chunk_id = db_mod.upsert_doc_chunk(
                db,
                doc_file_id,
                chunk_index,
                section["section_title"],
                sub_content,
                section["line_start"],
                section["line_end"],
            )

            # Generate and store embedding
            embedding = db_mod.embed_text(f"{section['section_title'] or ''}: {sub_content}")
            db_mod.upsert_doc_embedding(db, chunk_id, embedding)

            chunk_index += 1
            chunks_indexed += 1

    return {
        "file": filepath,
        "doc_type": doc_type,
        "chunks_indexed": chunks_indexed,
        "skipped": False,
        "reason": None,
    }


def index_doc_directory(dirpath: str, db) -> list[dict]:
    """Recursively index all documentation in a directory.

    Args:
        dirpath: Root directory to search.
        db: Database connection.

    Returns:
        List of result dicts from index_doc_file.
    """
    abs_dir = os.path.abspath(dirpath)
    results = []

    for root, dirs, files in os.walk(abs_dir):
        # Skip unwanted directories
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]

        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext in DOC_EXTENSIONS:
                filepath = os.path.join(root, filename)
                result = index_doc_file(filepath, db)
                results.append(result)

    return results


def extract_docstrings_from_code(db) -> list[dict]:
    """Extract docstrings from already-indexed code symbols.

    This function queries the existing symbols table and extracts
    docstrings from the source_text field.

    Args:
        db: Database connection.

    Returns:
        List of result dicts for indexed docstrings.
    """
    results = []

    # Get all symbols with their source text
    rows = db.execute(
        """
        SELECT s.id, s.name, s.kind, f.path, s.line_start, s.line_end, s.source_text
        FROM symbols s
        JOIN files f ON f.id = s.file_id
        WHERE s.kind IN ('function', 'class', 'method')
        """
    ).fetchall()

    for row in rows:
        symbol_id, name, kind, file_path, line_start, line_end, source_text = row

        # Extract docstring from source text
        docstring = _extract_docstring_from_source(source_text)
        if not docstring or len(docstring) < 20:
            continue

        # Check if we already have this docstring indexed
        existing = db.execute(
            """
            SELECT dc.id FROM doc_chunks dc
            JOIN doc_files df ON df.id = dc.doc_file_id
            WHERE df.path = ? AND dc.line_start = ? AND dc.section_title = ?
            """,
            (file_path, line_start, name),
        ).fetchone()

        if existing:
            continue

        # Create a doc_file entry for the code file if needed
        doc_file = db.execute(
            "SELECT id FROM doc_files WHERE path = ?", (file_path,)
        ).fetchone()

        if not doc_file:
            # Get file stats
            stat = os.stat(file_path) if os.path.exists(file_path) else None
            doc_file_id = db_mod.upsert_doc_file(
                db,
                file_path,
                stat.st_mtime if stat else 0,
                db_mod.file_hash(file_path) if stat else "",
                "docstring",
            )
        else:
            doc_file_id = doc_file[0]

        # Get next chunk index
        max_idx = db.execute(
            "SELECT COALESCE(MAX(chunk_index), -1) FROM doc_chunks WHERE doc_file_id = ?",
            (doc_file_id,),
        ).fetchone()[0]

        chunk_id = db_mod.upsert_doc_chunk(
            db,
            doc_file_id,
            max_idx + 1,
            name,  # Use symbol name as section title
            docstring,
            line_start,
            line_end,
        )

        # Generate and store embedding
        embedding = db_mod.embed_text(f"{kind} {name}: {docstring}")
        db_mod.upsert_doc_embedding(db, chunk_id, embedding)

        results.append({
            "symbol": name,
            "kind": kind,
            "file": file_path,
            "docstring_length": len(docstring),
        })

    return results


def _extract_docstring_from_source(source_text: str) -> str | None:
    """Extract docstring from Python source code text.

    Handles both single-line and multi-line docstrings.
    """
    lines = source_text.split("\n")
    if not lines:
        return None

    # Skip the def/class line
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("def ") or line.strip().startswith("class "):
            start_idx = i + 1
            break

    # Find docstring
    for i in range(start_idx, len(lines)):
        stripped = lines[i].strip()

        if not stripped or stripped.startswith("#"):
            continue

        # Check for triple-quoted docstring
        if stripped.startswith('"""') or stripped.startswith("'''"):
            quote = '"""' if stripped.startswith('"""') else "'''"

            # Single-line docstring
            if stripped.count(quote) >= 2:
                return stripped[len(quote):-len(quote)].strip()

            # Multi-line docstring
            docstring_lines = [stripped[len(quote):]]
            for j in range(i + 1, len(lines)):
                docstring_lines.append(lines[j])
                if quote in lines[j]:
                    # Remove closing quotes
                    docstring_lines[-1] = lines[j][:lines[j].index(quote)]
                    break

            return "\n".join(docstring_lines).strip()

        # Not a docstring
        break

    return None
