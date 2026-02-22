"""
Database layer for code-memory.

Manages a local SQLite database with three storage layers:
  1. Relational tables (files, symbols, references)
  2. FTS5 full-text index (symbols_fts) for BM25 keyword search
  3. sqlite-vec virtual table (symbol_embeddings) for dense vector search

All writes use upsert semantics so re-indexing is idempotent.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import sqlite_vec

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Embedding model (lazy-loaded singleton)
# ---------------------------------------------------------------------------

_model = None
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2


def get_embedding_model():
    """Lazy-load and cache the sentence-transformers model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_text(text: str) -> list[float]:
    """Generate a 384-dim dense vector embedding for *text*."""
    model = get_embedding_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()


# ---------------------------------------------------------------------------
# Database initialisation
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
-- 1. Tracked source files
CREATE TABLE IF NOT EXISTS files (
    id            INTEGER PRIMARY KEY,
    path          TEXT    UNIQUE NOT NULL,
    last_modified REAL   NOT NULL,
    file_hash     TEXT   NOT NULL
);

-- 2. Parsed AST symbols
CREATE TABLE IF NOT EXISTS symbols (
    id               INTEGER PRIMARY KEY,
    name             TEXT    NOT NULL,
    kind             TEXT    NOT NULL,
    file_id          INTEGER NOT NULL REFERENCES files(id),
    line_start       INTEGER NOT NULL,
    line_end         INTEGER NOT NULL,
    parent_symbol_id INTEGER,
    source_text      TEXT    NOT NULL,
    UNIQUE(file_id, name, kind, line_start)
);

-- 3. FTS5 content-sync'd to symbols (indexes name + source_text)
CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
    name,
    source_text,
    content=symbols,
    content_rowid=id
);

-- Triggers to keep FTS5 in sync with symbols table
CREATE TRIGGER IF NOT EXISTS symbols_ai AFTER INSERT ON symbols BEGIN
    INSERT INTO symbols_fts(rowid, name, source_text)
    VALUES (new.id, new.name, new.source_text);
END;

CREATE TRIGGER IF NOT EXISTS symbols_ad AFTER DELETE ON symbols BEGIN
    INSERT INTO symbols_fts(symbols_fts, rowid, name, source_text)
    VALUES ('delete', old.id, old.name, old.source_text);
END;

CREATE TRIGGER IF NOT EXISTS symbols_au AFTER UPDATE ON symbols BEGIN
    INSERT INTO symbols_fts(symbols_fts, rowid, name, source_text)
    VALUES ('delete', old.id, old.name, old.source_text);
    INSERT INTO symbols_fts(rowid, name, source_text)
    VALUES (new.id, new.name, new.source_text);
END;

-- 5. Cross-reference tracking
CREATE TABLE IF NOT EXISTS references_ (
    id          INTEGER PRIMARY KEY,
    symbol_name TEXT    NOT NULL,
    file_id     INTEGER NOT NULL REFERENCES files(id),
    line_number INTEGER NOT NULL,
    UNIQUE(symbol_name, file_id, line_number)
);

-- ---------------------------------------------------------------------------
-- Documentation tables (Milestone 4)
-- ---------------------------------------------------------------------------

-- 6. Tracked documentation files
CREATE TABLE IF NOT EXISTS doc_files (
    id            INTEGER PRIMARY KEY,
    path          TEXT    UNIQUE NOT NULL,
    last_modified REAL   NOT NULL,
    file_hash     TEXT   NOT NULL,
    doc_type      TEXT   NOT NULL  -- 'markdown', 'readme', 'docstring'
);

-- 7. Chunked documentation content
CREATE TABLE IF NOT EXISTS doc_chunks (
    id            INTEGER PRIMARY KEY,
    doc_file_id   INTEGER NOT NULL REFERENCES doc_files(id),
    chunk_index   INTEGER NOT NULL,
    section_title TEXT,
    content       TEXT    NOT NULL,
    line_start    INTEGER NOT NULL,
    line_end      INTEGER NOT NULL,
    UNIQUE(doc_file_id, chunk_index)
);

-- 8. FTS5 for documentation chunks (BM25 keyword search)
CREATE VIRTUAL TABLE IF NOT EXISTS doc_chunks_fts USING fts5(
    content,
    section_title,
    content=doc_chunks,
    content_rowid=id
);

-- Triggers to keep doc FTS5 in sync
CREATE TRIGGER IF NOT EXISTS doc_chunks_ai AFTER INSERT ON doc_chunks BEGIN
    INSERT INTO doc_chunks_fts(rowid, content, section_title)
    VALUES (new.id, new.content, new.section_title);
END;

CREATE TRIGGER IF NOT EXISTS doc_chunks_ad AFTER DELETE ON doc_chunks BEGIN
    INSERT INTO doc_chunks_fts(doc_chunks_fts, rowid, content, section_title)
    VALUES ('delete', old.id, old.content, old.section_title);
END;

CREATE TRIGGER IF NOT EXISTS doc_chunks_au AFTER UPDATE ON doc_chunks BEGIN
    INSERT INTO doc_chunks_fts(doc_chunks_fts, rowid, content, section_title)
    VALUES ('delete', old.id, old.content, old.section_title);
    INSERT INTO doc_chunks_fts(rowid, content, section_title)
    VALUES (new.id, new.content, new.section_title);
END;
"""


def get_db(db_path: str = "code_memory.db") -> sqlite3.Connection:
    """Open (or create) the database, load sqlite-vec, and ensure schema.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        A ready-to-use ``sqlite3.Connection`` with WAL mode and foreign keys.
    """
    db = sqlite3.connect(db_path)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")

    db.executescript(_SCHEMA_SQL)

    # sqlite-vec virtual table for code embeddings (must be created outside executescript)
    db.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS symbol_embeddings
        USING vec0(
            symbol_id INTEGER PRIMARY KEY,
            embedding float[{EMBEDDING_DIM}]
        )
        """
    )

    # sqlite-vec virtual table for documentation embeddings
    db.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS doc_embeddings
        USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding float[{EMBEDDING_DIM}]
        )
        """
    )
    db.commit()
    return db


# ---------------------------------------------------------------------------
# Upsert helpers
# ---------------------------------------------------------------------------


def file_hash(filepath: str) -> str:
    """Compute SHA-256 hex digest of a file's contents."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def upsert_file(db: sqlite3.Connection, path: str, last_modified: float, fhash: str) -> int:
    """Insert or update a file record. Returns the file_id."""
    cur = db.execute(
        """
        INSERT INTO files (path, last_modified, file_hash)
        VALUES (?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            last_modified = excluded.last_modified,
            file_hash     = excluded.file_hash
        """,
        (path, last_modified, fhash),
    )
    db.commit()
    # Fetch the id (needed because last_insert_rowid isn't reliable on update)
    row = db.execute("SELECT id FROM files WHERE path = ?", (path,)).fetchone()
    return row[0]


def delete_file_data(db: sqlite3.Connection, file_id: int) -> None:
    """Remove all symbols, embeddings, and references for a file.

    This is called before re-indexing to guarantee idempotency.
    """
    # Collect symbol ids for embedding cleanup
    sym_ids = [
        r[0] for r in db.execute("SELECT id FROM symbols WHERE file_id = ?", (file_id,)).fetchall()
    ]
    if sym_ids:
        placeholders = ",".join("?" * len(sym_ids))
        db.execute(f"DELETE FROM symbol_embeddings WHERE symbol_id IN ({placeholders})", sym_ids)

    db.execute("DELETE FROM symbols WHERE file_id = ?", (file_id,))
    db.execute("DELETE FROM references_ WHERE file_id = ?", (file_id,))
    db.commit()


def upsert_symbol(
    db: sqlite3.Connection,
    name: str,
    kind: str,
    file_id: int,
    line_start: int,
    line_end: int,
    parent_symbol_id: int | None,
    source_text: str,
) -> int:
    """Insert or update a symbol record. Returns the symbol_id."""
    db.execute(
        """
        INSERT INTO symbols (name, kind, file_id, line_start, line_end,
                             parent_symbol_id, source_text)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(file_id, name, kind, line_start) DO UPDATE SET
            line_end         = excluded.line_end,
            parent_symbol_id = excluded.parent_symbol_id,
            source_text      = excluded.source_text
        """,
        (name, kind, file_id, line_start, line_end, parent_symbol_id, source_text),
    )
    db.commit()
    row = db.execute(
        "SELECT id FROM symbols WHERE file_id = ? AND name = ? AND kind = ? AND line_start = ?",
        (file_id, name, kind, line_start),
    ).fetchone()
    return row[0]


def upsert_reference(
    db: sqlite3.Connection, symbol_name: str, file_id: int, line_number: int
) -> None:
    """Insert or update a cross-reference record."""
    db.execute(
        """
        INSERT INTO references_ (symbol_name, file_id, line_number)
        VALUES (?, ?, ?)
        ON CONFLICT(symbol_name, file_id, line_number) DO NOTHING
        """,
        (symbol_name, file_id, line_number),
    )
    db.commit()


def upsert_embedding(db: sqlite3.Connection, symbol_id: int, embedding: list[float]) -> None:
    """Insert or replace a symbol's dense vector embedding."""
    import struct

    blob = struct.pack(f"{len(embedding)}f", *embedding)
    # sqlite-vec doesn't support ON CONFLICT, so delete-then-insert
    db.execute("DELETE FROM symbol_embeddings WHERE symbol_id = ?", (symbol_id,))
    db.execute(
        "INSERT INTO symbol_embeddings (symbol_id, embedding) VALUES (?, ?)",
        (symbol_id, blob),
    )
    db.commit()


# ---------------------------------------------------------------------------
# Documentation upsert helpers (Milestone 4)
# ---------------------------------------------------------------------------


def upsert_doc_file(
    db: sqlite3.Connection, path: str, last_modified: float, fhash: str, doc_type: str
) -> int:
    """Insert or update a documentation file record. Returns doc_file_id."""
    db.execute(
        """
        INSERT INTO doc_files (path, last_modified, file_hash, doc_type)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            last_modified = excluded.last_modified,
            file_hash     = excluded.file_hash,
            doc_type      = excluded.doc_type
        """,
        (path, last_modified, fhash, doc_type),
    )
    db.commit()
    row = db.execute("SELECT id FROM doc_files WHERE path = ?", (path,)).fetchone()
    return row[0]


def delete_doc_file_data(db: sqlite3.Connection, doc_file_id: int) -> None:
    """Remove all chunks and embeddings for a documentation file.

    This is called before re-indexing to guarantee idempotency.
    """
    # Collect chunk ids for embedding cleanup
    chunk_ids = [
        r[0]
        for r in db.execute(
            "SELECT id FROM doc_chunks WHERE doc_file_id = ?", (doc_file_id,)
        ).fetchall()
    ]
    if chunk_ids:
        placeholders = ",".join("?" * len(chunk_ids))
        db.execute(f"DELETE FROM doc_embeddings WHERE chunk_id IN ({placeholders})", chunk_ids)

    db.execute("DELETE FROM doc_chunks WHERE doc_file_id = ?", (doc_file_id,))
    db.commit()


def upsert_doc_chunk(
    db: sqlite3.Connection,
    doc_file_id: int,
    chunk_index: int,
    section_title: str | None,
    content: str,
    line_start: int,
    line_end: int,
) -> int:
    """Insert or update a documentation chunk. Returns chunk_id."""
    db.execute(
        """
        INSERT INTO doc_chunks (doc_file_id, chunk_index, section_title,
                               content, line_start, line_end)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(doc_file_id, chunk_index) DO UPDATE SET
            section_title = excluded.section_title,
            content       = excluded.content,
            line_start    = excluded.line_start,
            line_end      = excluded.line_end
        """,
        (doc_file_id, chunk_index, section_title, content, line_start, line_end),
    )
    db.commit()
    row = db.execute(
        "SELECT id FROM doc_chunks WHERE doc_file_id = ? AND chunk_index = ?",
        (doc_file_id, chunk_index),
    ).fetchone()
    return row[0]


def upsert_doc_embedding(db: sqlite3.Connection, chunk_id: int, embedding: list[float]) -> None:
    """Insert or replace a documentation chunk's dense vector embedding."""
    import struct

    blob = struct.pack(f"{len(embedding)}f", *embedding)
    db.execute("DELETE FROM doc_embeddings WHERE chunk_id = ?", (chunk_id,))
    db.execute(
        "INSERT INTO doc_embeddings (chunk_id, embedding) VALUES (?, ?)",
        (chunk_id, blob),
    )
    db.commit()
