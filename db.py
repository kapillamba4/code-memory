"""
Database layer for code-memory.

Manages a local SQLite database with three storage layers:
  1. Relational tables (files, symbols, references)
  2. FTS5 full-text index (symbols_fts) for BM25 keyword search
  3. sqlite-vec virtual table (symbol_embeddings) for dense vector search

All writes use upsert semantics so re-indexing is idempotent.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import contextmanager
from typing import TYPE_CHECKING

import sqlite_vec
import xxhash

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding model (lazy-loaded singleton)
# ---------------------------------------------------------------------------

_model = None
_embedding_dim = None

# Model identifier - can be overridden via EMBEDDING_MODEL environment variable
DEFAULT_EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def get_embedding_model():
    """Lazy-load and cache the sentence-transformers model."""
    global _model, _embedding_dim
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(
            EMBEDDING_MODEL_NAME, trust_remote_code=True
        )
        # Cache the embedding dimension from the model
        _embedding_dim = _model.get_sentence_embedding_dimension()
        logger.info(f"Loaded embedding model '{EMBEDDING_MODEL_NAME}' with dimension: {_embedding_dim}")
    return _model


def get_embedding_dim() -> int:
    """Get the embedding dimension from the model.

    Loads the model if not already loaded.
    Returns the native embedding dimension of the model.
    """
    if _embedding_dim is None:
        get_embedding_model()
    return _embedding_dim


def embed_text(text: str, task_type: str = "nl2code") -> list[float]:
    """Generate a dense vector embedding for *text*.

    Uses jina-code-embeddings with task prefix for better code retrieval.

    Args:
        text: The text to embed.
        task_type: One of 'nl2code', 'code2code', 'code2nl', 'code2completion', 'qa'.
    """
    model = get_embedding_model()
    prefixed_text = f"{task_type}: {text}"
    vec = model.encode(prefixed_text, normalize_embeddings=True, show_progress_bar=False)
    return vec.tolist()


def embed_texts_batch(
    texts: list[str], batch_size: int = 32, task_type: str = "nl2code"
) -> list[list[float]]:
    """Generate embeddings for multiple texts at once.

    This is significantly faster than calling embed_text() in a loop
    because sentence-transformers is optimized for batch processing.

    Args:
        texts: List of text strings to embed.
        batch_size: Number of texts to process per batch (default 32).
        task_type: One of 'nl2code', 'code2code', 'code2nl', 'code2completion', 'qa'.

    Returns:
        List of embedding vectors (same order as input texts).
    """
    if not texts:
        return []

    model = get_embedding_model()

    # Add task prefix to all texts
    prefixed_texts = [f"{task_type}: {text}" for text in texts]

    # Batch encode with normalization
    vectors = model.encode(
        prefixed_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    return [v.tolist() for v in vectors]


def warmup_embedding_model() -> None:
    """Pre-load and warm up the embedding model.

    Call this at server startup to avoid cold-start latency on first search.
    The warmup encodes a dummy string to initialize internal tensors.
    """
    model = get_embedding_model()
    # Warmup encode to initialize lazy-loaded components
    model.encode("nl2code: warmup", normalize_embeddings=True, show_progress_bar=False)
    logger.info("Embedding model warmed up")


# ---------------------------------------------------------------------------
# Transaction support
# ---------------------------------------------------------------------------


@contextmanager
def transaction(db: sqlite3.Connection):
    """Context manager for explicit transaction control.

    Disables autocommit, yields control, then commits on success.
    On exception, rolls back automatically.

    Example:
        with transaction(db):
            for item in items:
                upsert_symbol(db, ..., auto_commit=False)
        # Single commit here
    """
    # Disable autocommit by starting a transaction
    db.execute("BEGIN")
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise


# ---------------------------------------------------------------------------
# Database initialisation
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
-- 0. Metadata table for tracking index version and model info
CREATE TABLE IF NOT EXISTS index_metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

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


def get_db(project_dir: str) -> sqlite3.Connection:
    """Open (or create) the database, load sqlite-vec, and ensure schema.

    The database is stored as {project_dir}/code_memory.db to ensure each
    project has its own isolated index.

    If the embedding model has changed since the last index, all indexed data
    is automatically invalidated and the index will need to be rebuilt.

    Args:
        project_dir: The project directory where code_memory.db will be stored.

    Returns:
        A ready-to-use ``sqlite3.Connection`` with WAL mode and foreign keys.
    """
    import os
    db_path = os.path.join(os.path.abspath(project_dir), "code_memory.db")
    db = sqlite3.connect(db_path, check_same_thread=False)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")

    db.executescript(_SCHEMA_SQL)

    # Get embedding dimension from the model (loads model if needed)
    embedding_dim = get_embedding_dim()

    # Check if the embedding model has changed
    stored_model = db.execute(
        "SELECT value FROM index_metadata WHERE key = 'embedding_model'"
    ).fetchone()
    stored_dim = db.execute(
        "SELECT value FROM index_metadata WHERE key = 'embedding_dim'"
    ).fetchone()

    model_changed = (
        stored_model is None
        or stored_model[0] != EMBEDDING_MODEL_NAME
        or stored_dim is None
        or int(stored_dim[0]) != embedding_dim
    )

    if model_changed:
        if stored_model is not None:
            # Model changed - invalidate existing index
            logger.info(
                f"Embedding model changed from '{stored_model[0] if stored_model else 'none'}' "
                f"to '{EMBEDDING_MODEL_NAME}'. Invalidating index..."
            )
            _invalidate_index(db, embedding_dim)
        else:
            # New database - just create the embedding tables
            _create_embedding_tables(db, embedding_dim)

        # Store the current model info
        db.execute(
            "INSERT OR REPLACE INTO index_metadata (key, value) VALUES ('embedding_model', ?)",
            (EMBEDDING_MODEL_NAME,)
        )
        db.execute(
            "INSERT OR REPLACE INTO index_metadata (key, value) VALUES ('embedding_dim', ?)",
            (str(embedding_dim),)
        )
        db.commit()

    return db


def _invalidate_index(db: sqlite3.Connection, embedding_dim: int) -> None:
    """Invalidate the index by clearing all data and recreating embedding tables.

    This is called when the embedding model changes.
    """
    # Drop existing embedding virtual tables
    db.execute("DROP TABLE IF EXISTS symbol_embeddings")
    db.execute("DROP TABLE IF EXISTS doc_embeddings")

    # Clear all indexed data (cascades will handle related data via foreign keys,
    # but we need to be explicit since FK enforcement may vary)
    db.execute("DELETE FROM symbol_embeddings")
    db.execute("DELETE FROM doc_embeddings")
    db.execute("DELETE FROM symbols")
    db.execute("DELETE FROM files")
    db.execute("DELETE FROM references_")
    db.execute("DELETE FROM doc_chunks")
    db.execute("DELETE FROM doc_files")

    # Recreate embedding tables with new dimension
    _create_embedding_tables(db, embedding_dim)
    logger.info("Index invalidated and embedding tables recreated")


def _create_embedding_tables(db: sqlite3.Connection, embedding_dim: int) -> None:
    """Create the embedding virtual tables with the specified dimension."""
    # sqlite-vec virtual table for code embeddings
    db.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS symbol_embeddings
        USING vec0(
            symbol_id INTEGER PRIMARY KEY,
            embedding float[{embedding_dim}]
        )
        """
    )

    # sqlite-vec virtual table for documentation embeddings
    db.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS doc_embeddings
        USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding float[{embedding_dim}]
        )
        """
    )


# ---------------------------------------------------------------------------
# Upsert helpers
# ---------------------------------------------------------------------------


def file_hash(filepath: str) -> str:
    """Compute fast non-cryptographic hash of a file's contents.

    Uses xxHash (xxh64) which is ~10x faster than SHA-256 while still
    providing excellent collision resistance for change detection.

    Args:
        filepath: Path to the file to hash.

    Returns:
        Hexadecimal string representation of the 64-bit hash.
    """
    h = xxhash.xxh64()
    with open(filepath, "rb") as f:
        # Read in 64KB chunks for memory efficiency
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def upsert_file(
    db: sqlite3.Connection,
    path: str,
    last_modified: float,
    fhash: str,
    auto_commit: bool = True,
) -> int:
    """Insert or update a file record. Returns the file_id."""
    db.execute(
        """
        INSERT INTO files (path, last_modified, file_hash)
        VALUES (?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            last_modified = excluded.last_modified,
            file_hash     = excluded.file_hash
        """,
        (path, last_modified, fhash),
    )
    if auto_commit:
        db.commit()
    # Fetch the id (needed because last_insert_rowid isn't reliable on update)
    row = db.execute("SELECT id FROM files WHERE path = ?", (path,)).fetchone()
    return row[0]


def delete_file_data(db: sqlite3.Connection, file_id: int, auto_commit: bool = True) -> None:
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
    if auto_commit:
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
    auto_commit: bool = True,
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
    if auto_commit:
        db.commit()
    row = db.execute(
        "SELECT id FROM symbols WHERE file_id = ? AND name = ? AND kind = ? AND line_start = ?",
        (file_id, name, kind, line_start),
    ).fetchone()
    return row[0]


def upsert_reference(
    db: sqlite3.Connection,
    symbol_name: str,
    file_id: int,
    line_number: int,
    auto_commit: bool = True,
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
    if auto_commit:
        db.commit()


def upsert_embedding(
    db: sqlite3.Connection,
    symbol_id: int,
    embedding: list[float],
    auto_commit: bool = True,
) -> None:
    """Insert or replace a symbol's dense vector embedding."""
    import struct

    blob = struct.pack(f"{len(embedding)}f", *embedding)
    # sqlite-vec doesn't support ON CONFLICT, so delete-then-insert
    db.execute("DELETE FROM symbol_embeddings WHERE symbol_id = ?", (symbol_id,))
    db.execute(
        "INSERT INTO symbol_embeddings (symbol_id, embedding) VALUES (?, ?)",
        (symbol_id, blob),
    )
    if auto_commit:
        db.commit()


# ---------------------------------------------------------------------------
# Documentation upsert helpers (Milestone 4)
# ---------------------------------------------------------------------------


def upsert_doc_file(
    db: sqlite3.Connection,
    path: str,
    last_modified: float,
    fhash: str,
    doc_type: str,
    auto_commit: bool = True,
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
    if auto_commit:
        db.commit()
    row = db.execute("SELECT id FROM doc_files WHERE path = ?", (path,)).fetchone()
    return row[0]


def delete_doc_file_data(db: sqlite3.Connection, doc_file_id: int, auto_commit: bool = True) -> None:
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
    if auto_commit:
        db.commit()


def upsert_doc_chunk(
    db: sqlite3.Connection,
    doc_file_id: int,
    chunk_index: int,
    section_title: str | None,
    content: str,
    line_start: int,
    line_end: int,
    auto_commit: bool = True,
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
    if auto_commit:
        db.commit()
    row = db.execute(
        "SELECT id FROM doc_chunks WHERE doc_file_id = ? AND chunk_index = ?",
        (doc_file_id, chunk_index),
    ).fetchone()
    return row[0]


def upsert_doc_embedding(
    db: sqlite3.Connection,
    chunk_id: int,
    embedding: list[float],
    auto_commit: bool = True,
) -> None:
    """Insert or replace a documentation chunk's dense vector embedding."""
    import struct

    blob = struct.pack(f"{len(embedding)}f", *embedding)
    db.execute("DELETE FROM doc_embeddings WHERE chunk_id = ?", (chunk_id,))
    db.execute(
        "INSERT INTO doc_embeddings (chunk_id, embedding) VALUES (?, ?)",
        (chunk_id, blob),
    )
    if auto_commit:
        db.commit()


# ---------------------------------------------------------------------------
# Index Statistics
# ---------------------------------------------------------------------------

def get_index_stats(db: sqlite3.Connection, project_dir: str) -> dict:
    """Get comprehensive statistics about the index.

    Args:
        db: An open sqlite3.Connection.
        project_dir: The project directory path.

    Returns:
        Dictionary with index health metrics including:
        - Total symbols, files, doc chunks indexed
        - Index freshness (last indexed timestamps)
        - Embedding model info and dimension
        - Database size and WAL status
    """
    import os

    # Get counts
    symbols_count = db.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
    files_count = db.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    doc_chunks_count = db.execute("SELECT COUNT(*) FROM doc_chunks").fetchone()[0]
    doc_files_count = db.execute("SELECT COUNT(*) FROM doc_files").fetchone()[0]
    references_count = db.execute("SELECT COUNT(*) FROM references_").fetchone()[0]
    symbol_embeddings_count = db.execute("SELECT COUNT(*) FROM symbol_embeddings").fetchone()[0]
    doc_embeddings_count = db.execute("SELECT COUNT(*) FROM doc_embeddings").fetchone()[0]

    # Get symbol kinds distribution
    symbol_kinds = dict(db.execute(
        "SELECT kind, COUNT(*) FROM symbols GROUP BY kind ORDER BY COUNT(*) DESC"
    ).fetchall())

    # Get file types distribution (by extension)
    file_extensions = dict(db.execute(
        """SELECT substr(path, instr(path, '.')) as ext, COUNT(*) as cnt
           FROM files
           WHERE path LIKE '%.%'
           GROUP BY ext
           ORDER BY cnt DESC
           LIMIT 10"""
    ).fetchall())

    # Get last indexed timestamps
    last_file_indexed = db.execute(
        "SELECT MAX(last_modified) FROM files"
    ).fetchone()[0]
    last_doc_indexed = db.execute(
        "SELECT MAX(last_modified) FROM doc_files"
    ).fetchone()[0]

    # Get embedding model info
    embedding_model = db.execute(
        "SELECT value FROM index_metadata WHERE key = 'embedding_model'"
    ).fetchone()
    embedding_dim = db.execute(
        "SELECT value FROM index_metadata WHERE key = 'embedding_dim'"
    ).fetchone()

    # Database file size
    db_path = os.path.join(os.path.abspath(project_dir), "code_memory.db")
    db_size_bytes = os.path.getsize(db_path) if os.path.exists(db_path) else 0
    db_size_mb = round(db_size_bytes / (1024 * 1024), 2)

    # WAL status
    wal_path = db_path + "-wal"
    wal_exists = os.path.exists(wal_path)
    wal_size_mb = round(os.path.getsize(wal_path) / (1024 * 1024), 2) if wal_exists else 0

    # Check journal mode
    journal_mode = db.execute("PRAGMA journal_mode").fetchone()[0]

    return {
        "indexed": symbols_count > 0 or doc_chunks_count > 0,
        "counts": {
            "symbols": symbols_count,
            "files": files_count,
            "doc_chunks": doc_chunks_count,
            "doc_files": doc_files_count,
            "references": references_count,
            "symbol_embeddings": symbol_embeddings_count,
            "doc_embeddings": doc_embeddings_count,
        },
        "distributions": {
            "symbol_kinds": symbol_kinds,
            "file_extensions": file_extensions,
        },
        "freshness": {
            "last_file_indexed": last_file_indexed,
            "last_doc_indexed": last_doc_indexed,
        },
        "embedding": {
            "model": embedding_model[0] if embedding_model else None,
            "dimension": int(embedding_dim[0]) if embedding_dim else None,
        },
        "database": {
            "size_mb": db_size_mb,
            "journal_mode": journal_mode,
            "wal_exists": wal_exists,
            "wal_size_mb": wal_size_mb,
        },
    }
