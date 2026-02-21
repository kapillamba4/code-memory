"""
Query layer for code-memory.

Provides hybrid retrieval (BM25 + dense vector) with Reciprocal Rank Fusion,
plus specialised query functions for definitions, references, and file
structure.
"""

from __future__ import annotations

import struct
from typing import Any

import db as db_mod


# ---------------------------------------------------------------------------
# Hybrid search (BM25 + vector → RRF)
# ---------------------------------------------------------------------------

_RRF_K = 60  # standard RRF constant


def _bm25_search(query: str, db, top_k: int = 50) -> list[dict]:
    """Run FTS5 BM25 search against ``symbols_fts``.

    Returns a ranked list of dicts with ``symbol_id`` and ``bm25_score``.
    """
    # FTS5 MATCH query — escape double-quotes in user input
    safe_query = query.replace('"', '""')
    try:
        rows = db.execute(
            """
            SELECT s.id, s.name, s.kind, f.path, s.line_start, s.line_end,
                   s.source_text, bm25(symbols_fts) AS score
            FROM symbols_fts
            JOIN symbols s ON s.id = symbols_fts.rowid
            JOIN files   f ON f.id = s.file_id
            WHERE symbols_fts MATCH ?
            ORDER BY score          -- bm25() returns negative; lower = better
            LIMIT ?
            """,
            (safe_query, top_k),
        ).fetchall()
    except Exception:
        # FTS MATCH can fail on certain queries (e.g. operators only)
        return []

    return [
        {
            "symbol_id": r[0],
            "name": r[1],
            "kind": r[2],
            "file_path": r[3],
            "line_start": r[4],
            "line_end": r[5],
            "source_text": r[6],
            "bm25_score": r[7],
        }
        for r in rows
    ]


def _vector_search(query: str, db, top_k: int = 50) -> list[dict]:
    """Run dense vector nearest-neighbour search via ``sqlite-vec``.

    Returns a ranked list of dicts with ``symbol_id`` and ``vec_distance``.
    """
    query_vec = db_mod.embed_text(query)
    query_blob = struct.pack(f"{len(query_vec)}f", *query_vec)

    rows = db.execute(
        """
        SELECT se.symbol_id, se.distance,
               s.name, s.kind, f.path, s.line_start, s.line_end, s.source_text
        FROM symbol_embeddings se
        JOIN symbols s ON s.id = se.symbol_id
        JOIN files   f ON f.id = s.file_id
        WHERE se.embedding MATCH ?
        AND   se.k = ?
        ORDER BY se.distance
        """,
        (query_blob, top_k),
    ).fetchall()

    return [
        {
            "symbol_id": r[0],
            "vec_distance": r[1],
            "name": r[2],
            "kind": r[3],
            "file_path": r[4],
            "line_start": r[5],
            "line_end": r[6],
            "source_text": r[7],
        }
        for r in rows
    ]


def hybrid_search(query: str, db, top_k: int = 10) -> list[dict]:
    """Hybrid BM25 + vector search with Reciprocal Rank Fusion.

    Runs both retrieval legs independently, then merges their ranked lists
    using RRF:  ``rrf_score(d) = Σ 1 / (k + rank(d))``  where ``k = 60``.

    Args:
        query: Free-text search query.
        db: An open ``sqlite3.Connection`` from ``db.get_db()``.
        top_k: Number of results to return.

    Returns:
        A list of result dicts sorted by descending RRF score.
    """
    bm25_results = _bm25_search(query, db, top_k=50)
    vec_results = _vector_search(query, db, top_k=50)

    # Build RRF score map keyed by symbol_id
    scores: dict[int, float] = {}
    details: dict[int, dict] = {}

    for rank, r in enumerate(bm25_results, start=1):
        sid = r["symbol_id"]
        scores[sid] = scores.get(sid, 0.0) + 1.0 / (_RRF_K + rank)
        details[sid] = {
            "name": r["name"],
            "kind": r["kind"],
            "file_path": r["file_path"],
            "line_start": r["line_start"],
            "line_end": r["line_end"],
            "source_text": r["source_text"],
        }

    for rank, r in enumerate(vec_results, start=1):
        sid = r["symbol_id"]
        scores[sid] = scores.get(sid, 0.0) + 1.0 / (_RRF_K + rank)
        if sid not in details:
            details[sid] = {
                "name": r["name"],
                "kind": r["kind"],
                "file_path": r["file_path"],
                "line_start": r["line_start"],
                "line_end": r["line_end"],
                "source_text": r["source_text"],
            }

    # Sort by descending RRF score
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    return [
        {**details[sid], "score": round(score, 6)}
        for sid, score in ranked
    ]


# ---------------------------------------------------------------------------
# Tool-facing query functions
# ---------------------------------------------------------------------------


def find_definition(symbol_name: str, db) -> list[dict]:
    """Find where *symbol_name* is defined using hybrid search.

    Post-filters for exact name matches first; falls back to top hybrid
    results as "best guesses" if no exact match is found.

    Args:
        symbol_name: The name of the symbol to find.
        db: An open ``sqlite3.Connection``.

    Returns:
        A list of result dicts.
    """
    results = hybrid_search(symbol_name, db, top_k=20)

    # Exact-match filter (case-sensitive)
    exact = [r for r in results if r["name"] == symbol_name]
    if exact:
        return exact

    # Fallback: return top results as best guesses
    return results[:5]


def find_references(symbol_name: str, db) -> list[dict]:
    """Find all cross-references to *symbol_name*.

    Queries the ``references_`` table for exact matches.

    Args:
        symbol_name: The name of the symbol to find references for.
        db: An open ``sqlite3.Connection``.

    Returns:
        A list of dicts with ``symbol_name``, ``file_path``, ``line_number``.
    """
    rows = db.execute(
        """
        SELECT r.symbol_name, f.path, r.line_number
        FROM references_ r
        JOIN files f ON f.id = r.file_id
        WHERE r.symbol_name = ?
        ORDER BY f.path, r.line_number
        """,
        (symbol_name,),
    ).fetchall()

    return [
        {"symbol_name": r[0], "file_path": r[1], "line_number": r[2]}
        for r in rows
    ]


def get_file_structure(file_path: str, db) -> list[dict]:
    """List all symbols in a given file, ordered by line number.

    Args:
        file_path: Absolute (or matching) path to the file.
        db: An open ``sqlite3.Connection``.

    Returns:
        A list of dicts with ``name``, ``kind``, ``line_start``, ``line_end``,
        ``parent``.
    """
    import os

    abs_path = os.path.abspath(file_path)

    rows = db.execute(
        """
        SELECT s.name, s.kind, s.line_start, s.line_end,
               p.name AS parent_name
        FROM symbols s
        JOIN files f ON f.id = s.file_id
        LEFT JOIN symbols p ON p.id = s.parent_symbol_id
        WHERE f.path = ?
        ORDER BY s.line_start
        """,
        (abs_path,),
    ).fetchall()

    return [
        {
            "name": r[0],
            "kind": r[1],
            "line_start": r[2],
            "line_end": r[3],
            "parent": r[4],
        }
        for r in rows
    ]
