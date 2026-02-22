"""
Query layer for code-memory.

Provides hybrid retrieval (BM25 + dense vector) with Reciprocal Rank Fusion,
plus specialised query functions for definitions, references, and file
structure.
"""

from __future__ import annotations

import struct

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


# ---------------------------------------------------------------------------
# Documentation search (Milestone 4)
# ---------------------------------------------------------------------------


def _doc_bm25_search(query: str, db, top_k: int = 50) -> list[dict]:
    """Run FTS5 BM25 search against ``doc_chunks_fts``.

    Returns a ranked list of dicts with chunk metadata and bm25_score.
    """
    safe_query = query.replace('"', '""')
    try:
        rows = db.execute(
            """
            SELECT dc.id, dc.section_title, dc.content, df.path, df.doc_type,
                   dc.line_start, dc.line_end, bm25(doc_chunks_fts) AS score
            FROM doc_chunks_fts
            JOIN doc_chunks dc ON dc.id = doc_chunks_fts.rowid
            JOIN doc_files   df ON df.id = dc.doc_file_id
            WHERE doc_chunks_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (safe_query, top_k),
        ).fetchall()
    except Exception:
        return []

    return [
        {
            "chunk_id": r[0],
            "section_title": r[1],
            "content": r[2],
            "source_file": r[3],
            "doc_type": r[4],
            "line_start": r[5],
            "line_end": r[6],
            "bm25_score": r[7],
        }
        for r in rows
    ]


def _doc_vector_search(query: str, db, top_k: int = 50) -> list[dict]:
    """Run dense vector nearest-neighbour search on doc_embeddings."""
    query_vec = db_mod.embed_text(query)
    query_blob = struct.pack(f"{len(query_vec)}f", *query_vec)

    rows = db.execute(
        """
        SELECT de.chunk_id, de.distance,
               dc.section_title, dc.content, df.path, df.doc_type,
               dc.line_start, dc.line_end
        FROM doc_embeddings de
        JOIN doc_chunks dc ON dc.id = de.chunk_id
        JOIN doc_files   df ON df.id = dc.doc_file_id
        WHERE de.embedding MATCH ?
        AND   de.k = ?
        ORDER BY de.distance
        """,
        (query_blob, top_k),
    ).fetchall()

    return [
        {
            "chunk_id": r[0],
            "vec_distance": r[1],
            "section_title": r[2],
            "content": r[3],
            "source_file": r[4],
            "doc_type": r[5],
            "line_start": r[6],
            "line_end": r[7],
        }
        for r in rows
    ]


def search_documentation(query: str, db, top_k: int = 10,
                         include_context: bool = False) -> list[dict]:
    """Perform hybrid search over documentation chunks.

    Uses BM25 + vector search with Reciprocal Rank Fusion.

    Args:
        query: Natural language query.
        db: Database connection.
        top_k: Maximum results to return.
        include_context: If True, include adjacent chunks for context.

    Returns:
        List of matching chunks with source attribution and RRF scores.
    """
    bm25_results = _doc_bm25_search(query, db, top_k=50)
    vec_results = _doc_vector_search(query, db, top_k=50)

    # Build RRF score map keyed by chunk_id
    scores: dict[int, float] = {}
    details: dict[int, dict] = {}

    for rank, r in enumerate(bm25_results, start=1):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank)
        details[cid] = {
            "content": r["content"],
            "source_file": r["source_file"],
            "section_title": r["section_title"],
            "line_start": r["line_start"],
            "line_end": r["line_end"],
            "doc_type": r["doc_type"],
        }

    for rank, r in enumerate(vec_results, start=1):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank)
        if cid not in details:
            details[cid] = {
                "content": r["content"],
                "source_file": r["source_file"],
                "section_title": r["section_title"],
                "line_start": r["line_start"],
                "line_end": r["line_end"],
                "doc_type": r["doc_type"],
            }

    # Sort by descending RRF score
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    results = [
        {**details[cid], "score": round(score, 6)}
        for cid, score in ranked
    ]

    # Optionally include adjacent chunks for context
    if include_context and results:
        results = _add_context_chunks(results, db)

    return results


def _add_context_chunks(results: list[dict], db) -> list[dict]:
    """Add adjacent chunks to results for additional context."""
    enriched = []

    for result in results:
        # Get the chunk's file and index
        row = db.execute(
            """
            SELECT dc.chunk_index, dc.doc_file_id
            FROM doc_chunks dc
            JOIN doc_files df ON df.id = dc.doc_file_id
            WHERE df.path = ? AND dc.line_start = ? AND dc.line_end = ?
            """,
            (result["source_file"], result["line_start"], result["line_end"]),
        ).fetchone()

        if not row:
            enriched.append(result)
            continue

        chunk_index, doc_file_id = row

        # Get previous and next chunks
        context_parts = []

        prev = db.execute(
            """
            SELECT content FROM doc_chunks
            WHERE doc_file_id = ? AND chunk_index = ?
            """,
            (doc_file_id, chunk_index - 1),
        ).fetchone()
        if prev:
            context_parts.append({"type": "previous", "content": prev[0][:200]})

        context_parts.append({"type": "current", "content": result["content"]})

        next_chunk = db.execute(
            """
            SELECT content FROM doc_chunks
            WHERE doc_file_id = ? AND chunk_index = ?
            """,
            (doc_file_id, chunk_index + 1),
        ).fetchone()
        if next_chunk:
            context_parts.append({"type": "next", "content": next_chunk[0][:200]})

        enriched.append({
            **result,
            "context": context_parts,
        })

    return enriched


# ---------------------------------------------------------------------------
# Topic Discovery (Semantic Code Search)
# ---------------------------------------------------------------------------


def discover_topic(topic_query: str, db, top_k: int = 15) -> list[dict]:
    """Discover files and code related to a high-level topic or feature.

    This function performs broad semantic search across both code symbols
    AND documentation chunks to find all files related to a conceptual topic.
    Results are aggregated and deduplicated by file path.

    This is the PRIMARY function for "find all files related to X" queries
    where X is a feature, domain concept, or topic (e.g., "auth", "workouts",
    "payment processing", "user notifications").

    Args:
        topic_query: A natural language topic, feature name, or domain concept.
                     Examples: "authentication", "workout tracking", "email notifications"
        db: An open ``sqlite3.Connection``.
        top_k: Maximum number of files to return (default 15).

    Returns:
        A list of file-level results, each containing:
        - file_path: Path to the relevant file
        - relevance_score: Combined semantic relevance score
        - matched_symbols: List of symbol names that matched the topic
        - matched_docs: List of doc section titles that matched
        - summary: Brief description of what in this file is relevant
    """
    # Run parallel searches on both code symbols and documentation
    code_results = hybrid_search(topic_query, db, top_k=50)
    doc_results = search_documentation(topic_query, db, top_k=50)

    # Aggregate by file path, collecting all matched items
    file_aggregates: dict[str, dict] = {}

    for r in code_results:
        fp = r.get("file_path", "")
        if not fp:
            continue
        if fp not in file_aggregates:
            file_aggregates[fp] = {
                "file_path": fp,
                "relevance_score": 0.0,
                "matched_symbols": [],
                "matched_docs": [],
                "symbol_kinds": set(),
            }
        file_aggregates[fp]["relevance_score"] += r.get("score", 0.5)
        file_aggregates[fp]["matched_symbols"].append(r.get("name", ""))
        file_aggregates[fp]["symbol_kinds"].add(r.get("kind", ""))

    for r in doc_results:
        fp = r.get("source_file", "")
        if not fp:
            continue
        if fp not in file_aggregates:
            file_aggregates[fp] = {
                "file_path": fp,
                "relevance_score": 0.0,
                "matched_symbols": [],
                "matched_docs": [],
                "symbol_kinds": set(),
            }
        file_aggregates[fp]["relevance_score"] += r.get("score", 0.5)
        section = r.get("section_title", "")
        if section:
            file_aggregates[fp]["matched_docs"].append(section)

    # Sort by relevance and take top_k
    sorted_files = sorted(
        file_aggregates.values(),
        key=lambda x: x["relevance_score"],
        reverse=True
    )[:top_k]

    # Build final results with summaries
    results = []
    for item in sorted_files:
        # Generate a summary of what matched
        symbol_summary = ", ".join(item["matched_symbols"][:5])
        if len(item["matched_symbols"]) > 5:
            symbol_summary += f" (+{len(item['matched_symbols']) - 5} more)"

        kinds = ", ".join(k for k in item["symbol_kinds"] if k)

        results.append({
            "file_path": item["file_path"],
            "relevance_score": round(item["relevance_score"], 4),
            "matched_symbols": item["matched_symbols"][:10],
            "matched_docs": item["matched_docs"][:5],
            "symbol_kinds": kinds,
            "summary": f"Contains {kinds}: {symbol_summary}" if kinds else f"Related symbols: {symbol_summary}",
        })

    return results
