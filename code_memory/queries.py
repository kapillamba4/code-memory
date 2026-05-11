"""
Query layer for code-memory.

Provides hybrid retrieval (BM25 + dense vector) with Reciprocal Rank Fusion,
plus specialised query functions for definitions, references, and file
structure.
"""

from __future__ import annotations

import logging
import struct

from . import db as db_mod
from . import validation as val

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hybrid search (BM25 + vector → RRF)
# ---------------------------------------------------------------------------

_RRF_K = 60  # standard RRF constant


def _bm25_search(query: str, db, top_k: int = 50) -> list[dict]:
    """Run FTS5 BM25 search against ``symbols_fts``.

    Returns a ranked list of dicts with ``symbol_id`` and ``bm25_score``.
    """
    # FTS5 MATCH query — escape double-quotes and special characters in user input
    safe_query = val.sanitize_fts_query(query)
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
    except Exception as exc:
        # FTS MATCH can fail on certain queries (e.g. operators only)
        logger.warning("BM25 code search failed for query %r: %s", query, exc)
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


def hybrid_search(query: str, db, top_k: int = 10, rerank: bool = True) -> list[dict]:
    """Hybrid BM25 + vector search with Reciprocal Rank Fusion.

    Runs both retrieval legs independently, then merges their ranked lists
    using RRF:  ``rrf_score(d) = Σ 1 / (k + rank(d))``  where ``k = 60``.

    Optionally reranks results using a cross-encoder for improved precision.

    Args:
        query: Free-text search query.
        db: An open ``sqlite3.Connection`` from ``db.get_db()``.
        top_k: Number of results to return.
        rerank: If True (default), apply cross-encoder reranking when available.

    Returns:
        A list of result dicts sorted by descending RRF score (or rerank score),
        including match_reason, match_highlights, and confidence.
    """
    bm25_results = _bm25_search(query, db, top_k=50)
    vec_results = _vector_search(query, db, top_k=50)

    # Build RRF score map keyed by symbol_id
    scores: dict[int, float] = {}
    details: dict[int, dict] = {}
    match_sources: dict[int, list[str]] = {}  # Track which search found each result

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
        match_sources[sid] = match_sources.get(sid, [])
        match_sources[sid].append("bm25")

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
        match_sources[sid] = match_sources.get(sid, [])
        if "vector" not in match_sources[sid]:
            match_sources[sid].append("vector")

    # Sort by descending RRF score
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    # Theoretical max RRF score: 1/(k+1) per source, 2/(k+1) for hybrid
    max_single_rrf = 1.0 / (_RRF_K + 1)  # ≈ 0.01639
    max_hybrid_rrf = 2.0 * max_single_rrf  # ≈ 0.03279

    # Build results with match metadata
    results = []
    for sid, raw_score in ranked:
        sources = match_sources.get(sid, [])
        is_hybrid = len(sources) == 2

        # Determine match reason
        if is_hybrid:
            match_reason = "hybrid (BM25 + semantic)"
        elif "bm25" in sources:
            match_reason = "keyword match (BM25)"
        else:
            match_reason = "semantic match (vector)"

        # Normalize score to 0-100 range for human readability.
        # Raw RRF scores are always tiny (~0.01-0.03) which is misleading as
        # a relevance indicator. Normalize against the theoretical maximum.
        max_rrf = max_hybrid_rrf if is_hybrid else max_single_rrf
        normalized_score = min(100.0, (raw_score / max_rrf) * 100.0)

        # Confidence: normalized score as 0-1 fraction.
        # No arbitrary cap — a single-source match can be 100% confident
        # if it's rank #1 in that source.
        confidence = round(normalized_score / 100.0, 3)

        result = {
            "symbol_id": sid,
            **details[sid],
            "score": round(normalized_score, 1),
            "match_reason": match_reason,
            "confidence": confidence,
            "match_highlights": [],  # Will be populated below if BM25 match
        }

        # Get highlights for BM25 matches using FTS5 highlight function
        if "bm25" in sources:
            highlights = _get_bm25_highlights(query, details[sid]["source_text"], db)
            result["match_highlights"] = highlights

        results.append(result)

    # Apply cross-encoder reranking for improved precision
    if rerank and db_mod.is_reranking_enabled():
        results = db_mod.rerank_results(query, results, top_k=top_k)

    return results


def _get_bm25_highlights(query: str, source_text: str, db) -> list[str]:
    """Extract highlighted snippets using FTS5.

    Returns up to 3 highlighted text snippets showing where the query matched.
    """
    if not source_text or not query:
        return []

    # Use FTS5 highlight function to get matched portions safely
    safe_query = val.sanitize_fts_query(query)
    try:
        # Create a temporary FTS5 query to get highlights
        # We use the snippet function which returns highlighted fragments
        rows = db.execute(
            """
            SELECT snippet(symbols_fts, 1, '>>>', '<<<', '...', 20) as highlight
            FROM symbols_fts
            WHERE symbols_fts MATCH ?
            LIMIT 3
            """,
            (safe_query,),
        ).fetchall()

        highlights = []
        for row in rows:
            if row[0] and row[0] not in ("...", ""):
                # Clean up the highlight markers for readability
                highlight = row[0].replace(">>>", "**").replace("<<<", "**")
                if len(highlight) > 10:  # Only include meaningful highlights
                    highlights.append(highlight)

        return highlights[:3]  # Return at most 3 highlights
    except Exception:
        # Fallback: find query terms in source text
        return _simple_highlights(query, source_text)


def _simple_highlights(query: str, source_text: str) -> list[str]:
    """Simple fallback highlight extraction when FTS5 isn't available."""
    highlights = []
    query_terms = query.lower().split()
    lines = source_text.split("\n")

    for line in lines[:20]:  # Check first 20 lines
        line_lower = line.lower()
        for term in query_terms:
            if term in line_lower and len(line.strip()) > 10:
                # Truncate long lines
                snippet = line.strip()[:100]
                if len(snippet) > 50:
                    snippet = snippet[:97] + "..."
                highlights.append(snippet)
                break
        if len(highlights) >= 3:
            break

    return highlights[:3]


# ---------------------------------------------------------------------------
# Tool-facing query functions
# ---------------------------------------------------------------------------


def find_definition(symbol_name: str, db, include_context: bool = True,
                    rerank: bool = True) -> list[dict]:
    """Find where *symbol_name* is defined using hybrid search.

    Post-filters for exact name matches first; falls back to top hybrid
    results as "best guesses" if no exact match is found.

    Args:
        symbol_name: The name of the symbol to find.
        db: An open ``sqlite3.Connection``.
        include_context: If True, include docstrings and parent symbol info.
        rerank: If True (default), apply cross-encoder reranking when available.

    Returns:
        A list of result dicts with enriched information.
    """
    results = hybrid_search(symbol_name, db, top_k=20, rerank=rerank)

    # Exact-match filter (case-sensitive)
    exact = [r for r in results if r["name"] == symbol_name]
    matched = exact if exact else results[:5]

    if not include_context:
        return matched

    # Enrich results with docstrings and parent information
    enriched = []
    for r in matched:
        symbol_id = r.get("symbol_id") or _get_symbol_id(r["name"], r["file_path"], db)
        enriched_result = {
            **r,
            "docstring": None,
            "parent": None,
            "signature": _extract_signature(r.get("source_text", "")),
        }

        # Get parent symbol
        if symbol_id:
            parent_row = db.execute(
                """
                SELECT p.name, p.kind
                FROM symbols s
                LEFT JOIN symbols p ON p.id = s.parent_symbol_id
                WHERE s.id = ?
                """,
                (symbol_id,),
            ).fetchone()
            if parent_row and parent_row[0]:
                enriched_result["parent"] = {"name": parent_row[0], "kind": parent_row[1]}

        # Get docstring from doc_chunks
        doc_row = db.execute(
            """
            SELECT dc.content
            FROM doc_chunks dc
            JOIN doc_files df ON df.id = dc.doc_file_id
            WHERE df.path = ? AND dc.line_start <= ? AND dc.line_end >= ?
            AND df.doc_type = 'docstring'
            LIMIT 1
            """,
            (r["file_path"], r["line_start"], r["line_start"]),
        ).fetchone()
        if doc_row:
            enriched_result["docstring"] = doc_row[0]

        enriched.append(enriched_result)

    return enriched


def _get_symbol_id(name: str, file_path: str, db) -> int | None:
    """Get symbol ID by name and file path."""
    row = db.execute(
        "SELECT id FROM symbols WHERE name = ? AND file_id = (SELECT id FROM files WHERE path = ?)",
        (name, file_path),
    ).fetchone()
    return row[0] if row else None


def _extract_signature(source_text: str) -> str | None:
    """Extract the function/class signature from source text."""
    if not source_text:
        return None
    lines = source_text.strip().split("\n")
    if not lines:
        return None
    # Return first meaningful line (signature)
    first_line = lines[0].strip()
    if len(first_line) > 100:
        return first_line[:100] + "..."
    return first_line if first_line else None


def find_references(symbol_name: str, db, include_context: bool = True) -> list[dict]:
    """Find all cross-references to *symbol_name*.

    Queries the ``references_`` table for exact matches.

    Args:
        symbol_name: The name of the symbol to find references for.
        db: An open ``sqlite3.Connection``.
        include_context: If True, include source context and containing symbol.

    Returns:
        A list of dicts with enriched reference information.
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

    if not include_context:
        return [
            {"symbol_name": r[0], "file_path": r[1], "line_number": r[2]}
            for r in rows
        ]

    # Enrich with context
    enriched = []
    file_cache: dict[str, list[str] | None] = {}
    for r in rows:
        ref = {
            "symbol_name": r[0],
            "file_path": r[1],
            "line_number": r[2],
            "source_line": None,
            "containing_symbol": None,
        }

        # Get the source line at this reference (cached per file)
        file_path = r[1]
        if file_path not in file_cache:
            try:
                with open(file_path) as f:
                    file_cache[file_path] = f.readlines()
            except Exception:
                file_cache[file_path] = None

        cached_lines = file_cache[file_path]
        if cached_lines and 0 < r[2] <= len(cached_lines):
            ref["source_line"] = cached_lines[r[2] - 1].strip()

        # Find containing symbol
        containing = db.execute(
            """
            SELECT s.name, s.kind
            FROM symbols s
            JOIN files f ON f.id = s.file_id
            WHERE f.path = ?
            AND s.line_start <= ? AND s.line_end >= ?
            ORDER BY (s.line_end - s.line_start)
            LIMIT 1
            """,
            (r[1], r[2], r[2]),
        ).fetchone()
        if containing:
            ref["containing_symbol"] = {"name": containing[0], "kind": containing[1]}

        enriched.append(ref)

    return enriched


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
    safe_query = val.sanitize_fts_query(query)
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
    except Exception as exc:
        logger.warning("BM25 doc search failed for query %r: %s", query, exc)
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
                         include_context: bool = False,
                         rerank: bool = True) -> list[dict]:
    """Perform hybrid search over documentation chunks.

    Uses BM25 + vector search with Reciprocal Rank Fusion.

    Args:
        query: Natural language query.
        db: Database connection.
        top_k: Maximum results to return.
        include_context: If True, include adjacent chunks for context.
        rerank: If True (default), apply cross-encoder reranking when available.

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

    # Normalize scores to 0-100 (same approach as hybrid_search).
    max_rrf = 2.0 / (_RRF_K + 1)  # theoretical max for hybrid hit
    results = [
        {**details[cid], "score": round(min(100.0, (raw / max_rrf) * 100.0), 1)}
        for cid, raw in ranked
    ]

    # Apply cross-encoder reranking for improved precision
    if rerank and db_mod.is_reranking_enabled():
        results = db_mod.rerank_results(query, results, top_k=top_k)

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


def discover_topic(topic_query: str, db, top_k: int = 15, include_snippets: bool = True,
                   rerank: bool = True) -> list[dict]:
    """Discover files and code related to a high-level topic or feature.

    This function performs broad semantic search across code symbols to find
    all files related to a conceptual topic. Results are aggregated and
    deduplicated by file path.

    This is the PRIMARY function for "find all files related to X" queries
    where X is a feature, domain concept, or topic (e.g., "auth", "workouts",
    "payment processing", "user notifications").

    Note: This function searches CODE only. Use search_docs() for documentation
    and README files.

    Args:
        topic_query: A natural language topic, feature name, or domain concept.
                     Examples: "authentication", "workout tracking", "email notifications"
        db: An open ``sqlite3.Connection``.
        top_k: Maximum number of files to return (default 15).
        include_snippets: If True, include code snippets for top symbols.
        rerank: If True (default), apply cross-encoder reranking when available.

    Returns:
        A list of file-level results, each containing:
        - file_path: Path to the relevant file
        - relevance_score: Combined semantic relevance score
        - matched_symbols: List of symbol names that matched the topic
        - symbol_kinds: Types of symbols found (function, class, etc.)
        - summary: Brief description of what in this file is relevant
        - top_snippets: Code snippets from top-matching symbols (if include_snippets)
    """
    # Search code symbols only (documentation is handled by search_docs)
    code_results = hybrid_search(topic_query, db, top_k=50, rerank=rerank)

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
                "symbol_kinds": set(),
                "symbol_details": [],  # Store full details for snippets
            }
        file_aggregates[fp]["relevance_score"] += r.get("score", 0.0)
        file_aggregates[fp]["matched_symbols"].append(r.get("name", ""))
        file_aggregates[fp]["symbol_kinds"].add(r.get("kind", ""))
        file_aggregates[fp]["symbol_details"].append({
            "name": r.get("name"),
            "kind": r.get("kind"),
            "line_start": r.get("line_start"),
            "line_end": r.get("line_end"),
            "source_text": r.get("source_text"),
            "score": r.get("score"),
        })

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

        result = {
            "file_path": item["file_path"],
            "relevance_score": round(item["relevance_score"], 4),
            "matched_symbols": item["matched_symbols"][:10],
            "symbol_kinds": kinds,
            "summary": f"Contains {kinds}: {symbol_summary}" if kinds else f"Related symbols: {symbol_summary}",
        }

        # Add top snippets if requested
        if include_snippets and item["symbol_details"]:
            # Sort by score and take top 2
            top_symbols = sorted(
                item["symbol_details"],
                key=lambda x: x.get("score", 0) or 0,
                reverse=True
            )[:2]
            result["top_snippets"] = [
                {
                    "name": s["name"],
                    "kind": s["kind"],
                    "line_range": f"{s['line_start']}-{s['line_end']}",
                    "code": _truncate_code(s.get("source_text", ""), max_lines=15),
                }
                for s in top_symbols if s.get("source_text")
            ]

        results.append(result)

    return results


def _truncate_code(source_text: str, max_lines: int = 15, max_chars: int = 500) -> str:
    """Truncate source code to a reasonable preview size."""
    if not source_text:
        return ""
    lines = source_text.strip().split("\n")
    if len(lines) <= max_lines and len(source_text) <= max_chars:
        return source_text.strip()
    truncated = "\n".join(lines[:max_lines])
    if len(truncated) > max_chars:
        truncated = truncated[:max_chars]
    return truncated + "\n// ... (truncated)"


# ---------------------------------------------------------------------------
# Dead-code detection
# ---------------------------------------------------------------------------

# Symbol kinds eligible for dead-code analysis.
_DEAD_CODE_DEFAULT_KINDS: tuple[str, ...] = ("function", "method", "class")
_DEAD_CODE_ALLOWED_KINDS: frozenset[str] = frozenset(_DEAD_CODE_DEFAULT_KINDS)

# Names that are almost always entry points or framework callbacks; never flag.
_DEAD_CODE_ENTRYPOINT_NAMES: frozenset[str] = frozenset({"main"})

# Languages where the AST reference extractor does NOT capture method calls
# made via member access (e.g. ``obj.method()``).  Methods defined in files
# with these extensions get a confidence penalty and an explicit caveat.
_MEMBER_ACCESS_BLIND_EXTENSIONS: frozenset[str] = frozenset({
    ".js", ".jsx", ".ts", ".tsx",
    ".go", ".rs",
    ".kt", ".kts",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx",
})

# Filenames that typically re-export public API (or list it via a string).
_REEXPORT_FILENAMES: frozenset[str] = frozenset({
    "__init__.py", "index.js", "index.jsx", "index.ts", "index.tsx",
    "mod.rs", "lib.rs",
})


def _is_test_path(path: str) -> bool:
    """Return True if *path* looks like a test or fixture file."""
    import os

    norm = path.replace("\\", "/").lower()
    basename = os.path.basename(norm)

    if basename == "conftest.py":
        return True
    if basename.startswith("test_"):
        return True
    if basename.endswith((
        "_test.py",
        ".test.js", ".test.jsx", ".test.ts", ".test.tsx",
        ".spec.js", ".spec.jsx", ".spec.ts", ".spec.tsx",
    )):
        return True

    parts = norm.split("/")
    return any(seg in {"tests", "test", "__tests__", "spec", "specs"} for seg in parts)


def _is_excluded_from_dead_code(
    name: str, kind: str, path: str, include_tests: bool
) -> tuple[bool, str | None]:
    """Return ``(excluded, reason)``.

    Symbols matching exclusion rules are never reported as dead code, regardless
    of reference count.  Used to filter framework hooks, entry points, and
    test fixtures that are invoked by mechanisms our reference extractor
    can't observe.
    """
    if not name or name.startswith("<anonymous@"):
        return True, "anonymous symbol"

    # Python dunder protocol methods: __init__, __call__, __str__, __enter__, ...
    if len(name) > 4 and name.startswith("__") and name.endswith("__"):
        return True, "dunder method (Python protocol)"

    if name in _DEAD_CODE_ENTRYPOINT_NAMES:
        return True, "common entry-point name"

    if kind == "file":
        return True, "file-level fallback symbol"

    if not include_tests and _is_test_path(path):
        return True, "in a test file (use include_tests=True to scan)"

    return False, None


def _has_decorator_above(path: str, line_start: int) -> bool:
    """Best-effort check for a decorator on the line(s) immediately above.

    Returns True if a non-blank line above ``line_start`` begins with ``@``.
    Falls back to False on read errors.  This heuristic catches Python and
    TypeScript decorators that aren't part of the symbol's source_text in
    the index.
    """
    try:
        with open(path) as f:
            lines = f.readlines()
    except OSError:
        return False

    # Walk upward past blank lines until we find content (or run out).
    idx = line_start - 2  # convert to zero-indexed line above line_start
    while idx >= 0:
        stripped = lines[idx].strip()
        if not stripped:
            idx -= 1
            continue
        return stripped.startswith("@")
    return False


def _score_dead_code_candidate(
    name: str,
    kind: str,
    path: str,
    name_share_count: int,
    has_decorator: bool,
) -> tuple[float, list[str]]:
    """Return ``(confidence, reasons)`` for a candidate that has no external refs.

    Confidence is in [0.0, 0.99] — never claims absolute certainty since
    dynamic dispatch, reflection, and string-based imports can always hide
    usages.  ``reasons`` is an ordered list of human-readable explanations.
    """
    import os

    reasons: list[str] = ["No references found outside this symbol's own definition"]
    confidence = 0.6

    ext = os.path.splitext(path)[1].lower()
    basename = os.path.basename(path)

    # ── Privacy ────────────────────────────────────────────────────────────
    if name.startswith("__") and not name.endswith("__"):
        confidence += 0.25
        reasons.append(f"Name '{name}' is name-mangled (Python private)")
    elif name.startswith("_"):
        confidence += 0.2
        reasons.append(f"Underscore-prefixed name '{name}' suggests an internal helper")
    else:
        confidence -= 0.05
        reasons.append(f"Public name '{name}' may be part of an exported API; verify before removing")

    # ── Name uniqueness ────────────────────────────────────────────────────
    if name_share_count > 1:
        confidence -= 0.3
        reasons.append(
            f"Name '{name}' is shared by {name_share_count} symbols; "
            "reference counts can't disambiguate which one is being used"
        )

    # ── Language / kind specific caveats ───────────────────────────────────
    if kind == "method":
        if ext in _MEMBER_ACCESS_BLIND_EXTENSIONS:
            confidence -= 0.3
            reasons.append(
                f"Method in a {ext} file: calls via member access (obj.{name}()) "
                "aren't captured by the reference index — verify manually"
            )
        else:
            reasons.append(
                f"Method in a {ext or 'unknown-ext'} file: dynamic dispatch may hide some callers"
            )
    elif kind == "class":
        confidence -= 0.05
        reasons.append("Class: dynamic instantiation (reflection, string lookup) may hide usages")

    # ── Re-export files ────────────────────────────────────────────────────
    if basename in _REEXPORT_FILENAMES:
        confidence -= 0.4
        reasons.append(f"Defined in {basename} — likely a re-export of a public API")

    # ── Decorators ─────────────────────────────────────────────────────────
    if has_decorator:
        confidence -= 0.25
        reasons.append("Decorated symbol — may be registered with a framework or DI system")

    confidence = max(0.0, min(0.99, confidence))
    return round(confidence, 3), reasons


def _source_excerpt(source_text: str | None, max_chars: int = 120) -> str | None:
    """Return the first non-empty trimmed line of *source_text*, truncated."""
    if not source_text:
        return None
    for line in source_text.splitlines():
        trimmed = line.strip()
        if trimmed:
            if len(trimmed) > max_chars:
                trimmed = trimmed[: max_chars - 3] + "..."
            return trimmed
    return None


def find_dead_code(
    db,
    *,
    min_confidence: float = 0.5,
    kinds: list[str] | None = None,
    include_tests: bool = False,
    top_k: int = 50,
) -> dict:
    """Find symbols that look like dead code.

    Cross-references the ``symbols`` table against the ``references_`` table
    to identify symbols with no reference outside their own body.  Each
    candidate is scored with a confidence in [0.0, 0.99] and a list of
    human-readable reasons.

    Args:
        db: Open ``sqlite3.Connection`` from ``db.get_db()``.
        min_confidence: Lower bound on confidence (filters out low-signal hits).
        kinds: Symbol kinds to consider; defaults to function/method/class.
        include_tests: If True, also scan test files (default False).
        top_k: Maximum candidates to return, sorted by confidence desc.

    Returns:
        Dict with:
          - candidates: list of result dicts with name, kind, file_path,
            line_start, line_end, confidence, reasons, source_excerpt.
          - scanned_symbols: count of symbols inspected after exclusions.
          - total_symbols: total symbols of the requested kinds in the index.
          - limitations: list of caveats explaining where false positives may
            arise (e.g., languages where member access isn't tracked).

    This function is a heuristic — it cannot detect symbols invoked via
    reflection, dynamic dispatch, string-based imports, or framework
    registration.  Treat results as candidates to investigate, not as a
    definitive deletion list.
    """
    import os

    requested_kinds = list(kinds) if kinds is not None else list(_DEAD_CODE_DEFAULT_KINDS)
    if not requested_kinds:
        return {
            "candidates": [],
            "scanned_symbols": 0,
            "total_symbols": 0,
            "limitations": [],
        }

    placeholders = ",".join("?" * len(requested_kinds))
    rows = db.execute(
        f"""
        SELECT s.id, s.name, s.kind, s.file_id, f.path,
               s.line_start, s.line_end, s.source_text
        FROM symbols s
        JOIN files f ON f.id = s.file_id
        WHERE s.kind IN ({placeholders})
        """,
        requested_kinds,
    ).fetchall()

    total_symbols = len(rows)
    if not rows:
        return {
            "candidates": [],
            "scanned_symbols": 0,
            "total_symbols": 0,
            "limitations": [],
        }

    # Count how many symbols share each name (across kinds in the index).
    name_share: dict[str, int] = {}
    for r in db.execute("SELECT name, COUNT(*) FROM symbols GROUP BY name").fetchall():
        name_share[r[0]] = r[1]

    # Prefetch all references and group by name; this avoids an N+1 query
    # pattern across thousands of candidate symbols.  Memory footprint is
    # ~24 bytes per ref, well within reasonable limits even for large repos.
    refs_by_name: dict[str, list[tuple[int, int]]] = {}
    for r_name, f_id, ln in db.execute(
        "SELECT symbol_name, file_id, line_number FROM references_"
    ).fetchall():
        refs_by_name.setdefault(r_name, []).append((f_id, ln))

    candidates: list[dict] = []
    scanned = 0
    seen_extensions: set[str] = set()

    for sid, name, kind, file_id, path, line_start, line_end, source_text in rows:
        excluded, _exclusion_reason = _is_excluded_from_dead_code(
            name, kind, path, include_tests
        )
        if excluded:
            continue

        scanned += 1
        seen_extensions.add(os.path.splitext(path)[1].lower())

        # An "external" reference is any reference to this name that lives
        # outside the symbol's own [line_start, line_end] body.  References
        # at the definition line and recursive self-calls are internal.
        has_external = False
        for ref_file_id, ref_line in refs_by_name.get(name, ()):
            if not (ref_file_id == file_id and line_start <= ref_line <= line_end):
                has_external = True
                break

        if has_external:
            continue

        confidence, reasons = _score_dead_code_candidate(
            name=name,
            kind=kind,
            path=path,
            name_share_count=name_share.get(name, 1),
            has_decorator=_has_decorator_above(path, line_start),
        )

        if confidence < min_confidence:
            continue

        candidates.append({
            "name": name,
            "kind": kind,
            "file_path": path,
            "line_start": line_start,
            "line_end": line_end,
            "confidence": confidence,
            "reasons": reasons,
            "source_excerpt": _source_excerpt(source_text),
        })

    # Highest confidence first; break ties by file/line for stable output.
    candidates.sort(key=lambda c: (-c["confidence"], c["file_path"], c["line_start"]))

    limitations: list[str] = []
    blind = sorted(seen_extensions & _MEMBER_ACCESS_BLIND_EXTENSIONS)
    if blind:
        limitations.append(
            "Member-access calls (obj.method()) aren't tracked for: "
            + ", ".join(blind)
            + ". Methods in these files have lower confidence."
        )
    shared = sum(1 for c in name_share.values() if c > 1)
    if shared:
        limitations.append(
            f"{shared} symbol name(s) are reused across multiple definitions; "
            "the reference index can't tell which definition a call resolves to."
        )
    limitations.append(
        "Dynamic dispatch, reflection, string-based imports, and "
        "framework-registered callbacks may produce false positives."
    )

    return {
        "candidates": candidates[:top_k],
        "scanned_symbols": scanned,
        "total_symbols": total_symbols,
        "limitations": limitations,
    }
