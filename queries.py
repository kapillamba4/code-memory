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
        A list of result dicts sorted by descending RRF score, including
        match_reason, match_highlights, and confidence.
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

    # Build results with match metadata
    results = []
    for sid, score in ranked:
        sources = match_sources.get(sid, [])
        is_hybrid = len(sources) == 2

        # Determine match reason
        if is_hybrid:
            match_reason = "hybrid (BM25 + semantic)"
        elif "bm25" in sources:
            match_reason = "keyword match (BM25)"
        else:
            match_reason = "semantic match (vector)"

        # Calculate confidence (normalize RRF score to 0-1 range)
        # Max possible RRF score for a single source is 1/61 ≈ 0.0164
        # For hybrid it's 2/61 ≈ 0.0328. We normalize accordingly.
        max_single_rrf = 1.0 / (_RRF_K + 1)  # ≈ 0.0164
        max_hybrid_rrf = 2.0 * max_single_rrf  # ≈ 0.0328
        if is_hybrid:
            confidence = min(1.0, score / max_hybrid_rrf)
        else:
            confidence = min(1.0, (score / max_single_rrf) * 0.7)  # Cap single-source at 0.7

        result = {
            **details[sid],
            "score": round(score, 6),
            "match_reason": match_reason,
            "confidence": round(confidence, 3),
            "match_highlights": [],  # Will be populated below if BM25 match
        }

        # Get highlights for BM25 matches using FTS5 highlight function
        if "bm25" in sources:
            highlights = _get_bm25_highlights(query, details[sid]["source_text"], db)
            result["match_highlights"] = highlights

        results.append(result)

    return results


def _get_bm25_highlights(query: str, source_text: str, db) -> list[str]:
    """Extract highlighted snippets using FTS5.

    Returns up to 3 highlighted text snippets showing where the query matched.
    """
    if not source_text or not query:
        return []

    # Use FTS5 highlight function to get matched portions
    safe_query = query.replace('"', '""')
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


def find_definition(symbol_name: str, db, include_context: bool = True) -> list[dict]:
    """Find where *symbol_name* is defined using hybrid search.

    Post-filters for exact name matches first; falls back to top hybrid
    results as "best guesses" if no exact match is found.

    Args:
        symbol_name: The name of the symbol to find.
        db: An open ``sqlite3.Connection``.
        include_context: If True, include docstrings and parent symbol info.

    Returns:
        A list of result dicts with enriched information.
    """
    results = hybrid_search(symbol_name, db, top_k=20)

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
    for r in rows:
        ref = {
            "symbol_name": r[0],
            "file_path": r[1],
            "line_number": r[2],
            "source_line": None,
            "containing_symbol": None,
        }

        # Get the source line at this reference
        try:
            with open(r[1]) as f:
                lines = f.readlines()
                if 0 < r[2] <= len(lines):
                    ref["source_line"] = lines[r[2] - 1].strip()
        except Exception:
            pass

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


def discover_topic(topic_query: str, db, top_k: int = 15, include_snippets: bool = True) -> list[dict]:
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
    code_results = hybrid_search(topic_query, db, top_k=50)

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
        file_aggregates[fp]["relevance_score"] += r.get("score", 0.5)
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
