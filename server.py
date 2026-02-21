"""
code-memory MCP Server

A deterministic, high-precision code intelligence layer exposed via the
Model Context Protocol (MCP).  Uses a "Progressive Disclosure" routing
architecture:

    1. "Who/Why?" → search_history  (Git data)
    2. "Where/What?" → search_code  (AST data + hybrid retrieval)
    3. "How?" → search_docs         (Semantic / Fuzzy logic)
"""

from typing import Literal

from mcp.server.fastmcp import FastMCP

import db as db_mod
import parser as parser_mod
import queries

# ── Initialise the FastMCP server ──────────────────────────────────────
mcp = FastMCP("code-memory")


# ── Tool 1: search_code ───────────────────────────────────────────────
@mcp.tool()
def search_code(
    query: str,
    search_type: Literal["definition", "references", "file_structure"],
) -> dict:
    """Search the indexed codebase for definitions, references, or file
    structure.

    Uses hybrid retrieval (BM25 keyword search + dense vector semantic
    search) with Reciprocal Rank Fusion for definition queries.

    - **definition**: Find where a symbol is defined (hybrid search).
    - **references**: Find all cross-references to a symbol name.
    - **file_structure**: List all symbols in a file, ordered by line.

    Run ``index_codebase`` first to populate the search index."""

    database = db_mod.get_db()

    if search_type == "definition":
        results = queries.find_definition(query, database)
        return {"search_type": "definition", "query": query, "results": results}

    elif search_type == "references":
        results = queries.find_references(query, database)
        return {"search_type": "references", "query": query, "results": results}

    elif search_type == "file_structure":
        results = queries.get_file_structure(query, database)
        return {"search_type": "file_structure", "query": query, "results": results}

    return {"error": f"Unknown search_type: {search_type}"}


# ── Tool 2: index_codebase ───────────────────────────────────────────
@mcp.tool()
def index_codebase(directory: str) -> dict:
    """Indexes or re-indexes source files in the given directory.

    Run this before using search_code to ensure the database is up to date.
    Uses tree-sitter for language-agnostic structural extraction and generates
    embeddings for semantic search.  Supports Python, JavaScript/TypeScript,
    Java, Kotlin, Go, Rust, C/C++, Ruby, and more.  Unsupported file types
    fall back to whole-file indexing.  Unchanged files (by mtime) are
    automatically skipped.

    Args:
        directory: The root directory to index (recursively).

    Returns:
        Summary of indexing results.
    """
    database = db_mod.get_db()
    results = parser_mod.index_directory(directory, database)

    indexed = [r for r in results if not r.get("skipped")]
    skipped = [r for r in results if r.get("skipped")]

    return {
        "status": "ok",
        "directory": directory,
        "files_indexed": len(indexed),
        "files_skipped": len(skipped),
        "total_symbols": sum(r.get("symbols_indexed", 0) for r in indexed),
        "total_references": sum(r.get("references_indexed", 0) for r in indexed),
        "details": indexed,
    }


# ── Tool 3: search_docs ──────────────────────────────────────────────
@mcp.tool()
def search_docs(query: str) -> dict:
    """Use this tool to understand the codebase conceptually. Ideal for
    'how does X work?', 'explain the architecture', or finding standard
    operating procedures in the documentation."""

    return {
        "status": "mocked",
        "tool": "search_docs",
        "query": query,
    }


# ── Tool 4: search_history ───────────────────────────────────────────
@mcp.tool()
def search_history(query: str, target_file: str | None = None) -> dict:
    """Use this tool to debug regressions, understand developer intent,
    or find out WHY a specific change was made by searching Git history
    and commit messages."""

    return {
        "status": "mocked",
        "tool": "search_history",
        "query": query,
        "target_file": target_file,
    }


# ── Entrypoint ────────────────────────────────────────────────────────
if __name__ == "__main__":
    mcp.run()
