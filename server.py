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
def search_history(
    query: str,
    search_type: Literal["commits", "file_history", "blame", "commit_detail"] = "commits",
    target_file: str | None = None,
    line_start: int | None = None,
    line_end: int | None = None,
) -> dict:
    """Search local Git history to debug regressions, understand developer
    intent, or find out WHY a specific change was made.

    **search_type options:**

    - ``commits`` — Search commit messages for *query* (case-insensitive).
      Optionally filter to commits that touched *target_file*.
    - ``file_history`` — Show the commit log for *target_file* (follows
      renames).  *target_file* is required; *query* is ignored.
    - ``blame`` — Run ``git blame`` on *target_file*, optionally limited to
      *line_start*–*line_end*.  *target_file* is required.
    - ``commit_detail`` — Get full metadata and diff for one commit.
      Pass the commit hash as *query*.  Optionally set *target_file* to
      restrict the diff to that file.
    """
    import git_search as gs
    from git.exc import InvalidGitRepositoryError, NoSuchPathError

    try:
        repo = gs.get_repo(".")
    except (InvalidGitRepositoryError, NoSuchPathError) as exc:
        return {"error": f"Git repository not found: {exc}"}

    if search_type == "commits":
        results = gs.search_commits(repo, query, target_file)
        return {"search_type": "commits", "query": query, "results": results}

    elif search_type == "file_history":
        if not target_file:
            return {"error": "target_file is required for file_history search"}
        results = gs.get_file_history(repo, target_file)
        return {"search_type": "file_history", "target_file": target_file, "results": results}

    elif search_type == "blame":
        if not target_file:
            return {"error": "target_file is required for blame search"}
        results = gs.get_blame(repo, target_file, line_start, line_end)
        return {"search_type": "blame", "target_file": target_file, "results": results}

    elif search_type == "commit_detail":
        result = gs.get_commit_detail(repo, query, target_file)
        return {"search_type": "commit_detail", "result": result}

    return {"error": f"Unknown search_type: {search_type}"}


# ── Entrypoint ────────────────────────────────────────────────────────
if __name__ == "__main__":
    mcp.run()
