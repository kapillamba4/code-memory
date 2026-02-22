"""
code-memory MCP Server

A deterministic, high-precision code intelligence layer exposed via the
Model Context Protocol (MCP).  Uses a "Progressive Disclosure" routing
architecture:

    1. "Who/Why?" → search_history  (Git data)
    2. "Where/What?" → search_code  (AST data + hybrid retrieval)
    3. "How?" → search_docs         (Semantic / Fuzzy logic)
"""

from __future__ import annotations

from typing import Literal

from mcp.server.fastmcp import FastMCP

import db as db_mod
import doc_parser as doc_parser_mod
import errors
import logging_config
import parser as parser_mod
import queries
import validation as val

# ── Initialize logging ───────────────────────────────────────────────────
logger = logging_config.setup_logging()
tool_logger = logging_config.get_logger("tools")

# ── Initialize the FastMCP server ────────────────────────────────────────
mcp = FastMCP(
    "code-memory",
    instructions="""
CRITICAL WORKFLOW: You MUST call `index_codebase` BEFORE using any search tools.

The search tools (search_code, search_docs, search_history) will return empty results
if the codebase has not been indexed. Always check if indexing is needed:

1. FIRST: Call `index_codebase(directory=".")` to index the current project
2. THEN: Use search_code, search_docs, or search_history to find information
3. RE-INDEX: If you modify files or haven't indexed recently, run index_codebase again

When to use each tool:
- search_code: Finding function/class definitions, references, file structure
- search_docs: Understanding architecture, reading documentation/READMEs
- search_history: Debugging regressions, understanding why changes were made

Indexing is incremental - unchanged files are skipped automatically.
"""
)


# ── Tool 0: check_index_status ─────────────────────────────────────────────
@mcp.tool()
def check_index_status(directory: str = ".") -> dict:
    """Check if the codebase has been indexed and get indexing statistics.

    Call this BEFORE using search tools to determine if indexing is needed.
    If symbols_indexed is 0, you MUST call index_codebase first.

    Args:
        directory: The directory to check (default: current directory).

    Returns:
        Dictionary with indexing status including:
        - indexed: boolean indicating if code has been indexed
        - symbols_indexed: number of code symbols in the index
        - doc_chunks_indexed: number of documentation chunks
        - suggestion: "index_codebase" if indexing is needed, "ready" otherwise
    """
    try:
        database = db_mod.get_db()

        # Count symbols
        symbols_count = database.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]

        # Count doc chunks
        doc_chunks_count = database.execute("SELECT COUNT(*) FROM doc_chunks").fetchone()[0]

        # Count files
        files_count = database.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        doc_files_count = database.execute("SELECT COUNT(*) FROM doc_files").fetchone()[0]

        indexed = symbols_count > 0 or doc_chunks_count > 0

        return {
            "indexed": indexed,
            "symbols_indexed": symbols_count,
            "doc_chunks_indexed": doc_chunks_count,
            "code_files_indexed": files_count,
            "doc_files_indexed": doc_files_count,
            "suggestion": "ready to search" if indexed else "CALL index_codebase FIRST",
        }

    except Exception as e:
        return {
            "indexed": False,
            "error": str(e),
            "suggestion": "CALL index_codebase FIRST",
        }


# ── Tool 1: search_code ───────────────────────────────────────────────────
@mcp.tool()
def search_code(
    query: str,
    search_type: Literal["definition", "references", "file_structure"],
) -> dict:
    """Search the indexed codebase for definitions, references, or file structure.

    IMPORTANT: This tool requires indexing first. If you get empty results or
    haven't indexed recently, call `index_codebase(directory=".")` before searching.

    Uses hybrid retrieval (BM25 keyword search + dense vector semantic search)
    with Reciprocal Rank Fusion.

    Args:
        query: Symbol name or semantic description of what you're looking for.
        search_type:
            - "definition": Find where a symbol is defined (hybrid search)
            - "references": Find all cross-references to a symbol name
            - "file_structure": List all symbols in a file, ordered by line

    Returns:
        List of matching results with file paths, line numbers, and source text.
    """
    with logging_config.ToolLogger("search_code", query=query, search_type=search_type) as log:
        try:
            # Validate inputs
            query = val.validate_query(query)
            search_type = val.validate_search_type(
                search_type, ["definition", "references", "file_structure"]
            )

            database = db_mod.get_db()

            # Check if anything is indexed
            symbols_count = database.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]

            if search_type == "definition":
                results = queries.find_definition(query, database)
                log.set_result_count(len(results))
                response = {"status": "ok", "search_type": "definition", "query": query, "results": results}
                if not results and symbols_count == 0:
                    response["hint"] = "No results. Codebase may not be indexed. Call index_codebase(directory='.') first."
                return response

            elif search_type == "references":
                results = queries.find_references(query, database)
                log.set_result_count(len(results))
                response = {"status": "ok", "search_type": "references", "query": query, "results": results}
                if not results and symbols_count == 0:
                    response["hint"] = "No results. Codebase may not be indexed. Call index_codebase(directory='.') first."
                return response

            elif search_type == "file_structure":
                results = queries.get_file_structure(query, database)
                log.set_result_count(len(results))
                response = {"status": "ok", "search_type": "file_structure", "query": query, "results": results}
                if not results and symbols_count == 0:
                    response["hint"] = "No results. Codebase may not be indexed. Call index_codebase(directory='.') first."
                return response

            return errors.format_error(errors.ValidationError(f"Unknown search_type: {search_type}"))

        except errors.CodeMemoryError as e:
            return e.to_dict()
        except Exception as e:
            return errors.format_error(e)


# ── Tool 2: index_codebase ────────────────────────────────────────────────
@mcp.tool()
def index_codebase(directory: str = ".") -> dict:
    """Indexes or re-indexes source files and documentation in the given directory.

    Run this before using search_code or search_docs to ensure the database
    is up to date. Uses tree-sitter for language-agnostic structural extraction
    and generates embeddings for semantic search. Supports Python, JavaScript/
    TypeScript, Java, Kotlin, Go, Rust, C/C++, Ruby, and more.

    Also indexes markdown documentation files and extracts docstrings from
    indexed code symbols. Unchanged files (by mtime) are automatically skipped.

    Args:
        directory: The root directory to index (recursively).

    Returns:
        Summary of indexing results including code and documentation stats.
    """
    with logging_config.ToolLogger("index_codebase", directory=directory) as log:
        try:
            # Validate directory
            directory_path = val.validate_directory(directory)

            database = db_mod.get_db()

            # Index code files
            code_logger = logging_config.IndexingLogger("code")
            code_logger.start(str(directory_path))

            code_results = parser_mod.index_directory(str(directory_path), database)
            for r in code_results:
                if r.get("skipped"):
                    code_logger.file_skipped(r.get("file", "unknown"), r.get("reason", "unknown"))
                else:
                    code_logger.file_indexed(r.get("file", "unknown"), r.get("symbols_indexed", 0))
            code_logger.complete()

            indexed = [r for r in code_results if not r.get("skipped")]
            skipped = [r for r in code_results if r.get("skipped")]

            # Index documentation files
            doc_logger = logging_config.IndexingLogger("documentation")
            doc_logger.start(str(directory_path))

            doc_results = doc_parser_mod.index_doc_directory(str(directory_path), database)
            for r in doc_results:
                if r.get("skipped"):
                    doc_logger.file_skipped(r.get("file", "unknown"), r.get("reason", "unknown"))
                else:
                    doc_logger.file_indexed(r.get("file", "unknown"), r.get("chunks_indexed", 0))
            doc_logger.complete()

            doc_indexed = [r for r in doc_results if not r.get("skipped")]
            doc_skipped = [r for r in doc_results if r.get("skipped")]

            # Extract docstrings from indexed code
            docstring_results = doc_parser_mod.extract_docstrings_from_code(database)

            total_symbols = sum(r.get("symbols_indexed", 0) for r in indexed)
            total_chunks = sum(r.get("chunks_indexed", 0) for r in doc_indexed)
            log.set_result_count(total_symbols + total_chunks + len(docstring_results))

            return {
                "status": "ok",
                "directory": str(directory_path),
                "code": {
                    "files_indexed": len(indexed),
                    "files_skipped": len(skipped),
                    "total_symbols": total_symbols,
                    "total_references": sum(r.get("references_indexed", 0) for r in indexed),
                },
                "documentation": {
                    "files_indexed": len(doc_indexed),
                    "files_skipped": len(doc_skipped),
                    "total_chunks": total_chunks,
                    "docstrings_extracted": len(docstring_results),
                },
                "details": {
                    "code": indexed,
                    "docs": doc_indexed,
                },
            }

        except errors.CodeMemoryError as e:
            return e.to_dict()
        except Exception as e:
            return errors.format_error(e)


# ── Tool 3: search_docs ────────────────────────────────────────────────────
@mcp.tool()
def search_docs(query: str, top_k: int = 10) -> dict:
    """Search documentation to understand the codebase conceptually.

    IMPORTANT: This tool requires indexing first. If you get empty results or
    haven't indexed recently, call `index_codebase(directory=".")` before searching.

    Ideal for questions like:
    - "How does authentication work?"
    - "Explain the architecture"
    - "What are the setup instructions?"

    Searches markdown files, READMEs, and docstrings using hybrid retrieval
    (BM25 + semantic search with RRF fusion).

    Args:
        query: A natural language question about the codebase.
        top_k: Maximum number of results to return (default 10).

    Returns:
        Dictionary with 'results' containing matching documentation chunks
        with source attribution (file, section, line numbers) and relevance score.
    """
    with logging_config.ToolLogger("search_docs", query=query, top_k=top_k) as log:
        try:
            # Validate inputs
            query = val.validate_query(query)
            top_k = val.validate_top_k(top_k)

            database = db_mod.get_db()

            # Check if anything is indexed
            doc_chunks_count = database.execute("SELECT COUNT(*) FROM doc_chunks").fetchone()[0]

            results = queries.search_documentation(query, database, top_k=top_k)
            log.set_result_count(len(results))

            response = {
                "status": "ok",
                "query": query,
                "results": results,
                "count": len(results),
            }

            if not results and doc_chunks_count == 0:
                response["hint"] = "No results. Documentation may not be indexed. Call index_codebase(directory='.') first."

            return response

        except errors.CodeMemoryError as e:
            return e.to_dict()
        except Exception as e:
            return errors.format_error(e)


# ── Tool 4: search_history ─────────────────────────────────────────────────
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
    with logging_config.ToolLogger("search_history", query=query, search_type=search_type,
                                   target_file=target_file) as log:
        try:
            import git_search as gs
            from git.exc import InvalidGitRepositoryError, NoSuchPathError

            # Validate inputs
            search_type = val.validate_search_type(
                search_type, ["commits", "file_history", "blame", "commit_detail"]
            )
            line_start, line_end = val.validate_line_range(line_start, line_end)

            # Get git repository
            try:
                repo = gs.get_repo(".")
            except (InvalidGitRepositoryError, NoSuchPathError) as exc:
                raise errors.GitError(f"Git repository not found: {exc}")

            if search_type == "commits":
                query = val.validate_query(query, min_length=1)
                results = gs.search_commits(repo, query, target_file)
                log.set_result_count(len(results))
                return {"status": "ok", "search_type": "commits", "query": query, "results": results}

            elif search_type == "file_history":
                if not target_file:
                    raise errors.ValidationError("target_file is required for file_history search")
                results = gs.get_file_history(repo, target_file)
                log.set_result_count(len(results))
                return {"status": "ok", "search_type": "file_history", "target_file": target_file, "results": results}

            elif search_type == "blame":
                if not target_file:
                    raise errors.ValidationError("target_file is required for blame search")
                results = gs.get_blame(repo, target_file, line_start, line_end)
                log.set_result_count(len(results))
                return {"status": "ok", "search_type": "blame", "target_file": target_file, "results": results}

            elif search_type == "commit_detail":
                result = gs.get_commit_detail(repo, query, target_file)
                return {"status": "ok", "search_type": "commit_detail", "result": result}

            return errors.format_error(errors.ValidationError(f"Unknown search_type: {search_type}"))

        except errors.CodeMemoryError as e:
            return e.to_dict()
        except Exception as e:
            return errors.format_error(e)


# ── Entrypoint ────────────────────────────────────────────────────────────
def main():
    """Entry point for the MCP server when installed as a package."""
    mcp.run()


if __name__ == "__main__":
    main()
