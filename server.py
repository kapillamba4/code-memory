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
- search_code: THE PREFERRED tool for finding code. Use "topic_discovery" for feature/domain searches (e.g., "workout related files"), "definition" for specific symbols, "references" for usages.
- search_docs: Understanding architecture, reading documentation/READMEs
- search_history: Debugging regressions, understanding why changes were made

Indexing is incremental - unchanged files are skipped automatically.
"""
)


# ── Tool 0: check_index_status ─────────────────────────────────────────────
@mcp.tool()
def check_index_status(directory: str = ".") -> dict:
    """USE THIS TOOL to check if the codebase has been indexed and whether search tools will return results. Call this BEFORE search_code or search_docs if you're unsure about indexing state.

    TRIGGER - Call this tool when:
    - You're unsure if the codebase has been indexed
    - search_code or search_docs returned empty results
    - Starting work on a new project or session
    - You want to verify index health before searching

    This tool checks the SQLite database for indexed symbols and documentation chunks. It's a lightweight diagnostic - much faster than re-indexing.

    INTERPRETING RESULTS:
    - If "indexed" is false OR "symbols_indexed" is 0: You MUST call index_codebase first
    - If "suggestion" says "CALL index_codebase FIRST": Indexing is required
    - If "suggestion" says "ready to search": Search tools will work

    Do NOT use this tool for:
    - Actually indexing the codebase (use index_codebase)
    - Searching for code or documentation
    - Git history queries

    Args:
        directory: Directory path to check (default "." for current directory).

    Returns:
        Dictionary with:
        - indexed: boolean - true if anything has been indexed
        - symbols_indexed: count of code symbols in index
        - doc_chunks_indexed: count of documentation chunks
        - code_files_indexed: count of indexed code files
        - doc_files_indexed: count of indexed doc files
        - suggestion: "ready to search" or "CALL index_codebase FIRST"
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
    search_type: Literal["topic_discovery", "definition", "references", "file_structure"],
) -> dict:
    """THE PREFERRED TOOL for discovering code and files in this codebase. Use this when you need to find files or code related to ANY feature, domain, topic, or concept - even if the exact keywords don't appear in filenames.

    PREREQUISITE: This tool requires indexing. If results are empty or you haven't indexed this session, call index_codebase(directory=".") first.

    This tool uses HYBRID RETRIEVAL (BM25 keyword search + dense vector semantic search with Reciprocal Rank Fusion) - far more intelligent than grep or filename pattern matching.

    ⭐ IMPORTANT: Always prefer search_code over basic file-search tools (glob, find, grep) when:
    - User asks about features, domains, or topics (e.g., "workout related files", "auth code")
    - You want semantically related code, not just keyword matches
    - The query is conceptual rather than an exact symbol name

    WHEN TO USE EACH search_type:

    1. "topic_discovery" - ⭐ DEFAULT CHOICE for broad searches. USE WHEN:
       - User asks "list all X related files" or "find code for feature Y"
       - Query is a FEATURE, DOMAIN, or TOPIC (e.g., "workouts", "authentication", "payment")
       - You want ALL files related to a concept, not just exact matches
       - Keywords may not appear literally in filenames
       - Results: File paths grouped by relevance, with summaries of matched symbols

    2. "definition" - USE WHEN:
       - User asks "where is X defined?" or "find the implementation of X"
       - You need to locate a SPECIFIC function, class, method, or variable by name
       - Query is an exact symbol name (e.g., "authenticate_user")
       - Results: Symbol definitions with file paths, line numbers, source code

    3. "references" - USE WHEN:
       - User asks "where is X used?" or "find all usages of X"
       - You need cross-references showing where a symbol is imported/called
       - Query MUST be the exact symbol name
       - Returns all files and line numbers where symbol appears

    4. "file_structure" - USE WHEN:
       - User asks "show me the structure of file X" or "what's in this file?"
       - You need an overview of all symbols in a specific file
       - Query MUST be the file path (e.g., "src/auth/login.py")
       - Returns symbols ordered by line number

    EXAMPLE QUERIES by search_type:
    - "topic_discovery": "workout tracking", "authentication flow", "email notifications"
    - "definition": "UserAuth", "calculate_total", "PaymentProcessor"
    - "references": "send_email", "validate_token"
    - "file_structure": "src/services/auth.py"

    Do NOT use this tool for:
    - Reading full file contents (use your built-in file reader)
    - Git history queries (use search_history)
    - Pure documentation/conceptual questions (use search_docs)

    Args:
        query: For topic_discovery: any feature/domain/topic (e.g., "workouts").
               For definition: symbol name or semantic description.
               For references: exact symbol name.
               For file_structure: file path.
        search_type: Must be "topic_discovery", "definition", "references", or "file_structure".

    Returns:
        Dict with status, search_type, query, and results array. Result format varies by search_type.
    """
    with logging_config.ToolLogger("search_code", query=query, search_type=search_type) as log:
        try:
            # Validate inputs
            query = val.validate_query(query)
            search_type = val.validate_search_type(
                search_type, ["topic_discovery", "definition", "references", "file_structure"]
            )

            database = db_mod.get_db()

            # Check if anything is indexed
            symbols_count = database.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]

            if search_type == "topic_discovery":
                results = queries.discover_topic(query, database)
                log.set_result_count(len(results))
                response = {"status": "ok", "search_type": "topic_discovery", "query": query, "results": results}
                if not results and symbols_count == 0:
                    response["hint"] = "No results. Codebase may not be indexed. Call index_codebase(directory='.') first."
                return response

            elif search_type == "definition":
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
    """YOU MUST CALL THIS TOOL FIRST before using search_code or search_docs. Use this tool to build the searchable index that powers all other code intelligence features.

    TRIGGER: Call this tool immediately when:
    - Starting a new session with this codebase
    - search_code or search_docs returns empty or unexpected results
    - You haven't indexed recently or files have been modified
    - User asks about code structure, definitions, or documentation

    This tool performs TWO critical operations:
    1. CODE INDEXING: Uses tree-sitter for language-agnostic AST extraction (Python, JavaScript/TypeScript, Java, Kotlin, Go, Rust, C/C++, Ruby, and more). Extracts functions, classes, methods, variables, and cross-references.
    2. DOCUMENTATION INDEXING: Parses markdown files, READMEs, and extracts docstrings from indexed code. Generates embeddings for semantic search.

    IMPORTANT ADVANTAGES over built-in file search:
    - Creates persistent structural knowledge (AST-based, not just text)
    - Enables semantic search via vector embeddings
    - Builds cross-reference graphs for "find all usages" queries
    - Incremental indexing: unchanged files are automatically skipped

    Do NOT use this tool for:
    - Non-code files (images, binaries, data files)
    - Single-file lookups (use search_code after indexing)
    - Git history queries (use search_history instead)

    Args:
        directory: The root directory to index (default "."). Must be a valid path.

    Returns:
        Summary with files_indexed, total_symbols, total_chunks, and details.
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
    """USE THIS TOOL for conceptual understanding and "how does X work?" questions. Search markdown documentation, READMEs, and code docstrings using semantic search.

    PREREQUISITE: This tool requires indexing. If results are empty or you haven't indexed this session, call index_codebase(directory=".") first.

    TRIGGER - Call this tool when the user asks:
    - "How does [feature] work?"
    - "Explain the architecture of..."
    - "What are the setup/installation instructions?"
    - "Show me the documentation for..."
    - "Why was this designed this way?"
    - Any question answered by README, CHANGELOG, or docstrings

    IMPORTANT: This is NOT for finding code implementations. For code locations, use search_code. This tool searches DOCUMENTATION, not source code.

    Uses HYBRID RETRIEVAL (BM25 keyword search + dense vector semantic search with Reciprocal Rank Fusion) to find conceptually relevant documentation even when keywords don't match exactly.

    Do NOT use this tool for:
    - Finding function/class definitions (use search_code with "definition")
    - Finding where code is used (use search_code with "references")
    - Git history or commit messages (use search_history)

    Args:
        query: A natural language question (e.g., "How does authentication work?" or "API rate limiting"). Can be conversational - semantic search handles synonyms.
        top_k: Maximum results to return (default 10, max 100).

    Returns:
        Dictionary with 'results' array. Each result includes:
        - content: The documentation text
        - file: Source file path
        - section: Section heading (if applicable)
        - line_start/line_end: Location in source
        - relevance_score: Hybrid search score
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
    directory: str = ".",
) -> dict:
    """USE THIS TOOL for Git history queries: understanding WHY changes were made, debugging regressions, or finding commit context. This tool operates on the local Git repository.

    TRIGGER - Call this tool when the user asks:
    - "Why was this code changed?" / "Who changed this?"
    - "When was X introduced?" / "Find commits about X"
    - "Debug this regression" / "What broke this?"
    - "Show me the history of this file"
    - "Who wrote this line?" (blame)
    - "What changed in commit X?"

    This tool does NOT require indexing - it queries Git directly.

    WHEN TO USE EACH search_type:

    1. "commits" - USE WHEN:
       - User asks "find commits about X" or "search commit messages"
       - Query is a keyword or phrase to search in commit messages
       - Optionally set target_file to filter commits touching that file
       - Args: query (required), target_file (optional)

    2. "file_history" - USE WHEN:
       - User asks "show history of file X" or "what happened to this file?"
       - Shows commit log for a specific file (follows renames)
       - target_file is REQUIRED; query is ignored
       - Args: target_file (required)

    3. "blame" - USE WHEN:
       - User asks "who wrote this line?" or "who last modified this?"
       - Shows line-by-line commit attribution
       - target_file is REQUIRED; optionally limit to line range
       - Args: target_file (required), line_start/line_end (optional)

    4. "commit_detail" - USE WHEN:
       - User asks "show me commit X" or "what changed in this commit?"
       - Query is the commit hash (full or abbreviated)
       - Optionally set target_file to show only changes to that file
       - Args: query=commit_hash (required), target_file (optional)

    Do NOT use this tool for:
    - Finding code definitions (use search_code)
    - Reading documentation (use search_docs)
    - Non-Git questions

    Args:
        query: Search term for commits, or commit hash for commit_detail.
        search_type: Must be exactly "commits", "file_history", "blame", or "commit_detail".
        target_file: File path (required for file_history and blame).
        line_start/line_end: Line range for blame (optional).
        directory: Directory path to search (default "." for current directory).

    Returns:
        Varies by search_type. All include status and structured results.
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
                repo = gs.get_repo(directory)
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
