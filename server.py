"""
code-memory MCP Server

A deterministic, high-precision code intelligence layer exposed via the
Model Context Protocol (MCP).  Uses a "Progressive Disclosure" routing
architecture:

    1. "Who/Why?" → search_history  (Git data)
    2. "Where/What?" → search_code  (AST data)
    3. "How?" → search_docs         (Semantic / Fuzzy logic)
"""

from typing import Literal

from mcp.server.fastmcp import FastMCP

# ── Initialise the FastMCP server ──────────────────────────────────────
mcp = FastMCP("code-memory")


# ── Tool 1: search_code ───────────────────────────────────────────────
@mcp.tool()
def search_code(
    query: str,
    search_type: Literal["definition", "references", "file_structure"],
) -> dict:
    """Use this tool to find exact structural code definitions, locate where
    functions/classes are defined, or map out dependency references (call
    graphs). Do NOT use this for conceptual questions."""

    return {
        "status": "mocked",
        "tool": "search_code",
        "query": query,
        "search_type": search_type,
    }


# ── Tool 2: search_docs ──────────────────────────────────────────────
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


# ── Tool 3: search_history ───────────────────────────────────────────
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
