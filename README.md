# code-memory

A deterministic, high-precision **code intelligence layer** exposed as a [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server.

`code-memory` gives your AI coding assistant structured access to your codebase through three focused pathways — eliminating context-window bloat and vague "search everything" queries.

## Architecture: Progressive Disclosure

Instead of a single monolithic search, `code-memory` routes queries through **three purpose-built tools**:

| Question Type | Tool | Data Source |
|---|---|---|
| **"Where / What / How?"** — find definitions, references, structure, semantic search | `search_code` | BM25 + Dense Vector (SQLite vec) |
| **"Architecture / Patterns"** — understand architecture, explain workflows | `search_docs` | Semantic / Fuzzy |
| **"Who / Why?"** — debug regressions, understand intent | `search_history` | Git + BM25 + Dense Vector (SQLite vec) |
| **"Setup / Prepare"** — index parsing & embedding generation | `index_codebase` | AST Parser + `sentence-transformers` |

This forces the LLM to pick the *right retrieval strategy* before any data is fetched.

## Quickstart

### Prerequisites

- Python ≥ 3.13
- [`uv`](https://docs.astral.sh/uv/) package manager

### Install & Run

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/code-memory.git
cd code-memory

# Install dependencies
uv sync

# Run the MCP server (stdio transport)
uv run mcp run server.py
```

### Development

```bash
# Run with the MCP Inspector for interactive debugging
uv run mcp dev server.py

# Format / Lint
make lint

# Run tests
make test
```

## Configure Your MCP Host

### Gemini CLI / Gemini Code Assist

Add to your MCP settings (e.g. `~/.gemini/settings.json`):

```json
{
  "mcpServers": {
    "code-memory": {
      "command": "uvx",
      "args": ["code-memory"]
    }
  }
}
```

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "code-memory": {
      "command": "uvx",
      "args": ["code-memory"]
    }
  }
}
```

### VS Code (Copilot / Continue)

Add to `.vscode/mcp.json` in your workspace:

```json
{
  "servers": {
    "code-memory": {
      "command": "uvx",
      "args": ["code-memory"]
    }
  }
}
```

## Tools

### `index_codebase`

Indexes or re-indexes Python source files in the given directory. Run this before using `search_code` to ensure the database is up to date. Uses Python's `ast` module for structural extraction and generates dense vector embeddings using `sentence-transformers` (runs locally, in-process) for semantic search. All data is stored locally using `sqlite-vec`.

```
index_codebase(directory=".")
```

### `search_code`

Perform semantic search and find structural code definitions, locate where functions/classes are defined, or map out dependency references (call graphs). Uses hybrid retrieval (BM25 + vector embeddings) to find exact matches and semantic similarities.

```
search_code(query="parse python files", search_type="definition")
search_code(query="how do we establish the database connection", search_type="references")
search_code(query="src/auth/", search_type="file_structure")
```

### `search_docs`

Understand the codebase conceptually — how things work, architectural patterns, SOPs.

```
search_docs(query="how does the authentication flow work?")
```

### `search_history`

Debug regressions and understand developer intent through Git history.

```
search_history(query="fix login timeout")
search_history(query="jane.doe", target_file="src/auth/login.py")
```

## Project Structure

```
code-memory/
├── server.py          # MCP server entry point (FastMCP)
├── pyproject.toml     # Project metadata & dependencies
├── Makefile           # Dev workflow shortcuts
└── prompts/           # Milestone prompt engineering files
    ├── milestone_1.xml
    └── milestone_2.xml
```

## Roadmap

- [x] **Milestone 1** — Project scaffolding & MCP protocol wiring
- [ ] **Milestone 2** — Implement `search_code` with AST parsing + SQLite + `sqlite-vec`
- [ ] **Milestone 3** — Implement `search_history` with Git integration
- [ ] **Milestone 4** — Implement `search_docs` with semantic search
- [ ] **Milestone 5** — Production hardening & packaging

## License

MIT
