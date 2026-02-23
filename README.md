# code-memory

<img src="assets/logo.png" alt="code-memory logo" width="128" height="128">

A deterministic, high-precision **code intelligence layer** exposed as a [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server.

- **No API key required** — runs entirely locally with sentence-transformers
- **1 min setup** — just `uvx code-memory` and you're ready
- **Token saving by 50%** — precise code retrieval instead of dumping entire files

**Please help star code-memory if you like this project!**

## Why code-memory?

Finding the right context from a large codebase is **expensive**, **inaccurate**, and **limited by context windows**. Dumping files into prompts wastes tokens, and LLMs lose track of the actual task as context fills up.

Instead of manually hunting with `grep`/`find` or dumping raw file text, `code-memory` runs semantic searches against a locally indexed codebase. Inspired by [claude-context](https://github.com/redmonkez12/claude-context), but designed from the ground up for large-scale local search.

## Supported Languages

### Full AST Support (Tree-sitter)

These languages have structural parsing with symbol extraction (functions, classes, methods, etc.):

| Language | Extensions |
|----------|------------|
| Python | `.py` |
| JavaScript | `.js`, `.jsx` |
| TypeScript | `.ts`, `.tsx` |
| Java | `.java` |
| Go | `.go` |
| Rust | `.rs` |
| C | `.c`, `.h` |
| C++ | `.cpp`, `.hpp`, `.cc`, `.cxx` |
| Ruby | `.rb` |
| Kotlin | `.kt`, `.kts` |

### Fallback Support (Whole-file Indexing)

These file types are indexed as complete units for BM25 and semantic search:

| Category | Extensions |
|----------|------------|
| C# | `.cs` |
| Swift | `.swift` |
| Scala | `.scala` |
| Lua | `.lua` |
| Shell | `.sh`, `.bash`, `.zsh` |
| Config | `.yaml`, `.yml`, `.toml`, `.json` |
| Web | `.html`, `.css`, `.scss` |
| Database | `.sql` |
| Docs | `.md`, `.txt` |

> **Note:** Files and directories matching patterns in your `.gitignore` are automatically skipped during indexing. This excludes build artifacts, dependencies, and other generated files.

## Architecture: Progressive Disclosure

Instead of a single monolithic search, `code-memory` routes queries through **three purpose-built tools**:

| Question Type | Tool | Data Source |
|---|---|---|
| **"Where / What / How?"** — find definitions, references, structure, semantic search | `search_code` | BM25 + Dense Vector (SQLite vec) |
| **"Architecture / Patterns"** — understand architecture, explain workflows | `search_docs` | Semantic / Fuzzy |
| **"Who / Why?"** — debug regressions, understand intent | `search_history` | Git + BM25 + Dense Vector (SQLite vec) |
| **"Setup / Prepare"** — index parsing & embedding generation | `index_codebase` | AST Parser + `sentence-transformers` |

This forces the LLM to pick the *right retrieval strategy* before any data is fetched.

## Installation

### From PyPI (Recommended)

```bash
# Install with pip
pip install code-memory

# Or with uvx (for MCP hosts)
uvx code-memory
```

### From Source

```bash
# Clone the repo
git clone https://github.com/kapillamba4/code-memory.git
cd code-memory

# Install dependencies
uv sync

# Run the MCP server (stdio transport)
uv run mcp run server.py
```

### Pre-built Binaries (Standalone)

Download standalone executables from [GitHub Releases](https://github.com/kapillamba4/code-memory/releases) — no Python installation required.

| Platform | Architecture | File |
|----------|-------------|------|
| Linux | x86_64 | `code-memory-linux-x86_64` |
| macOS | x86_64 (Intel) | `code-memory-macos-x86_64` |
| macOS | ARM64 (Apple Silicon) | `code-memory-macos-arm64` |
| Windows | x86_64 | `code-memory-windows-x86_64.exe` |

```bash
# Linux/macOS: Download and make executable
chmod +x code-memory-*
./code-memory-*

# Windows: Run directly
code-memory-windows-x86_64.exe
```

**Note:** The first run will download the embedding model (~600MB) to `~/.cache/huggingface/`. Subsequent runs use the cached model.

## Quickstart

### Prerequisites

- Python ≥ 3.13
- [`uv`](https://docs.astral.sh/uv/) package manager (recommended) or pip

### Install & Run

```bash
# Install from PyPI
pip install code-memory

# Or run directly with uvx
uvx code-memory
```

### Development

```bash
# Run with the MCP Inspector for interactive debugging
uv run mcp dev server.py

# Run tests
uv run pytest tests/ -v

# Lint and format
uv run ruff check .
uv run ruff format .

# Build package
uv build

# Build standalone binary (requires pyinstaller)
pip install pyinstaller
pyinstaller --clean code-memory.spec
# Binary output: dist/code-memory
```

## Configure Your MCP Host

You can use either `uvx` (requires Python) or the standalone binary (no dependencies).

### Using uvx (Python required)

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

### Claude Code (CLI)

Add to `.mcp.json` in your project root or `~/.mcp.json` for global access:

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

### Using Standalone Binary (No Python required)

Replace the path with the location of your downloaded binary:

```json
{
  "mcpServers": {
    "code-memory": {
      "command": "/path/to/code-memory-linux-x86_64"
    }
  }
}
```

For Windows:
```json
{
  "mcpServers": {
    "code-memory": {
      "command": "C:\\path\\to\\code-memory-windows-x86_64.exe"
    }
  }
}
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CODE_MEMORY_LOG_LEVEL` | Logging verbosity (DEBUG, INFO, WARNING, ERROR) | INFO |

Example:
```bash
CODE_MEMORY_LOG_LEVEL=DEBUG uvx code-memory
```

## Tools

### `index_codebase`

Indexes or re-indexes source files and documentation in the given directory. Run this before using `search_code` or `search_docs` to ensure the database is up to date. Uses tree-sitter for language-agnostic structural extraction and generates dense vector embeddings using `sentence-transformers` (runs locally, in-process) for semantic search.

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

Understand the codebase conceptually — how things work, architectural patterns, SOPs. Searches markdown documentation, READMEs, and docstrings extracted from code.

```
search_docs(query="how does the authentication flow work?")
search_docs(query="installation instructions", top_k=5)
```

### `search_history`

Debug regressions and understand developer intent through Git history.

```
search_history(query="fix login timeout", search_type="commits")
search_history(query="src/auth/login.py", search_type="file_history", target_file="src/auth/login.py")
search_history(query="server.py", search_type="blame", target_file="server.py", line_start=1, line_end=20)
```

## Project Structure

```
code-memory/
├── server.py          # MCP server entry point (FastMCP)
├── db.py              # SQLite database layer with sqlite-vec
├── parser.py          # Tree-sitter-based code parser
├── doc_parser.py      # Markdown documentation parser
├── queries.py         # Hybrid retrieval query layer
├── git_search.py      # Git history search module
├── errors.py          # Custom exception hierarchy
├── validation.py      # Input validation functions
├── logging_config.py  # Structured logging configuration
├── tests/             # Test suite
├── pyproject.toml     # Project metadata & dependencies
└── prompts/           # Milestone prompt engineering files
```

## Troubleshooting

### "Git repository not found" error

Make sure you're running `search_history` from within a git repository. The tool searches upward from the current directory to find `.git`.

### Empty search results

Run `index_codebase(directory=".")` first to index your code and documentation. The index is stored locally in `code_memory.db`.

### Slow indexing

Indexing generates embeddings using a local sentence-transformers model. The first run downloads the model (~600MB for `jina-code-embeddings-0.5b`). Subsequent runs are faster.

### Embedding model errors

Ensure you have enough disk space and memory. The `jina-code-embeddings-0.5b` model requires ~1GB RAM when loaded.

## Roadmap

- [x] **Milestone 1** — Project scaffolding & MCP protocol wiring
- [x] **Milestone 2** — Implement `search_code` with AST parsing + SQLite + `sqlite-vec`
- [x] **Milestone 3** — Implement `search_history` with Git integration
- [x] **Milestone 4** — Implement `search_docs` with semantic search
- [x] **Milestone 5** — Production hardening & packaging

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

## License

MIT
