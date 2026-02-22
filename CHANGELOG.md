# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-02-22

### Added
- Initial release
- `search_code` tool with hybrid retrieval (BM25 + vector + RRF fusion)
  - Find symbol definitions with semantic search
  - Find cross-references to symbols
  - Get file structure (all symbols in a file)
- `search_docs` tool for documentation search
  - Index markdown files and READMEs
  - Extract docstrings from code symbols
  - Hybrid search over documentation chunks
- `search_history` tool for Git history search
  - Search commit messages
  - Get file history with rename tracking
  - Run git blame with line range support
  - Get detailed commit information
- `index_codebase` tool for code and documentation indexing
  - Multi-language AST parsing with tree-sitter
  - Supports Python, JavaScript/TypeScript, Java, Kotlin, Go, Rust, C/C++, Ruby
  - Incremental indexing (skips unchanged files)
  - Generates embeddings for semantic search
- Production hardening
  - Structured error handling with custom exceptions
  - Input validation with clear error messages
  - Configurable logging via `CODE_MEMORY_LOG_LEVEL` environment variable
  - CI/CD with GitHub Actions
  - Test suite with pytest

### Dependencies
- mcp[cli] - Model Context Protocol server
- sqlite-vec - Vector search extension for SQLite
- sentence-transformers - Local embedding model (all-MiniLM-L6-v2)
- tree-sitter + language packages - Multi-language AST parsing
- gitpython - Git operations
- markdown-it-py - Markdown parsing
