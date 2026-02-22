# Contributing to code-memory

Thank you for your interest in contributing to code-memory! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/) package manager

### Clone and Install

```bash
# Clone the repository
git clone https://github.com/kapillamba4/code-memory.git
cd code-memory

# Install dependencies (including dev dependencies)
uv sync --all-extras
```

### Run the Server

```bash
# Run the MCP server
uv run mcp run server.py

# Run with MCP Inspector for debugging
uv run mcp dev server.py
```

## Development Workflow

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ -v --cov --cov-report=term-missing
```

### Linting and Formatting

```bash
# Run ruff linter
uv run ruff check .

# Run ruff formatter
uv run ruff format .

# Run type checking
uv run mypy .
```

### Building

```bash
# Build the package
uv build
```

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions
- Maximum line length is 100 characters
- Use type hints for function parameters and return types
- Write docstrings for public functions and classes

## Project Structure

```
code-memory/
├── server.py          # MCP server entry point
├── db.py              # SQLite database layer
├── parser.py          # Tree-sitter code parser
├── doc_parser.py      # Markdown documentation parser
├── queries.py         # Hybrid retrieval query layer
├── git_search.py      # Git history search module
├── errors.py          # Custom exception hierarchy
├── validation.py      # Input validation functions
├── logging_config.py  # Structured logging configuration
├── tests/             # Test suite
├── prompts/           # Milestone prompt files
└── pyproject.toml     # Project configuration
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting (`uv run pytest && uv run ruff check .`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### PR Guidelines

- Include a clear description of the changes
- Reference any related issues
- Ensure all tests pass
- Add tests for new functionality
- Update documentation if needed

## Adding New Features

### Adding a New Tool

1. Add the tool function in `server.py` with the `@mcp.tool()` decorator
2. Add input validation using functions from `validation.py`
3. Wrap the implementation in error handling
4. Add logging using `ToolLogger`
5. Write tests in `tests/test_tools.py`

### Adding a New Language Parser

1. Add the tree-sitter language package to `pyproject.toml`
2. Update the language registry in `parser.py`
3. Add node-type mappings for the new language
4. Write tests for the new language

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CODE_MEMORY_LOG_LEVEL` | Logging verbosity (DEBUG, INFO, WARNING, ERROR) | INFO |

## License

By contributing to code-memory, you agree that your contributions will be licensed under the MIT License.
