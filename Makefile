.PHONY: run dev inspect lint format test clean help

# ── Server ─────────────────────────────────────────────────────────────
run: ## Run the MCP server (stdio transport)
	uv run mcp run server.py

dev: ## Run with MCP Inspector for interactive debugging
	uv run mcp dev server.py

# ── Code Quality ───────────────────────────────────────────────────────
lint: ## Run ruff linter
	uv run ruff check .

format: ## Auto-format with ruff
	uv run ruff format .

# ── Testing ────────────────────────────────────────────────────────────
test: ## Run test suite
	uv run pytest -v

# ── Housekeeping ───────────────────────────────────────────────────────
clean: ## Remove caches and build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/

# ── Help ───────────────────────────────────────────────────────────────
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
