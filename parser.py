"""
Language-agnostic AST parser and incremental indexer for code-memory.

Uses **tree-sitter** for multi-language structural parsing.  Supports
Python, JavaScript, TypeScript, Java, Go, Rust, C, C++, and Ruby out of
the box.  Falls back to whole-file indexing for unsupported languages so
that every source file is still searchable via BM25 / vector search.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from tree_sitter import Language, Node, Parser

import db as db_mod

logger = logging.getLogger(__name__)

# ── Directories to skip ───────────────────────────────────────────────
_SKIP_DIRS = frozenset({
    ".venv", "venv", "__pycache__", ".git", "node_modules",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
    "dist", "build", "target", "bin", "obj",
})

# ── File extensions we consider "source code" ─────────────────────────
_SOURCE_EXTENSIONS = frozenset({
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java",
    ".go", ".rs", ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx",
    ".rb", ".cs", ".swift", ".kt", ".kts", ".scala", ".lua",
    ".sh", ".bash", ".zsh", ".yaml", ".yml", ".toml", ".json",
    ".html", ".css", ".scss", ".sql", ".md", ".txt",
    ".dockerfile", ".makefile",
})

# ---------------------------------------------------------------------------
# Tree-sitter language registry  (lazy-loaded)
# ---------------------------------------------------------------------------

_LANGUAGES: dict[str, Language] = {}


def _load_language(ext: str) -> Language | None:
    """Return a tree-sitter Language for the given file extension, or None."""
    if ext in _LANGUAGES:
        return _LANGUAGES[ext]

    lang = _try_import_language(ext)
    if lang is not None:
        _LANGUAGES[ext] = lang
    return lang


def _try_import_language(ext: str) -> Language | None:
    """Attempt to import the tree-sitter grammar for *ext*."""
    try:
        if ext == ".py":
            import tree_sitter_python as mod
        elif ext in (".js", ".jsx"):
            import tree_sitter_javascript as mod
        elif ext in (".ts", ".tsx"):
            import tree_sitter_typescript as ts_mod
            # TypeScript grammar exposes typescript and tsx separately
            if ext == ".tsx":
                return Language(ts_mod.language_tsx())
            return Language(ts_mod.language_typescript())
        elif ext == ".java":
            import tree_sitter_java as mod
        elif ext == ".go":
            import tree_sitter_go as mod
        elif ext == ".rs":
            import tree_sitter_rust as mod
        elif ext in (".c", ".h"):
            import tree_sitter_c as mod
        elif ext in (".cpp", ".hpp", ".cc", ".cxx"):
            import tree_sitter_cpp as mod
        elif ext == ".rb":
            import tree_sitter_ruby as mod
        elif ext in (".kt", ".kts"):
            import tree_sitter_kotlin as mod
        else:
            return None
        return Language(mod.language())
    except ImportError:
        logger.debug("No tree-sitter grammar for %s", ext)
        return None


# ---------------------------------------------------------------------------
# Tree-sitter node-type → symbol kind mapping (per language family)
# ---------------------------------------------------------------------------

# Maps tree-sitter node types to our normalised (kind, is_container) pairs
_NODE_KIND_MAP: dict[str, tuple[str, bool]] = {
    # Python
    "function_definition": ("function", False),
    "class_definition":    ("class", True),
    # JS / TS
    "function_declaration":       ("function", False),
    "arrow_function":             ("function", False),
    "class_declaration":          ("class", True),
    "method_definition":          ("method", False),
    "lexical_declaration":        ("variable", False),
    # Java
    "method_declaration":         ("method", False),
    "constructor_declaration":    ("method", False),
    "interface_declaration":      ("class", True),
    # Go  (function_declaration already mapped above for JS/TS/Kotlin)
    "type_spec":                  ("class", False),
    # Rust
    "function_item":              ("function", False),
    "struct_item":                ("class", False),
    "impl_item":                  ("class", True),
    "enum_item":                  ("class", False),
    "trait_item":                 ("class", True),
    # C / C++
    "struct_specifier":           ("class", False),
    "class_specifier":            ("class", True),
    # Kotlin
    "object_declaration":         ("class", True),
    "companion_object":           ("class", True),
    # Ruby
    "method":                     ("method", False),
    "singleton_method":           ("method", False),
    "class":                      ("class", True),
    "module":                     ("class", True),
}


def _node_name(node: Node, source: bytes) -> str:
    """Extract the symbol name from a tree-sitter node."""
    # Most definitions have a 'name' or 'identifier' child
    for child in node.children:
        if child.type in ("identifier", "name", "property_identifier",
                          "type_identifier", "constant"):
            return source[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
    # Fallback: first identifier anywhere in the node
    ident = _first_identifier(node, source)
    if ident:
        return ident
    return f"<anonymous@{node.start_point[0] + 1}>"


def _first_identifier(node: Node, source: bytes) -> str | None:
    """DFS for the first identifier node."""
    if node.type in ("identifier", "name"):
        return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
    for child in node.children:
        result = _first_identifier(child, source)
        if result:
            return result
    return None


# ---------------------------------------------------------------------------
# Symbol extraction via tree-sitter
# ---------------------------------------------------------------------------

def _extract_symbols(
    tree_root: Node,
    source: bytes,
) -> list[dict[str, Any]]:
    """Walk the tree-sitter AST and extract symbols.

    Returns a flat list of dicts with keys:
      name, kind, line_start, line_end, source_text, children (nested symbols)
    """
    symbols: list[dict[str, Any]] = []

    def _walk(node: Node, parent_kind: str | None = None):
        node_type = node.type
        mapping = _NODE_KIND_MAP.get(node_type)

        if mapping:
            kind, is_container = mapping
            # Promote function → method if parent is a class/container
            if kind == "function" and parent_kind in ("class",):
                kind = "method"

            name = _node_name(node, source)
            src_text = source[node.start_byte:node.end_byte].decode(
                "utf-8", errors="replace"
            )
            sym = {
                "name": name,
                "kind": kind,
                "line_start": node.start_point[0] + 1,  # 1-indexed
                "line_end": node.end_point[0] + 1,
                "source_text": src_text,
                "children": [],
            }
            symbols.append(sym)

            # Recurse into container nodes (classes, impl blocks, etc.)
            if is_container:
                child_syms_before = len(symbols)
                for child in node.children:
                    _walk(child, parent_kind=kind)
                # Attach newly-found children
                sym["children"] = symbols[child_syms_before:]
            return

        # Not a symbol node — recurse into children
        for child in node.children:
            _walk(child, parent_kind=parent_kind)

    _walk(tree_root)
    return symbols


def _extract_references(tree_root: Node, source: bytes) -> list[dict[str, Any]]:
    """Extract identifier references from the tree-sitter AST."""
    refs: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    def _walk(node: Node):
        if node.type in ("identifier", "name", "type_identifier"):
            name = source[node.start_byte:node.end_byte].decode(
                "utf-8", errors="replace"
            )
            line = node.start_point[0] + 1
            key = (name, line)
            if key not in seen:
                seen.add(key)
                refs.append({"name": name, "line": line})
        for child in node.children:
            _walk(child)

    _walk(tree_root)
    return refs


# ---------------------------------------------------------------------------
# Single-file indexer
# ---------------------------------------------------------------------------

def index_file(filepath: str, db) -> dict:
    """Parse a single source file and index its symbols + references.

    Uses tree-sitter when a grammar is available for the file's language.
    Falls back to indexing the whole file as a single symbol otherwise.
    Skips the file if its ``last_modified`` timestamp has not changed.

    Args:
        filepath: Absolute path to a source file.
        db: An open ``sqlite3.Connection`` from ``db.get_db()``.

    Returns:
        A dict with ``file``, ``symbols_indexed``, ``references_indexed``,
        and ``skipped`` keys.
    """
    filepath = os.path.abspath(filepath)
    ext = os.path.splitext(filepath)[1].lower()

    # ── Check freshness ───────────────────────────────────────────────
    mtime = os.path.getmtime(filepath)
    row = db.execute(
        "SELECT id, last_modified FROM files WHERE path = ?", (filepath,)
    ).fetchone()

    if row and row[1] >= mtime:
        return {"file": filepath, "symbols_indexed": 0,
                "references_indexed": 0, "skipped": True}

    # ── Read file ─────────────────────────────────────────────────────
    source_bytes = Path(filepath).read_bytes()
    source_text = source_bytes.decode("utf-8", errors="replace")

    fhash = db_mod.file_hash(filepath)
    file_id = db_mod.upsert_file(db, filepath, mtime, fhash)

    # Delete stale data before re-inserting
    db_mod.delete_file_data(db, file_id)

    symbols_indexed = 0
    references_indexed = 0

    # ── Try tree-sitter parsing ───────────────────────────────────────
    lang = _load_language(ext)

    if lang is not None:
        parser = Parser(lang)
        tree = parser.parse(source_bytes)

        # Extract and store symbols
        raw_symbols = _extract_symbols(tree.root_node, source_bytes)

        # Flatten: process top-level symbols and nested children
        def _store_symbols(sym_list, parent_id=None):
            nonlocal symbols_indexed
            for sym in sym_list:
                sym_id = db_mod.upsert_symbol(
                    db, sym["name"], sym["kind"], file_id,
                    sym["line_start"], sym["line_end"],
                    parent_id, sym["source_text"],
                )
                symbols_indexed += 1

                # Generate embedding
                embed_input = f"{sym['kind']} {sym['name']}: {sym['source_text'][:1000]}"
                vec = db_mod.embed_text(embed_input)
                db_mod.upsert_embedding(db, sym_id, vec)

                # Recurse into children
                if sym.get("children"):
                    _store_symbols(sym["children"], parent_id=sym_id)

        _store_symbols(raw_symbols)

        # Extract and store references
        refs = _extract_references(tree.root_node, source_bytes)
        for ref in refs:
            db_mod.upsert_reference(db, ref["name"], file_id, ref["line"])
            references_indexed += 1

    else:
        # ── Fallback: index entire file as one symbol ─────────────────
        basename = os.path.basename(filepath)
        sym_id = db_mod.upsert_symbol(
            db, basename, "file", file_id,
            1, source_text.count("\n") + 1,
            None, source_text[:5000],
        )
        symbols_indexed += 1

        embed_input = f"file {basename}: {source_text[:1000]}"
        vec = db_mod.embed_text(embed_input)
        db_mod.upsert_embedding(db, sym_id, vec)

    db.commit()
    return {
        "file": filepath,
        "symbols_indexed": symbols_indexed,
        "references_indexed": references_indexed,
        "skipped": False,
    }


# ---------------------------------------------------------------------------
# Directory indexer
# ---------------------------------------------------------------------------

def index_directory(dirpath: str, db) -> list[dict]:
    """Recursively index all source files under *dirpath*.

    Skips directories in ``_SKIP_DIRS`` and unchanged files.  Indexes any
    file with a recognised source-code extension.

    Args:
        dirpath: Root directory to scan.
        db: An open ``sqlite3.Connection`` from ``db.get_db()``.

    Returns:
        A list of per-file result dicts (see :func:`index_file`).
    """
    results: list[dict] = []
    dirpath = os.path.abspath(dirpath)

    for root, dirs, files in os.walk(dirpath, topdown=True):
        # Prune skipped directories in-place
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS
                   and not d.endswith(".egg-info")]

        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            # Accept files with known extensions, or files with a
            # tree-sitter grammar available
            if ext not in _SOURCE_EXTENSIONS and _load_language(ext) is None:
                continue

            fpath = os.path.join(root, fname)
            try:
                result = index_file(fpath, db)
                results.append(result)
            except Exception:
                logger.exception("Failed to index %s", fpath)
                results.append({
                    "file": fpath,
                    "symbols_indexed": 0,
                    "references_indexed": 0,
                    "skipped": True,
                    "error": True,
                })
    return results
