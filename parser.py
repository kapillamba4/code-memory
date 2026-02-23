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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import pathspec
from tree_sitter import Language, Node, Parser

import db as db_mod

logger = logging.getLogger(__name__)

# Number of worker threads for parallel indexing (configurable via env)
MAX_WORKERS = int(os.environ.get("CODE_MEMORY_MAX_WORKERS", "4"))

# ── Directories to always skip (even without .gitignore) ───────────────
_SKIP_DIRS = frozenset({
    ".venv", "venv", "__pycache__", ".git", "node_modules",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
    "dist", "build", "target", "bin", "obj",
})


def _load_gitignore_spec(root_dir: str) -> pathspec.PathSpec | None:
    """Load .gitignore patterns from the given directory.

    Returns a PathSpec object if .gitignore exists, None otherwise.
    """
    gitignore_path = os.path.join(root_dir, ".gitignore")
    if not os.path.isfile(gitignore_path):
        return None

    try:
        with open(gitignore_path, encoding="utf-8") as f:
            lines = f.readlines()
        return pathspec.PathSpec.from_lines("gitwildmatch", lines)
    except (OSError, UnicodeDecodeError) as e:
        logger.debug("Failed to read .gitignore: %s", e)
        return None


class GitignoreMatcher:
    """Manages .gitignore matching with support for nested .gitignore files.

    Git reads all .gitignore files in the directory tree, not just the root.
    Each nested .gitignore applies patterns relative to its own directory.
    """

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self._specs: dict[str, pathspec.PathSpec] = {}

        # Load root .gitignore if it exists
        root_spec = _load_gitignore_spec(root_dir)
        if root_spec:
            self._specs["."] = root_spec

    def _load_spec_for_dir(self, abs_dir: str, rel_dir: str) -> None:
        """Load .gitignore for a directory if not already loaded."""
        if rel_dir in self._specs:
            return

        spec = _load_gitignore_spec(abs_dir)
        if spec:
            self._specs[rel_dir] = spec

    def _get_parent_specs(self, rel_path: str) -> list[tuple[str, pathspec.PathSpec]]:
        """Get all applicable gitignore specs for a given path.

        Returns list of (base_dir, spec) tuples for specs that apply to this path.
        """
        result = []
        path_parts = rel_path.replace("\\", "/").split("/")

        # Check all ancestor directories that have .gitignore files
        for base_dir, spec in self._specs.items():
            if base_dir == ".":
                # Root spec applies to everything
                result.append((base_dir, spec))
            else:
                # Nested spec only applies if path is under that directory
                base_parts = base_dir.replace("\\", "/").split("/")
                if path_parts[:len(base_parts)] == base_parts:
                    result.append((base_dir, spec))

        return result

    def should_skip(self, rel_path: str, is_dir: bool) -> bool:
        """Check if a path should be skipped based on all applicable .gitignore patterns."""
        # Normalize path separators for matching
        rel_path = rel_path.replace("\\", "/")

        for base_dir, spec in self._get_parent_specs(rel_path):
            # For nested gitignores, compute path relative to that gitignore's directory
            if base_dir == ".":
                check_path = rel_path
            else:
                base_prefix = base_dir.replace("\\", "/") + "/"
                if rel_path.startswith(base_prefix):
                    check_path = rel_path[len(base_prefix):]
                else:
                    continue

            # Check both the path as-is and with trailing slash for directories
            if spec.match_file(check_path):
                return True
            if is_dir and spec.match_file(check_path + "/"):
                return True

        return False

    def check_dir_for_gitignore(self, abs_dir: str, rel_dir: str) -> None:
        """Check if directory contains a .gitignore and load it."""
        self._load_spec_for_dir(abs_dir, rel_dir)

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

    Optimized version using batch embeddings and transaction-based writes.

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

    fhash = db_mod.file_hash(filepath)  # Now uses xxHash
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

        # Extract symbols
        raw_symbols = _extract_symbols(tree.root_node, source_bytes)

        # === BATCH PROCESSING ===
        # Collect all symbols (including nested) for batch embedding
        all_symbols: list[tuple[dict, int | None]] = []  # (sym, parent_id)
        all_embed_inputs: list[str] = []

        def _collect_for_batch(sym_list, parent_id=None):
            for sym in sym_list:
                all_symbols.append((sym, parent_id))
                embed_input = f"{sym['kind']} {sym['name']}: {sym['source_text'][:1000]}"
                all_embed_inputs.append(embed_input)
                if sym.get("children"):
                    _collect_for_batch(sym["children"], parent_id=None)

        _collect_for_batch(raw_symbols)

        # Batch embed all at once
        if all_embed_inputs:
            embeddings = db_mod.embed_texts_batch(all_embed_inputs, batch_size=64)

            # Store all in single transaction
            with db_mod.transaction(db):
                for i, (sym, parent_id) in enumerate(all_symbols):
                    sym_id = db_mod.upsert_symbol(
                        db, sym["name"], sym["kind"], file_id,
                        sym["line_start"], sym["line_end"],
                        parent_id, sym["source_text"],
                        auto_commit=False
                    )
                    db_mod.upsert_embedding(db, sym_id, embeddings[i], auto_commit=False)
                    symbols_indexed += 1

        # Extract and store references (also batched)
        refs = _extract_references(tree.root_node, source_bytes)
        if refs:
            with db_mod.transaction(db):
                for ref in refs:
                    db_mod.upsert_reference(db, ref["name"], file_id, ref["line"], auto_commit=False)
                    references_indexed += 1

    else:
        # ── Fallback: index entire file as one symbol ─────────────────
        basename = os.path.basename(filepath)
        embeddings = db_mod.embed_texts_batch([f"file {basename}: {source_text[:1000]}"])

        with db_mod.transaction(db):
            sym_id = db_mod.upsert_symbol(
                db, basename, "file", file_id,
                1, source_text.count("\n") + 1,
                None, source_text[:5000],
                auto_commit=False
            )
            db_mod.upsert_embedding(db, sym_id, embeddings[0], auto_commit=False)
            symbols_indexed += 1

    return {
        "file": filepath,
        "symbols_indexed": symbols_indexed,
        "references_indexed": references_indexed,
        "skipped": False,
    }


# ---------------------------------------------------------------------------
# Directory indexer
# ---------------------------------------------------------------------------

def index_directory(dirpath: str, db, progress_callback=None) -> list[dict]:
    """Recursively index all source files under *dirpath* using parallel processing.

    Uses ThreadPoolExecutor for parallel file I/O and parsing, while keeping
    embedding generation sequential (sentence-transformers releases GIL during
    inference). Processes files in batches for embedding efficiency.

    Skips directories in ``_SKIP_DIRS``, files matching ``.gitignore`` patterns
    (including nested .gitignore files), and unchanged files.  Indexes any file
    with a recognised source-code extension.

    Args:
        dirpath: Root directory to scan.
        db: An open ``sqlite3.Connection`` from ``db.get_db()``.
        progress_callback: Optional callback(current, total, message) for progress updates.

    Returns:
        A list of per-file result dicts (see :func:`index_file`).
    """
    import time

    results: list[dict] = []
    dirpath = os.path.abspath(dirpath)
    total_start = time.perf_counter()

    # Initialize gitignore matcher (supports nested .gitignore files)
    gitignore = GitignoreMatcher(dirpath)
    logger.debug("Initialized gitignore matcher for %s", dirpath)

    # First pass: collect all files to index
    total_files = 0
    file_list = []
    for root, dirs, files in os.walk(dirpath, topdown=True):
        rel_root = os.path.relpath(root, dirpath)
        if rel_root != ".":
            gitignore.check_dir_for_gitignore(root, rel_root)
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.endswith(".egg-info")
                   and not gitignore.should_skip(os.path.join(rel_root, d) if rel_root != "." else d, is_dir=True)]
        for fname in sorted(files):
            rel_path = os.path.join(rel_root, fname) if rel_root != "." else fname
            if gitignore.should_skip(rel_path, is_dir=False):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext in _SOURCE_EXTENSIONS or _load_language(ext) is not None:
                file_list.append(os.path.join(root, fname))
                total_files += 1

    if not file_list:
        return []

    # Report initial phase
    if progress_callback:
        progress_callback(0, total_files, "Scanning files for changes...")

    # Phase 1: Parallel file freshness check and parsing
    # Each worker returns parsed data (not yet stored to DB)
    files_processed = 0
    parsed_files: list[tuple[str, dict | None, Exception | None]] = []  # (filepath, parsed_data, error)

    def _parse_file_task(fpath: str) -> tuple[str, dict | None, Exception | None]:
        """Parse a single file and return extracted data (without DB writes)."""
        try:
            parsed = _parse_file_for_indexing(fpath, db)
            return (fpath, parsed, None)
        except Exception as e:
            return (fpath, None, e)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all parsing tasks
        future_to_path = {executor.submit(_parse_file_task, fpath): fpath for fpath in file_list}

        for future in as_completed(future_to_path):
            fpath, parsed_data, error = future.result()
            parsed_files.append((fpath, parsed_data, error))

            files_processed += 1
            if progress_callback:
                fname = os.path.basename(fpath)
                progress_callback(files_processed, total_files, f"Parsing: {fname}")

    # Phase 2: Batch embedding generation (sequential, GIL released during inference)
    if progress_callback:
        progress_callback(total_files, total_files, "Generating embeddings...")

    # Collect all texts that need embedding
    embedding_batches: list[tuple[str, list[tuple]]] = []  # (filepath, [(embed_text, symbol_data), ...])

    for fpath, parsed_data, error in parsed_files:
        if error or parsed_data is None or parsed_data.get("skipped"):
            continue

        embed_inputs = []
        for sym in parsed_data.get("symbols", []):
            embed_input = f"{sym['kind']} {sym['name']}: {sym['source_text'][:1000]}"
            embed_inputs.append((embed_input, sym))

        if embed_inputs:
            embedding_batches.append((fpath, embed_inputs, parsed_data))

    # Generate embeddings in batch
    all_embed_texts = []
    for fpath, embed_inputs, _ in embedding_batches:
        for embed_text, _ in embed_inputs:
            all_embed_texts.append(embed_text)

    all_embeddings = db_mod.embed_texts_batch(all_embed_texts, batch_size=64) if all_embed_texts else []

    # Phase 3: Sequential DB writes (to avoid SQLite conflicts)
    if progress_callback:
        progress_callback(total_files, total_files, "Storing to database...")

    embed_idx = 0
    for fpath, parsed_data, error in parsed_files:
        if error:
            logger.exception("Failed to index %s", fpath)
            results.append({
                "file": fpath,
                "symbols_indexed": 0,
                "references_indexed": 0,
                "skipped": True,
                "error": True,
            })
            continue

        if parsed_data is None or parsed_data.get("skipped"):
            results.append({
                "file": fpath,
                "symbols_indexed": 0,
                "references_indexed": 0,
                "skipped": True,
            })
            continue

        # Find embeddings for this file
        file_result = _store_parsed_file(fpath, parsed_data, db, embedding_batches, all_embeddings, embed_idx)
        embed_idx += len(parsed_data.get("symbols", []))
        results.append(file_result)

    # Log performance summary
    total_elapsed = time.perf_counter() - total_start
    total_symbols = sum(r.get("symbols_indexed", 0) for r in results)
    total_refs = sum(r.get("references_indexed", 0) for r in results)
    files_indexed = sum(1 for r in results if not r.get("skipped"))
    files_skipped = sum(1 for r in results if r.get("skipped") and not r.get("error"))

    if total_files > 0:
        files_per_sec = total_files / total_elapsed if total_elapsed > 0 else 0
        logger.info(
            "Indexed %d files (%d skipped) in %.2fs (%.1f files/s) - %d symbols, %d references",
            files_indexed, files_skipped, total_elapsed, files_per_sec, total_symbols, total_refs
        )
    else:
        logger.info(
            "Indexed %d files (%d skipped) in %.2fs - %d symbols, %d references",
            files_indexed, files_skipped, total_elapsed, total_symbols, total_refs
        )

    return results


def _parse_file_for_indexing(filepath: str, db) -> dict | None:
    """Parse a file and extract symbols/references without DB writes.

    Returns parsed data structure or None if skipped.
    """
    filepath = os.path.abspath(filepath)
    ext = os.path.splitext(filepath)[1].lower()

    # Check freshness
    mtime = os.path.getmtime(filepath)
    row = db.execute(
        "SELECT id, last_modified FROM files WHERE path = ?", (filepath,)
    ).fetchone()

    if row and row[1] >= mtime:
        return {"skipped": True, "file_id": row[0]}

    # Read file
    source_bytes = Path(filepath).read_bytes()
    source_text = source_bytes.decode("utf-8", errors="replace")

    fhash = db_mod.file_hash(filepath)

    result = {
        "skipped": False,
        "mtime": mtime,
        "fhash": fhash,
        "symbols": [],
        "references": [],
        "fallback": False,
    }

    # Try tree-sitter parsing
    lang = _load_language(ext)

    if lang is not None:
        parser = Parser(lang)
        tree = parser.parse(source_bytes)

        # Extract symbols (flat list for batch processing)
        raw_symbols = _extract_symbols(tree.root_node, source_bytes)
        all_symbols: list[dict] = []

        def _collect_symbols(sym_list):
            for sym in sym_list:
                all_symbols.append(sym)
                if sym.get("children"):
                    _collect_symbols(sym["children"])

        _collect_symbols(raw_symbols)
        result["symbols"] = all_symbols

        # Extract references
        refs = _extract_references(tree.root_node, source_bytes)
        result["references"] = refs
    else:
        # Fallback: entire file as one symbol
        basename = os.path.basename(filepath)
        result["symbols"] = [{
            "name": basename,
            "kind": "file",
            "line_start": 1,
            "line_end": source_text.count("\n") + 1,
            "source_text": source_text[:5000],
            "parent_id": None,
        }]
        result["fallback"] = True

    return result


def _store_parsed_file(
    filepath: str,
    parsed_data: dict,
    db,
    embedding_batches: list,
    all_embeddings: list,
    start_embed_idx: int
) -> dict:
    """Store parsed file data to database with pre-computed embeddings."""
    filepath = os.path.abspath(filepath)

    # Upsert file record
    file_id = db_mod.upsert_file(db, filepath, parsed_data["mtime"], parsed_data["fhash"])

    # Delete stale data
    db_mod.delete_file_data(db, file_id)

    symbols_indexed = 0
    references_indexed = 0

    # Find embeddings for this file
    file_embeddings = None
    embed_offset = 0
    for bfpath, embed_inputs, _ in embedding_batches:
        if bfpath == filepath:
            file_embeddings = all_embeddings[start_embed_idx + embed_offset:start_embed_idx + embed_offset + len(embed_inputs)]
            break
        embed_offset += len(embed_inputs)

    # Store symbols with embeddings
    if parsed_data.get("symbols") and file_embeddings:
        with db_mod.transaction(db):
            for i, sym in enumerate(parsed_data["symbols"]):
                sym_id = db_mod.upsert_symbol(
                    db, sym["name"], sym["kind"], file_id,
                    sym["line_start"], sym["line_end"],
                    sym.get("parent_id"), sym["source_text"],
                    auto_commit=False
                )
                if i < len(file_embeddings):
                    db_mod.upsert_embedding(db, sym_id, file_embeddings[i], auto_commit=False)
                symbols_indexed += 1

    # Store references
    if parsed_data.get("references"):
        with db_mod.transaction(db):
            for ref in parsed_data["references"]:
                db_mod.upsert_reference(db, ref["name"], file_id, ref["line"], auto_commit=False)
                references_indexed += 1

    return {
        "file": filepath,
        "symbols_indexed": symbols_indexed,
        "references_indexed": references_indexed,
        "skipped": False,
    }
