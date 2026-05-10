"""Tests for find_dead_code: query layer + server tool."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import db as db_mod
import queries
from queries import (
    _has_decorator_above,
    _is_excluded_from_dead_code,
    _is_test_path,
    _score_dead_code_candidate,
    _source_excerpt,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_dead_code_db(tmp_path: Path) -> sqlite3.Connection:
    """Create a SQLite DB with just the tables find_dead_code needs.

    Avoids loading sqlite-vec or the embedding model so unit tests stay fast.
    """
    db_path = tmp_path / "test.db"
    db = sqlite3.connect(db_path)
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS files (
            id            INTEGER PRIMARY KEY,
            path          TEXT    UNIQUE NOT NULL,
            last_modified REAL    NOT NULL,
            file_hash     TEXT    NOT NULL
        );
        CREATE TABLE IF NOT EXISTS symbols (
            id               INTEGER PRIMARY KEY,
            name             TEXT    NOT NULL,
            kind             TEXT    NOT NULL,
            file_id          INTEGER NOT NULL REFERENCES files(id),
            line_start       INTEGER NOT NULL,
            line_end         INTEGER NOT NULL,
            parent_symbol_id INTEGER,
            source_text      TEXT    NOT NULL
        );
        CREATE TABLE IF NOT EXISTS references_ (
            id          INTEGER PRIMARY KEY,
            symbol_name TEXT    NOT NULL,
            file_id     INTEGER NOT NULL REFERENCES files(id),
            line_number INTEGER NOT NULL
        );
        """
    )
    db.commit()
    return db


def _add_file(db: sqlite3.Connection, path: str) -> int:
    cur = db.execute(
        "INSERT INTO files (path, last_modified, file_hash) VALUES (?, ?, ?)",
        (path, 0.0, "x"),
    )
    db.commit()
    return cur.lastrowid


def _add_symbol(
    db: sqlite3.Connection,
    file_id: int,
    name: str,
    kind: str,
    line_start: int,
    line_end: int,
    source_text: str = "",
    parent_id: int | None = None,
) -> int:
    cur = db.execute(
        """INSERT INTO symbols
               (name, kind, file_id, line_start, line_end, parent_symbol_id, source_text)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (name, kind, file_id, line_start, line_end, parent_id, source_text),
    )
    db.commit()
    return cur.lastrowid


def _add_ref(db: sqlite3.Connection, file_id: int, name: str, line: int) -> None:
    db.execute(
        "INSERT INTO references_ (symbol_name, file_id, line_number) VALUES (?, ?, ?)",
        (name, file_id, line),
    )
    db.commit()


@pytest.fixture
def dc_db(temp_dir):
    """Empty database with the schema find_dead_code needs."""
    db = _build_dead_code_db(temp_dir)
    yield db
    db.close()


# ---------------------------------------------------------------------------
# Helper: _is_test_path
# ---------------------------------------------------------------------------


class TestIsTestPath:
    def test_test_prefix(self):
        assert _is_test_path("/repo/tests/test_foo.py") is True

    def test_test_suffix(self):
        assert _is_test_path("/repo/foo_test.py") is True

    def test_jest_spec(self):
        assert _is_test_path("/repo/foo.spec.ts") is True
        assert _is_test_path("/repo/foo.test.tsx") is True

    def test_conftest(self):
        assert _is_test_path("/repo/conftest.py") is True

    def test_tests_directory(self):
        assert _is_test_path("/repo/tests/sub/file.py") is True
        assert _is_test_path("/repo/__tests__/file.js") is True

    def test_normal_file(self):
        assert _is_test_path("/repo/src/foo.py") is False


# ---------------------------------------------------------------------------
# Helper: _is_excluded_from_dead_code
# ---------------------------------------------------------------------------


class TestIsExcluded:
    def test_dunder_excluded(self):
        excluded, reason = _is_excluded_from_dead_code(
            "__init__", "method", "/x.py", False
        )
        assert excluded is True
        assert "dunder" in reason

    def test_short_double_underscore_not_dunder(self):
        # __x is name-mangled, not a protocol method — keep it as a candidate
        excluded, _ = _is_excluded_from_dead_code("__x", "function", "/x.py", False)
        assert excluded is False

    def test_main_excluded(self):
        excluded, reason = _is_excluded_from_dead_code(
            "main", "function", "/x.py", False
        )
        assert excluded is True
        assert "entry-point" in reason

    def test_anonymous_excluded(self):
        excluded, _ = _is_excluded_from_dead_code(
            "<anonymous@5>", "function", "/x.py", False
        )
        assert excluded is True

    def test_file_kind_excluded(self):
        excluded, _ = _is_excluded_from_dead_code("foo", "file", "/x.py", False)
        assert excluded is True

    def test_test_file_excluded_by_default(self):
        excluded, _ = _is_excluded_from_dead_code(
            "helper", "function", "/repo/tests/foo.py", False
        )
        assert excluded is True

    def test_test_file_included_when_opted_in(self):
        excluded, _ = _is_excluded_from_dead_code(
            "helper", "function", "/repo/tests/foo.py", True
        )
        assert excluded is False

    def test_normal_function_not_excluded(self):
        excluded, _ = _is_excluded_from_dead_code(
            "compute", "function", "/repo/src/x.py", False
        )
        assert excluded is False


# ---------------------------------------------------------------------------
# Helper: _score_dead_code_candidate
# ---------------------------------------------------------------------------


class TestScoreDeadCodeCandidate:
    def test_public_function_mentions_api_caveat(self):
        conf, reasons = _score_dead_code_candidate(
            "compute", "function", "/repo/src/x.py", 1, False,
        )
        assert 0.0 < conf < 1.0
        assert any(
            "public" in r.lower() or "exported api" in r.lower() for r in reasons
        )

    def test_private_higher_than_public(self):
        public_conf, _ = _score_dead_code_candidate(
            "compute", "function", "/repo/src/x.py", 1, False,
        )
        private_conf, _ = _score_dead_code_candidate(
            "_compute", "function", "/repo/src/x.py", 1, False,
        )
        assert private_conf > public_conf

    def test_name_mangled_highest_privacy(self):
        mangled_conf, reasons = _score_dead_code_candidate(
            "__internal", "function", "/repo/src/x.py", 1, False,
        )
        assert mangled_conf > 0.7
        assert any("name-mangled" in r.lower() for r in reasons)

    def test_shared_name_lowers_confidence(self):
        unique_conf, _ = _score_dead_code_candidate(
            "_helper", "function", "/repo/src/x.py", 1, False,
        )
        shared_conf, reasons = _score_dead_code_candidate(
            "_helper", "function", "/repo/src/x.py", 5, False,
        )
        assert shared_conf < unique_conf
        assert any("shared by 5" in r for r in reasons)

    def test_method_in_member_blind_lang_lower(self):
        py_conf, _ = _score_dead_code_candidate(
            "_helper", "method", "/repo/src/x.py", 1, False,
        )
        js_conf, reasons = _score_dead_code_candidate(
            "_helper", "method", "/repo/src/x.js", 1, False,
        )
        assert js_conf < py_conf
        assert any("member access" in r.lower() for r in reasons)

    def test_class_kind_slightly_lower(self):
        fn_conf, _ = _score_dead_code_candidate(
            "_X", "function", "/repo/src/x.py", 1, False,
        )
        cls_conf, reasons = _score_dead_code_candidate(
            "_X", "class", "/repo/src/x.py", 1, False,
        )
        assert cls_conf < fn_conf
        assert any("dynamic instantiation" in r.lower() for r in reasons)

    def test_init_py_lowers_confidence(self):
        normal_conf, _ = _score_dead_code_candidate(
            "compute", "function", "/repo/src/foo.py", 1, False,
        )
        init_conf, reasons = _score_dead_code_candidate(
            "compute", "function", "/repo/src/__init__.py", 1, False,
        )
        assert init_conf < normal_conf
        assert any("__init__.py" in r for r in reasons)

    def test_decorator_lowers_confidence(self):
        plain_conf, _ = _score_dead_code_candidate(
            "_compute", "function", "/repo/src/x.py", 1, False,
        )
        deco_conf, reasons = _score_dead_code_candidate(
            "_compute", "function", "/repo/src/x.py", 1, True,
        )
        assert deco_conf < plain_conf
        assert any("decorat" in r.lower() for r in reasons)

    def test_confidence_clamped_to_range(self):
        # All penalties stacked: should still be in [0, 0.99]
        conf, _ = _score_dead_code_candidate(
            "Foo", "method", "/repo/src/__init__.py", 50, True,
        )
        assert 0.0 <= conf <= 0.99

    def test_confidence_never_exceeds_99(self):
        # All boosts: should still be capped at 0.99
        conf, _ = _score_dead_code_candidate(
            "_internal", "function", "/repo/src/x.py", 1, False,
        )
        assert conf <= 0.99


# ---------------------------------------------------------------------------
# Helper: _source_excerpt
# ---------------------------------------------------------------------------


class TestSourceExcerpt:
    def test_first_nonempty_line(self):
        assert _source_excerpt("\n\ndef foo():\n    return 1") == "def foo():"

    def test_truncates_long_line(self):
        long_line = "x" * 200
        out = _source_excerpt(long_line)
        assert out is not None
        assert len(out) <= 120

    def test_none_for_empty(self):
        assert _source_excerpt(None) is None
        assert _source_excerpt("") is None
        assert _source_excerpt("\n\n  \n") is None


# ---------------------------------------------------------------------------
# Helper: _has_decorator_above
# ---------------------------------------------------------------------------


class TestHasDecoratorAbove:
    def test_with_decorator(self, temp_dir):
        f = temp_dir / "x.py"
        f.write_text("@decorator\ndef foo():\n    pass\n")
        assert _has_decorator_above(str(f), 2) is True

    def test_no_decorator(self, temp_dir):
        f = temp_dir / "x.py"
        f.write_text("def foo():\n    pass\n")
        assert _has_decorator_above(str(f), 1) is False

    def test_blank_lines_skipped(self, temp_dir):
        f = temp_dir / "x.py"
        f.write_text("@decorator\n\n\ndef foo():\n    pass\n")
        assert _has_decorator_above(str(f), 4) is True

    def test_missing_file_returns_false(self, temp_dir):
        assert _has_decorator_above(str(temp_dir / "missing.py"), 1) is False


# ---------------------------------------------------------------------------
# find_dead_code: core behavior
# ---------------------------------------------------------------------------


class TestFindDeadCodeBasics:
    def test_dead_function_flagged(self, dc_db):
        fid = _add_file(dc_db, "/repo/foo.py")
        _add_symbol(dc_db, fid, "compute", "function", 1, 3, "def compute():\n    return 1")
        # Self-reference at the def line is internal, no external refs anywhere
        _add_ref(dc_db, fid, "compute", 1)

        result = queries.find_dead_code(dc_db)
        assert len(result["candidates"]) == 1
        c = result["candidates"][0]
        assert c["name"] == "compute"
        assert c["kind"] == "function"
        assert 0.0 < c["confidence"] <= 0.99
        assert any("No references" in r for r in c["reasons"])

    def test_alive_function_not_flagged(self, dc_db):
        fid = _add_file(dc_db, "/repo/foo.py")
        _add_symbol(dc_db, fid, "compute", "function", 1, 3, "def compute():\n    return 1")
        _add_ref(dc_db, fid, "compute", 1)
        # External call at line 10 outside the function body
        _add_ref(dc_db, fid, "compute", 10)

        result = queries.find_dead_code(dc_db)
        assert result["candidates"] == []

    def test_recursive_function_still_flagged(self, dc_db):
        fid = _add_file(dc_db, "/repo/foo.py")
        _add_symbol(dc_db, fid, "compute", "function", 1, 5, "def compute():\n    compute()")
        _add_ref(dc_db, fid, "compute", 1)  # def line
        _add_ref(dc_db, fid, "compute", 2)  # recursion within body — still internal

        result = queries.find_dead_code(dc_db)
        assert len(result["candidates"]) == 1
        assert result["candidates"][0]["name"] == "compute"

    def test_method_called_from_sibling_alive(self, dc_db):
        fid = _add_file(dc_db, "/repo/foo.py")
        cls_id = _add_symbol(dc_db, fid, "Foo", "class", 1, 10, "class Foo: ...")
        _add_symbol(dc_db, fid, "bar", "method", 2, 4, "def bar(self): ...", parent_id=cls_id)
        _add_symbol(dc_db, fid, "baz", "method", 5, 7, "def baz(self): ...", parent_id=cls_id)
        # bar referenced from baz (line 6) — outside bar's [2,4] range
        _add_ref(dc_db, fid, "bar", 2)
        _add_ref(dc_db, fid, "bar", 6)
        _add_ref(dc_db, fid, "baz", 5)
        _add_ref(dc_db, fid, "Foo", 1)

        result = queries.find_dead_code(dc_db)
        names = {c["name"] for c in result["candidates"]}
        assert "bar" not in names
        assert "baz" in names
        assert "Foo" in names


# ---------------------------------------------------------------------------
# find_dead_code: exclusions
# ---------------------------------------------------------------------------


class TestFindDeadCodeExclusions:
    def test_dunder_excluded(self, dc_db):
        fid = _add_file(dc_db, "/repo/foo.py")
        _add_symbol(dc_db, fid, "__init__", "method", 1, 3, "def __init__(self): ...")
        _add_ref(dc_db, fid, "__init__", 1)

        assert queries.find_dead_code(dc_db)["candidates"] == []

    def test_main_excluded(self, dc_db):
        fid = _add_file(dc_db, "/repo/foo.py")
        _add_symbol(dc_db, fid, "main", "function", 1, 3, "def main(): ...")
        _add_ref(dc_db, fid, "main", 1)

        assert queries.find_dead_code(dc_db)["candidates"] == []

    def test_test_files_excluded_by_default(self, dc_db):
        fid = _add_file(dc_db, "/repo/tests/test_foo.py")
        _add_symbol(dc_db, fid, "helper", "function", 1, 3, "def helper(): ...")
        _add_ref(dc_db, fid, "helper", 1)

        assert queries.find_dead_code(dc_db)["candidates"] == []

    def test_test_files_included_when_opted_in(self, dc_db):
        fid = _add_file(dc_db, "/repo/tests/test_foo.py")
        _add_symbol(dc_db, fid, "helper", "function", 1, 3, "def helper(): ...")
        _add_ref(dc_db, fid, "helper", 1)

        result = queries.find_dead_code(dc_db, include_tests=True)
        assert len(result["candidates"]) == 1

    def test_anonymous_excluded(self, dc_db):
        fid = _add_file(dc_db, "/repo/foo.js")
        _add_symbol(dc_db, fid, "<anonymous@5>", "function", 5, 7, "() => 1")

        assert queries.find_dead_code(dc_db)["candidates"] == []

    def test_file_fallback_kind_excluded(self, dc_db):
        fid = _add_file(dc_db, "/repo/foo.unknown")
        _add_symbol(dc_db, fid, "foo.unknown", "file", 1, 5, "...")
        # 'file' isn't in the default kinds anyway, but the exclusion guards
        # against an explicit kinds=['file'] request as well.
        result = queries.find_dead_code(dc_db, kinds=["file"])
        assert result["candidates"] == []


# ---------------------------------------------------------------------------
# find_dead_code: filters and shape
# ---------------------------------------------------------------------------


class TestFindDeadCodeFilters:
    def test_min_confidence_filters(self, dc_db):
        fid = _add_file(dc_db, "/repo/foo.py")
        _add_symbol(dc_db, fid, "compute", "function", 1, 3, "def compute(): ...")
        _add_ref(dc_db, fid, "compute", 1)

        assert len(queries.find_dead_code(dc_db, min_confidence=0.0)["candidates"]) == 1
        assert queries.find_dead_code(dc_db, min_confidence=0.99)["candidates"] == []

    def test_kinds_filter(self, dc_db):
        fid = _add_file(dc_db, "/repo/foo.py")
        cls_id = _add_symbol(dc_db, fid, "Foo", "class", 1, 5, "class Foo: ...")
        _add_symbol(dc_db, fid, "bar", "method", 2, 4, "def bar(self): ...", parent_id=cls_id)
        _add_ref(dc_db, fid, "Foo", 1)
        _add_ref(dc_db, fid, "bar", 2)

        method_only = queries.find_dead_code(dc_db, kinds=["method"])
        assert {c["name"] for c in method_only["candidates"]} == {"bar"}

        class_only = queries.find_dead_code(dc_db, kinds=["class"])
        assert {c["name"] for c in class_only["candidates"]} == {"Foo"}

    def test_top_k_caps_results(self, dc_db):
        fid = _add_file(dc_db, "/repo/foo.py")
        for i in range(20):
            name = f"_dead_{i}"
            _add_symbol(dc_db, fid, name, "function", i * 5 + 1, i * 5 + 3, f"def {name}(): ...")
            _add_ref(dc_db, fid, name, i * 5 + 1)

        result = queries.find_dead_code(dc_db, top_k=5)
        assert len(result["candidates"]) == 5

    def test_empty_kinds_returns_empty(self, dc_db):
        fid = _add_file(dc_db, "/repo/foo.py")
        _add_symbol(dc_db, fid, "compute", "function", 1, 3, "def compute(): ...")
        _add_ref(dc_db, fid, "compute", 1)

        result = queries.find_dead_code(dc_db, kinds=[])
        assert result["candidates"] == []
        assert result["total_symbols"] == 0


class TestFindDeadCodeShape:
    def test_response_shape(self, dc_db):
        fid = _add_file(dc_db, "/repo/foo.py")
        _add_symbol(dc_db, fid, "_internal", "function", 1, 3, "def _internal(): ...")
        _add_ref(dc_db, fid, "_internal", 1)

        result = queries.find_dead_code(dc_db)
        for top_key in ("candidates", "scanned_symbols", "total_symbols", "limitations"):
            assert top_key in result

        c = result["candidates"][0]
        for key in (
            "name", "kind", "file_path", "line_start", "line_end",
            "confidence", "reasons", "source_excerpt",
        ):
            assert key in c
        assert isinstance(c["reasons"], list)
        assert all(isinstance(r, str) for r in c["reasons"])

    def test_limitations_includes_member_access_caveat_for_js(self, dc_db):
        fid = _add_file(dc_db, "/repo/foo.js")
        _add_symbol(dc_db, fid, "_helper", "method", 1, 3, "function _helper() {}")
        _add_ref(dc_db, fid, "_helper", 1)

        result = queries.find_dead_code(dc_db)
        assert any("member-access" in lim.lower() for lim in result["limitations"])

    def test_sorted_by_confidence_desc(self, dc_db):
        fid = _add_file(dc_db, "/repo/foo.py")
        _add_symbol(dc_db, fid, "_priv_a", "function", 1, 3, "def _priv_a(): ...")
        _add_ref(dc_db, fid, "_priv_a", 1)
        _add_symbol(dc_db, fid, "public_b", "function", 5, 7, "def public_b(): ...")
        _add_ref(dc_db, fid, "public_b", 5)

        result = queries.find_dead_code(dc_db, min_confidence=0.0)
        confidences = [c["confidence"] for c in result["candidates"]]
        assert confidences == sorted(confidences, reverse=True)

    def test_no_candidates_returns_empty_list_and_counts(self, dc_db):
        # Empty DB: nothing to scan
        result = queries.find_dead_code(dc_db)
        assert result["candidates"] == []
        assert result["total_symbols"] == 0
        assert result["scanned_symbols"] == 0


# ---------------------------------------------------------------------------
# find_dead_code: cross-file behavior
# ---------------------------------------------------------------------------


class TestFindDeadCodeCrossFile:
    def test_external_ref_in_different_file(self, dc_db):
        f1 = _add_file(dc_db, "/repo/a.py")
        f2 = _add_file(dc_db, "/repo/b.py")
        _add_symbol(dc_db, f1, "compute", "function", 1, 3, "def compute(): ...")
        _add_ref(dc_db, f1, "compute", 1)  # def line in a.py
        _add_ref(dc_db, f2, "compute", 5)  # used in b.py — external

        assert queries.find_dead_code(dc_db)["candidates"] == []

    def test_shared_name_alive_via_either_caller(self, dc_db):
        # Two same-named definitions in different files; if either has any
        # non-self reference, both end up alive (the reference index can't
        # disambiguate by signature).
        f1 = _add_file(dc_db, "/repo/a.py")
        f2 = _add_file(dc_db, "/repo/b.py")
        _add_symbol(dc_db, f1, "process", "function", 1, 3, "def process(): ...")
        _add_symbol(dc_db, f2, "process", "function", 1, 3, "def process(): ...")
        _add_ref(dc_db, f1, "process", 1)
        _add_ref(dc_db, f2, "process", 1)
        _add_ref(dc_db, f2, "process", 8)  # call in b.py

        result = queries.find_dead_code(dc_db)
        assert all(c["name"] != "process" for c in result["candidates"])


# ---------------------------------------------------------------------------
# server.find_dead_code: input validation
# ---------------------------------------------------------------------------


class TestFindDeadCodeServerValidation:
    def test_nonexistent_directory_returns_error(self):
        import server

        result = server.find_dead_code("/nonexistent/path")
        assert result.get("error") is True
        assert "ValidationError" in result.get("error_type", "")

    def test_min_confidence_above_one_returns_error(self, temp_dir):
        import server

        result = server.find_dead_code(str(temp_dir), min_confidence=1.5)
        assert result.get("error") is True
        assert "ValidationError" in result.get("error_type", "")

    def test_negative_min_confidence_returns_error(self, temp_dir):
        import server

        result = server.find_dead_code(str(temp_dir), min_confidence=-0.1)
        assert result.get("error") is True

    def test_invalid_kind_returns_error(self, temp_dir):
        import server

        result = server.find_dead_code(str(temp_dir), kinds=["function", "variable"])
        assert result.get("error") is True
        assert "ValidationError" in result.get("error_type", "")

    def test_empty_kinds_list_returns_error(self, temp_dir):
        import server

        result = server.find_dead_code(str(temp_dir), kinds=[])
        assert result.get("error") is True

    def test_top_k_too_large_returns_error(self, temp_dir):
        import server

        result = server.find_dead_code(str(temp_dir), top_k=10000)
        assert result.get("error") is True


# ---------------------------------------------------------------------------
# server.find_dead_code: end-to-end via real db_mod.get_db
# ---------------------------------------------------------------------------


@pytest.fixture
def prepopulated_directory(temp_dir):
    """Directory with a code_memory.db that bypasses embedding-model loading.

    Pre-creates the schema and an index_metadata row matching the configured
    ``EMBEDDING_MODEL_NAME`` so ``db_mod.get_db()`` short-circuits the model
    load on the next open.  We pick a tiny embedding dimension (8) — find_dead_code
    never reads the embedding tables, so the value is irrelevant beyond
    making the schema valid.
    """
    import sqlite_vec

    db_path = temp_dir / "code_memory.db"
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.executescript(db_mod._SCHEMA_SQL)
    db_mod._create_embedding_tables(conn, 8)
    conn.execute(
        "INSERT INTO index_metadata (key, value) VALUES ('embedding_model', ?)",
        (db_mod.EMBEDDING_MODEL_NAME,),
    )
    conn.execute(
        "INSERT INTO index_metadata (key, value) VALUES ('embedding_dim', ?)",
        ("8",),
    )
    conn.commit()
    yield temp_dir, conn
    conn.close()


class TestFindDeadCodeServerEndToEnd:
    def test_returns_candidates(self, prepopulated_directory):
        import server

        directory, conn = prepopulated_directory
        src_path = str(directory / "src.py")
        conn.execute(
            "INSERT INTO files (path, last_modified, file_hash) VALUES (?, ?, ?)",
            (src_path, 0.0, "h"),
        )
        fid = conn.execute(
            "SELECT id FROM files WHERE path = ?", (src_path,)
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO symbols (name, kind, file_id, line_start, line_end, "
            "parent_symbol_id, source_text) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("_dead_function", "function", fid, 1, 3, None,
             "def _dead_function():\n    pass"),
        )
        conn.execute(
            "INSERT INTO references_ (symbol_name, file_id, line_number) VALUES (?, ?, ?)",
            ("_dead_function", fid, 1),
        )
        conn.commit()

        result = server.find_dead_code(str(directory))
        assert result.get("status") == "ok"
        assert result.get("count") == 1
        assert result["candidates"][0]["name"] == "_dead_function"
        assert isinstance(result["limitations"], list)
        assert "directory" in result

    def test_empty_index_returns_hint(self, prepopulated_directory):
        import server

        directory, _ = prepopulated_directory
        result = server.find_dead_code(str(directory))
        assert result.get("status") == "ok"
        assert result.get("count") == 0
        assert "hint" in result
        assert "index_codebase" in result["hint"]
