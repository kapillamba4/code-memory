"""
Git history search module for code-memory.

Provides structured access to local Git data (commits, diffs, blame) via
the ``gitpython`` library.  All functions return plain dicts so the MCP
layer can serialise them directly to JSON.

Design rules
------------
- NO shell-outs — everything goes through ``git.Repo`` Python API.
- Errors are returned as ``{"error": "…"}`` dicts, never raised.
- Results are capped with sensible defaults to keep LLM context small.
- All timestamps are ISO 8601.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import git
from git.exc import InvalidGitRepositoryError, NoSuchPathError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _commit_to_dict(commit: git.Commit, *, include_files_changed_count: bool = False) -> dict[str, Any]:
    """Serialise a ``git.Commit`` to a flat dict.

    Args:
        include_files_changed_count: If True, compute the number of files
            changed (triggers a diff — slow for bulk iteration).
    """
    dt = datetime.fromtimestamp(commit.committed_date, tz=UTC)
    result: dict[str, Any] = {
        "hash": commit.hexsha[:7],
        "full_hash": commit.hexsha,
        "message": commit.message.strip(),
        "author": str(commit.author),
        "author_email": str(commit.author.email) if commit.author.email else "",
        "date": dt.isoformat(),
    }
    if include_files_changed_count:
        try:
            result["files_changed"] = commit.stats.total["files"]
        except Exception:
            result["files_changed"] = 0
    return result


# ---------------------------------------------------------------------------
# 1. Repository resolution
# ---------------------------------------------------------------------------

def get_repo(path: str = ".") -> git.Repo:
    """Resolve the Git repository that contains *path*.

    Searches upward from *path* for a ``.git`` directory so callers can
    pass any file or subdirectory inside the repo.

    Args:
        path: A file or directory inside the repository.

    Returns:
        A ``git.Repo`` instance.

    Raises:
        InvalidGitRepositoryError: When no ``.git`` can be found.
        NoSuchPathError: When *path* does not exist.
    """
    resolved = Path(path).resolve()
    return git.Repo(str(resolved), search_parent_directories=True)


# ---------------------------------------------------------------------------
# 2. Commit message search
# ---------------------------------------------------------------------------

def search_commits(
    repo: git.Repo,
    query: str,
    target_file: str | None = None,
    max_results: int = 20,
) -> list[dict[str, Any]]:
    """Search commit messages for *query* (case-insensitive substring).

    Args:
        repo: An open ``git.Repo``.
        query: Text to match against commit messages.
        target_file: If given, restrict to commits that touched this file.
        max_results: Maximum commits to return.

    Returns:
        A list of commit dicts, most-recent-first.
    """
    try:
        query_lower = query.lower()
        results: list[dict[str, Any]] = []

        iter_kwargs: dict[str, Any] = {"max_count": max_results * 5}
        if target_file:
            iter_kwargs["paths"] = target_file

        for commit in repo.iter_commits(**iter_kwargs):
            if query_lower in commit.message.lower():
                results.append(_commit_to_dict(commit))
                if len(results) >= max_results:
                    break

        return results

    except (InvalidGitRepositoryError, NoSuchPathError, ValueError) as exc:
        return [{"error": str(exc)}]
    except Exception as exc:
        return [{"error": f"Unexpected error: {exc}"}]


# ---------------------------------------------------------------------------
# 3. Commit detail (with optional diff)
# ---------------------------------------------------------------------------

def get_commit_detail(
    repo: git.Repo,
    commit_hash: str,
    target_file: str | None = None,
) -> dict[str, Any]:
    """Return detailed metadata (and optionally a diff) for one commit.

    Args:
        repo: An open ``git.Repo``.
        commit_hash: Full or abbreviated SHA.
        target_file: If given, include the unified diff for this file only.

    Returns:
        A dict with full commit info, file stats, and optional diff text.
    """
    try:
        commit = repo.commit(commit_hash)
    except Exception as exc:
        return {"error": f"Could not resolve commit '{commit_hash}': {exc}"}

    try:
        dt = datetime.fromtimestamp(commit.committed_date, tz=UTC)

        parent_hashes = [p.hexsha[:7] for p in commit.parents]

        # File-level stats
        files_changed: list[dict[str, Any]] = []
        try:
            for fpath, stat in commit.stats.files.items():
                files_changed.append({
                    "path": fpath,
                    "insertions": stat.get("insertions", 0),
                    "deletions": stat.get("deletions", 0),
                })
        except Exception:
            pass

        # Optional diff for a specific file
        diff_text: str | None = None
        if target_file:
            try:
                if commit.parents:
                    diffs = commit.parents[0].diff(commit, paths=[target_file], create_patch=True)
                else:
                    diffs = commit.diff(git.NULL_TREE, paths=[target_file], create_patch=True)

                parts = []
                for d in diffs:
                    if d.diff:
                        decoded = d.diff.decode("utf-8", errors="replace") if isinstance(d.diff, bytes) else d.diff
                        parts.append(decoded)
                diff_text = "\n".join(parts) if parts else None
            except Exception:
                diff_text = None

        return {
            "hash": commit.hexsha[:7],
            "full_hash": commit.hexsha,
            "message": commit.message.strip(),
            "author": str(commit.author),
            "author_email": str(commit.author.email) if commit.author.email else "",
            "date": dt.isoformat(),
            "parent_hashes": parent_hashes,
            "files_changed": files_changed,
            "diff": diff_text,
        }

    except (InvalidGitRepositoryError, NoSuchPathError, ValueError) as exc:
        return {"error": str(exc)}
    except Exception as exc:
        return {"error": f"Unexpected error: {exc}"}


# ---------------------------------------------------------------------------
# 4. File history (git log --follow)
# ---------------------------------------------------------------------------

def get_file_history(
    repo: git.Repo,
    file_path: str,
    max_results: int = 20,
) -> list[dict[str, Any]]:
    """Return the commit history for a single file, following renames.

    Equivalent to ``git log --follow <file_path>``.

    Args:
        repo: An open ``git.Repo``.
        file_path: Path to the file (relative to repo root).
        max_results: Maximum commits to return.

    Returns:
        A list of commit dicts, most-recent-first.
    """
    try:
        results: list[dict[str, Any]] = []
        for commit in repo.iter_commits(paths=file_path, max_count=max_results, follow=True):
            results.append(_commit_to_dict(commit))
        return results

    except (InvalidGitRepositoryError, NoSuchPathError, ValueError) as exc:
        return [{"error": str(exc)}]
    except Exception as exc:
        return [{"error": f"Unexpected error: {exc}"}]


# ---------------------------------------------------------------------------
# 5. Blame
# ---------------------------------------------------------------------------

def get_blame(
    repo: git.Repo,
    file_path: str,
    line_start: int | None = None,
    line_end: int | None = None,
) -> list[dict[str, Any]]:
    """Run ``git blame`` on *file_path*, optionally limited to a line range.

    Consecutive lines from the same commit are grouped into a single entry
    with ``line_start`` / ``line_end`` and a merged ``line_content`` to keep
    the output compact.

    Args:
        repo: An open ``git.Repo``.
        file_path: Path to file (relative to repo root).
        line_start: First line of interest (1-indexed, inclusive).
        line_end: Last line of interest (1-indexed, inclusive).

    Returns:
        A list of grouped blame entry dicts.
    """
    try:
        blame_data = repo.blame("HEAD", file_path)
    except Exception as exc:
        return [{"error": f"Blame failed for '{file_path}': {exc}"}]

    try:
        # Flatten blame into per-line entries
        flat: list[dict[str, Any]] = []
        current_line = 1
        for commit, lines in blame_data:
            for line in lines:
                line_text = line.decode("utf-8", errors="replace") if isinstance(line, bytes) else line
                flat.append({
                    "line_number": current_line,
                    "commit_hash": commit.hexsha[:7],
                    "full_hash": commit.hexsha,
                    "author": str(commit.author),
                    "date": datetime.fromtimestamp(
                        commit.committed_date, tz=UTC
                    ).isoformat(),
                    "line_content": line_text,
                    "commit_message": commit.message.strip().split("\n")[0],
                })
                current_line += 1

        # Filter to requested line range
        if line_start is not None:
            flat = [e for e in flat if e["line_number"] >= line_start]
        if line_end is not None:
            flat = [e for e in flat if e["line_number"] <= line_end]

        # Group consecutive lines from the same commit
        grouped: list[dict[str, Any]] = []
        for entry in flat:
            if (
                grouped
                and grouped[-1]["commit_hash"] == entry["commit_hash"]
                and grouped[-1]["line_end"] == entry["line_number"] - 1
            ):
                grouped[-1]["line_end"] = entry["line_number"]
                grouped[-1]["line_content"] += "\n" + entry["line_content"]
            else:
                grouped.append({
                    "line_start": entry["line_number"],
                    "line_end": entry["line_number"],
                    "commit_hash": entry["commit_hash"],
                    "author": entry["author"],
                    "date": entry["date"],
                    "line_content": entry["line_content"],
                    "commit_message": entry["commit_message"],
                })

        return grouped

    except (InvalidGitRepositoryError, NoSuchPathError, ValueError) as exc:
        return [{"error": str(exc)}]
    except Exception as exc:
        return [{"error": f"Unexpected error: {exc}"}]
