"""Tests for neural_memory.agent — staleness detection."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from neural_memory.agent import AgentReport, check_staleness, format_agent_report
from neural_memory.models import IndexMode, IndexState
from neural_memory.storage import Storage


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_db(tmp_path: Path) -> Path:
    """Create the .neural-memory directory and an empty memory.db file."""
    mem_dir = tmp_path / ".neural-memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    db = mem_dir / "memory.db"
    # Open storage so the schema is initialised, then close immediately.
    with Storage(str(tmp_path)) as s:
        pass
    return db


def _make_indexed_db(tmp_path: Path, commit_hash: str = "abc123") -> None:
    """Create a db that looks like a full index has been run."""
    _make_db(tmp_path)
    with Storage(str(tmp_path)) as s:
        state = IndexState(
            total_nodes=10,
            total_edges=5,
            index_mode=IndexMode.AST_ONLY,
            last_full_index="2026-01-01T00:00:00Z",
            last_commit_hash=commit_hash,
        )
        s.save_index_state(state)


# ── Status: uninitialized (no db) ─────────────────────────────────────────────

def test_uninitialized_no_db(tmp_path):
    """When memory.db does not exist the report status must be 'uninitialized'."""
    report = check_staleness(str(tmp_path))
    assert report.status == "uninitialized"
    assert report.suggested_action == "neural-index"
    assert "neural-index" in report.message.lower() or "neural-index" in report.suggested_action


# ── Status: uninitialized (db exists, but no full index) ──────────────────────

def test_uninitialized_no_index(tmp_path):
    """When the db exists but last_full_index is None the status is 'uninitialized'."""
    _make_db(tmp_path)
    # db exists but IndexState keeps its default (last_full_index = None)
    report = check_staleness(str(tmp_path))
    assert report.status == "uninitialized"
    assert report.suggested_action == "neural-index"


# ── Status: healthy ────────────────────────────────────────────────────────────

def test_healthy(tmp_path):
    """When the index is at HEAD commit the report status must be 'healthy'."""
    head_hash = "deadbeef" * 5  # 40-char fake SHA

    _make_indexed_db(tmp_path, commit_hash=head_hash)

    with (
        patch("neural_memory.agent._git_head", return_value=head_hash),
        patch("neural_memory.agent._git_commits_since", return_value=0),
        patch("neural_memory.agent._git_changed_files_since", return_value=[]),
    ):
        report = check_staleness(str(tmp_path))

    assert report.status == "healthy"
    assert report.commits_behind == 0
    assert report.stale_files == []


# ── Status: stale (commits behind) ────────────────────────────────────────────

def test_stale_commits_behind(tmp_path):
    """When HEAD has moved ahead of the indexed commit the status must be 'stale'."""
    old_hash = "oldhash1" * 5
    new_hash = "newhash2" * 5

    _make_indexed_db(tmp_path, commit_hash=old_hash)

    with (
        patch("neural_memory.agent._git_head", return_value=new_hash),
        patch("neural_memory.agent._git_commits_since", return_value=7),
        patch("neural_memory.agent._git_changed_files_since", return_value=["src/foo.py", "src/bar.py"]),
    ):
        report = check_staleness(str(tmp_path))

    assert report.status == "stale"
    assert report.commits_behind > 0
    assert report.commits_behind == 7
    assert len(report.stale_files) == 2
    assert report.suggested_action == "neural-update"


# ── Status: error ─────────────────────────────────────────────────────────────

def test_error_handling(tmp_path):
    """If Storage raises unexpectedly the report status must be 'error' (no crash)."""
    _make_db(tmp_path)

    with patch("neural_memory.agent.Storage") as mock_storage_cls:
        # Make the context manager __enter__ raise
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(side_effect=RuntimeError("db exploded"))
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_storage_cls.return_value = mock_cm

        report = check_staleness(str(tmp_path))

    assert report.status == "error"
    assert report.message  # non-empty error message


# ── format_agent_report ────────────────────────────────────────────────────────

def test_format_agent_report_healthy():
    """format_agent_report returns a non-empty string for a healthy report."""
    report = AgentReport(
        status="healthy",
        message="Neural memory is up to date. 42 nodes across 7 files.",
    )
    output = format_agent_report(report)
    assert isinstance(output, str)
    assert len(output) > 0
    assert "healthy" in output.lower() or "up to date" in output.lower() or "✅" in output


def test_format_agent_report_stale():
    """format_agent_report mentions commits_behind for a stale report."""
    report = AgentReport(
        status="stale",
        message="Neural memory is 3 commits behind.",
        commits_behind=3,
        stale_files=["app/main.py", "app/utils.py"],
        suggested_action="neural-update",
    )
    output = format_agent_report(report)
    assert isinstance(output, str)
    assert "3" in output            # commits_behind count must appear
    assert "neural-update" in output


def test_format_agent_report_uninitialized():
    """format_agent_report handles uninitialized status without crashing."""
    report = AgentReport(
        status="uninitialized",
        message="Neural memory has not been initialized.",
        suggested_action="neural-index",
    )
    output = format_agent_report(report)
    assert isinstance(output, str)
    assert "neural-index" in output


def test_format_agent_report_many_stale_files():
    """When stale_files > 5, format_agent_report shows '... and N more'."""
    files = [f"src/file_{i}.py" for i in range(8)]
    report = AgentReport(
        status="stale",
        message="Stale.",
        commits_behind=2,
        stale_files=files,
        suggested_action="neural-update",
    )
    output = format_agent_report(report)
    assert "more" in output
