"""Neural Agent — hook-based staleness detection and action orchestration.

Runs lightweight checks (~50ms) to determine if neural memory needs attention.
Designed to be called on Claude Code invocation, not as a daemon.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import NeuralConfig, load_config, get_memory_dir, DB_FILE
from .storage import Storage

# All source file extensions tracked for staleness detection
_SOURCE_EXTENSIONS = frozenset([
    ".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".go", ".rs", ".java", ".c", ".cpp", ".h", ".hpp", ".cs", ".rb",
    ".sql", ".prisma",
])


@dataclass
class AgentReport:
    """What the agent found during its check."""
    status: str                  # "healthy" | "stale" | "uninitialized" | "error"
    message: str                 # Human-readable summary
    commits_behind: int = 0
    stale_files: list[str] = None
    suggested_action: str = ""   # "neural-index" | "neural-update" | ""
    details: dict = None

    def __post_init__(self):
        if self.stale_files is None:
            self.stale_files = []
        if self.details is None:
            self.details = {}


def _git_head(project_root: str) -> Optional[str]:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=project_root, timeout=5
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _git_commits_since(since_hash: str, project_root: str) -> int:
    try:
        r = subprocess.run(
            ["git", "rev-list", "--count", f"{since_hash}..HEAD"],
            capture_output=True, text=True, cwd=project_root, timeout=5
        )
        return int(r.stdout.strip()) if r.returncode == 0 else 0
    except Exception:
        return 0


def _git_changed_files_since(since_hash: str, project_root: str) -> list[str]:
    try:
        r = subprocess.run(
            ["git", "diff", "--name-only", since_hash, "HEAD"],
            capture_output=True, text=True, cwd=project_root, timeout=5
        )
        if r.returncode == 0:
            return [f for f in r.stdout.strip().split("\n")
                    if f.strip() and any(f.endswith(ext) for ext in _SOURCE_EXTENSIONS)]
        return []
    except Exception:
        return []


def check_staleness(project_root: str = ".") -> AgentReport:
    """Fast staleness check — designed to run in <50ms.

    Checks:
    1. Does .neural-memory/memory.db exist?
    2. How many commits behind is the index?
    3. Are there obviously stale files?
    """
    memory_dir = get_memory_dir(project_root)
    db_path = memory_dir / DB_FILE

    # Check 1: Is there even an index?
    if not db_path.exists():
        return AgentReport(
            status="uninitialized",
            message="Neural memory has not been initialized. Run `/neural-index` to build the knowledge graph.",
            suggested_action="neural-index"
        )

    try:
        config = load_config(project_root)
    except Exception as e:
        return AgentReport(
            status="error",
            message=f"Failed to load neural memory config: {e}",
        )

    # Check 2: Git staleness
    try:
        with Storage(project_root) as storage:
            state = storage.get_index_state()
            stats = storage.get_stats()
    except Exception as e:
        return AgentReport(
            status="error",
            message=f"Failed to read neural memory database: {e}",
        )

    if not state.last_full_index:
        return AgentReport(
            status="uninitialized",
            message="Neural memory exists but has never been fully indexed. Run `/neural-index`.",
            suggested_action="neural-index"
        )

    details = {
        "last_full_index": state.last_full_index,
        "last_incremental": state.last_incremental_update,
        "total_nodes": stats["total_nodes"],
        "total_edges": stats["total_edges"],
        "total_files": stats["total_files"],
        "index_mode": state.index_mode.value,
    }

    # Git comparison
    current_head = _git_head(project_root)
    commits_behind = 0
    stale_files = []

    if state.last_commit_hash and current_head:
        if state.last_commit_hash != current_head:
            commits_behind = _git_commits_since(state.last_commit_hash, project_root)
            stale_files = _git_changed_files_since(state.last_commit_hash, project_root)

    # Decision
    if commits_behind == 0 and not stale_files:
        return AgentReport(
            status="healthy",
            message=f"Neural memory is up to date. {stats['total_nodes']} nodes across {stats['total_files']} files.",
            details=details,
        )
    elif commits_behind >= config.staleness_threshold:
        return AgentReport(
            status="stale",
            message=f"Neural memory is {commits_behind} commits behind with {len(stale_files)} changed Python files. Run `/neural-update` to sync.",
            commits_behind=commits_behind,
            stale_files=stale_files[:20],  # Cap for display
            suggested_action="neural-update",
            details=details,
        )
    elif stale_files:
        return AgentReport(
            status="stale",
            message=f"Neural memory is {commits_behind} commit(s) behind. {len(stale_files)} file(s) changed. Consider `/neural-update`.",
            commits_behind=commits_behind,
            stale_files=stale_files[:20],
            suggested_action="neural-update",
            details=details,
        )
    else:
        return AgentReport(
            status="healthy",
            message=f"Neural memory is current. {stats['total_nodes']} nodes across {stats['total_files']} files.",
            details=details,
        )


def format_agent_report(report: AgentReport) -> str:
    """Format an agent report for display."""
    icons = {
        "healthy": "✅",
        "stale": "⚠️",
        "uninitialized": "🔧",
        "error": "❌",
    }
    icon = icons.get(report.status, "❓")
    lines = [f"{icon} **Neural Memory Agent**: {report.message}"]

    if report.commits_behind > 0:
        lines.append(f"   Commits behind: {report.commits_behind}")

    if report.stale_files:
        lines.append(f"   Changed files ({len(report.stale_files)}):")
        for f in report.stale_files[:5]:
            lines.append(f"   - {f}")
        if len(report.stale_files) > 5:
            lines.append(f"   ... and {len(report.stale_files) - 5} more")

    if report.suggested_action:
        lines.append(f"   **Suggested**: `/{report.suggested_action}`")

    return "\n".join(lines)
