"""Claude Code hook entry points for neural memory.

These are thin CLI wrappers designed to run fast and silently as
Claude Code hooks. They read from environment variables set by the
hook harness and write to stdout (which the harness injects into
Claude's context).

Entry points (defined in pyproject.toml):
  neural-context-hook   → UserPromptSubmit: inject compact context
  neural-session-hook   → Stop: archive completed items, auto-update

Design constraints:
  - Must complete in < 500ms on typical hardware
  - Must not crash (all errors are swallowed silently)
  - Must not write to stderr except on explicit --debug flag
  - Output to stdout is injected verbatim by the harness
"""

from __future__ import annotations

import os
import sys


def prompt_context() -> None:
    """UserPromptSubmit hook — inject compact neural memory context.

    Called by Claude Code before each user prompt. Reads the prompt text
    from the CLAUDE_USER_PROMPT environment variable (set by the harness),
    then emits a token-budgeted context snapshot to stdout.

    If the neural memory index doesn't exist yet, emits a single-line hint
    and exits — never crashes.
    """
    query_hint = os.environ.get("CLAUDE_USER_PROMPT", "").strip()
    project_root = os.environ.get("CLAUDE_PROJECT_ROOT", ".").strip() or "."

    try:
        from .context import build_context
        output = build_context(
            project_root=project_root,
            query_hint=query_hint or None,
            token_budget=500,
        )
        # Only emit if there's meaningful content (index exists)
        if "uninitialized" not in output and "unavailable" not in output:
            print(output)
        elif "uninitialized" in output:
            print("<!-- neural-memory: not indexed. Run /neural-index to build the knowledge graph. -->")
    except Exception:
        # Silent failure — never break the user's workflow
        pass


def session_end() -> None:
    """Stop hook — archive completed items and optionally auto-update.

    Called by Claude Code when the session ends. Performs two tasks:
    1. Archives tasks with status=done and bugs with status=fixed
    2. Triggers incremental update if index is stale and < 10 files changed

    Both operations are best-effort and silent on failure.
    """
    project_root = os.environ.get("CLAUDE_PROJECT_ROOT", ".").strip() or "."

    # Archive completed items
    try:
        from .storage import Storage
        with Storage(project_root) as storage:
            count = storage.archive_completed()
        if count > 0 and "--debug" in sys.argv:
            print(f"neural-memory: archived {count} completed item(s)", file=sys.stderr)
    except Exception:
        pass

    # Auto-update if stale and small change set
    try:
        from .agent import check_staleness
        report = check_staleness(project_root)
        if report.status == "stale" and 0 < report.commits_behind <= 3 and len(report.stale_files) <= 10:
            from .indexer import incremental_update
            incremental_update(project_root=project_root)
    except Exception:
        pass
