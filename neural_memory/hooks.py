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

    Strategy:
    - Session start (empty prompt / first call): inject full context + saved session context.
    - Subsequent prompts: inject nothing. The full context at session start is enough.
      Claude already has the codebase in its active context window after the first message.

    This fixes the "circular token savings" problem where every prompt injected 500 tokens
    of redundant context that Claude already had from reading the files.
    """
    query_hint = os.environ.get("CLAUDE_USER_PROMPT", "").strip()
    project_root = os.environ.get("CLAUDE_PROJECT_ROOT", ".").strip() or "."

    # Per-process flag: have we already done the session-start injection?
    _first_call_key = f"_neural_first_call_{os.getpid()}"
    _start_time_key = f"_neural_session_start_{os.getpid()}"
    is_first_call = not os.environ.get(_first_call_key)
    if is_first_call:
        from datetime import datetime, timezone
        os.environ[_first_call_key] = "1"
        os.environ[_start_time_key] = datetime.now(timezone.utc).isoformat()

    if not is_first_call:
        # Subsequent prompts: stay silent. Claude has context from session start.
        return

    # Session start: inject full context
    try:
        from .context import build_context
        output = build_context(
            project_root=project_root,
            query_hint=query_hint or None,
            token_budget=500,
        )
        if "uninitialized" not in output and "unavailable" not in output:
            print(output)
        elif "uninitialized" in output:
            print("<!-- neural-memory: not indexed. Run /neural-index to build the knowledge graph. -->")
    except Exception:
        pass

    # Inject saved session context + recent session history on first call
    try:
        from pathlib import Path
        from .config import get_memory_dir
        ctx_file = get_memory_dir(project_root) / "session_context.md"
        if ctx_file.exists():
            saved = ctx_file.read_text(encoding="utf-8").strip()
            if saved:
                print("<!-- neural-memory: saved session context -->")
                print(saved)
                print("<!-- /neural-memory: saved session context -->")
    except Exception:
        pass

    # Inject recent session history (last 3 sessions) for cross-session memory
    try:
        from .storage import Storage
        with Storage(project_root) as storage:
            recent = storage.get_recent_sessions(limit=3)
        if recent:
            lines = ["<!-- neural-memory: recent session history -->"]
            for s in recent:
                date = (s["started_at"] or "")[:10]
                summary = s["summary"] or "(no summary)"
                files = s.get("files_touched", [])
                files_str = f" | files: {', '.join(files[:4])}" if files else ""
                lines.append(f"Session {date}: {summary}{files_str}")
            lines.append("<!-- /neural-memory: recent session history -->")
            print("\n".join(lines))
    except Exception:
        pass


def session_end() -> None:
    """Stop hook — archive completed items, write session log, and auto-update.

    Called by Claude Code when the session ends. Performs three tasks:
    1. Archives tasks with status=done and bugs with status=fixed
    2. Writes a session summary to the session_log table for cross-session memory
    3. Triggers incremental update if the index is stale (always, not just small diffs)

    All operations are best-effort and silent on failure.
    """
    import subprocess
    import uuid
    from datetime import datetime, timezone

    project_root = os.environ.get("CLAUDE_PROJECT_ROOT", ".").strip() or "."
    now = datetime.now(timezone.utc).isoformat()

    # Recover session start time recorded by prompt_context() on first call
    _start_time_key = f"_neural_session_start_{os.getpid()}"
    started_at = os.environ.get(_start_time_key) or now

    # Save session context snapshot for cross-session continuity
    try:
        from .context import save_session_context
        save_session_context(project_root=project_root)
    except Exception:
        pass

    # Determine files touched this session via git (best-effort)
    files_touched: list[str] = []
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD~1", "HEAD"],
            capture_output=True, text=True, cwd=project_root, timeout=5,
        )
        if result.returncode == 0:
            files_touched = [f.strip() for f in result.stdout.strip().splitlines() if f.strip()][:20]
    except Exception:
        pass

    # Archive completed items, collect recent insights, write session log — one connection
    try:
        from .storage import Storage
        with Storage(project_root) as storage:
            count = storage.archive_completed()
            if count > 0 and "--debug" in sys.argv:
                print(f"neural-memory: archived {count} completed item(s)", file=sys.stderr)

            insights_added: list[str] = []
            summary = ""
            try:
                insights = storage.get_insights()
                recent_insights = [n.name for n in insights[:3]]
                if recent_insights:
                    insights_added = recent_insights
                    summary = f"Insights: {'; '.join(recent_insights[:2])}"
            except Exception:
                pass

            storage.save_session(
                session_id=str(uuid.uuid4())[:8],
                started_at=started_at,
                ended_at=now,
                summary=summary,
                files_touched=files_touched,
                insights_added=insights_added,
                data={},
            )
    except Exception:
        pass

    # Auto-update whenever stale — session end is the right time, user isn't waiting
    try:
        from .agent import check_staleness
        report = check_staleness(project_root)
        if report.status == "stale":
            from .indexer import incremental_update
            incremental_update(project_root=project_root)
    except Exception:
        pass
