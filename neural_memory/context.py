"""Token-budgeted context builder for neural memory.

Produces a compact snapshot (~300–500 tokens) suitable for automatic
injection at session start via the UserPromptSubmit hook.

The output is intentionally terse — it tells Claude what matters right
now (staleness, active bugs/tasks, relevant nodes) without flooding the
context window. Each section has a hard token budget enforced by character
count (≈4 chars/token heuristic).
"""

from __future__ import annotations

from typing import Optional


# ── Character limits per section (≈4 chars/token) ─────────────────────────────

_CHARS_STALENESS = 300      # ~75 tokens
_CHARS_OVERVIEW = 400       # ~100 tokens
_CHARS_BUGS = 500           # ~125 tokens
_CHARS_TASKS = 600          # ~150 tokens
_CHARS_RELEVANT = 800       # ~200 tokens


def _trunc(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3] + "..."


def build_context(
    project_root: str = ".",
    query_hint: Optional[str] = None,
    token_budget: int = 500,
) -> str:
    """Build a compact context snapshot for the current project.

    Args:
        project_root: Project root directory path.
        query_hint: Optional string (e.g. the user's prompt) to drive
                    semantic pre-fetch of relevant code nodes.
        token_budget: Approximate token budget. Scales section limits.
                      Default 500 is designed for hook injection.

    Returns:
        Markdown-formatted context string, always under budget.
    """
    # Scale character budgets proportionally if token_budget differs from default
    scale = token_budget / 500.0
    chars_staleness = int(_CHARS_STALENESS * scale)
    chars_overview = int(_CHARS_OVERVIEW * scale)
    chars_bugs = int(_CHARS_BUGS * scale)
    chars_tasks = int(_CHARS_TASKS * scale)
    chars_relevant = int(_CHARS_RELEVANT * scale)

    lines: list[str] = ["<!-- neural-memory context -->"]

    # ── 1. Staleness ──────────────────────────────────────────────────────────
    try:
        from .agent import check_staleness
        report = check_staleness(project_root)
        status_icon = {"healthy": "✓", "stale": "⚠", "uninitialized": "○", "error": "✗"}.get(report.status, "?")
        if report.status == "healthy":
            staleness_line = f"Neural memory {status_icon} {report.message}"
        else:
            staleness_line = f"Neural memory {status_icon} {report.message}"
        lines.append(_trunc(staleness_line, chars_staleness))
    except Exception:
        lines.append("Neural memory: status unavailable")

    # ── 2. Project overview ───────────────────────────────────────────────────
    try:
        from .storage import Storage
        from .models import NodeType
        with Storage(project_root) as storage:
            overview_nodes = storage.get_nodes_by_type(NodeType.PROJECT_OVERVIEW)
            if overview_nodes:
                ov = overview_nodes[0]
                summary = ov.summary_short or ov.summary_detailed or ""
                if summary:
                    lines.append(_trunc(f"Project: {summary}", chars_overview))
    except Exception:
        pass

    # ── 3. Active bugs ────────────────────────────────────────────────────────
    try:
        with Storage(project_root) as storage:
            bugs = storage.get_active_items("bugs")
        if bugs:
            bug_lines = [f"Active bugs ({len(bugs)}):"]
            budget_remaining = chars_bugs - 20
            for bug in bugs[:5]:
                severity = f"[{bug.severity}] " if bug.severity else ""
                line = f"  • {severity}{bug.summary_short or bug.name}"
                if len(line) > budget_remaining:
                    break
                bug_lines.append(_trunc(line, 120))
                budget_remaining -= len(line)
            lines.append("\n".join(bug_lines))
    except Exception:
        pass

    # ── 4. Active tasks ───────────────────────────────────────────────────────
    try:
        with Storage(project_root) as storage:
            tasks = storage.get_active_items("tasks")
        if tasks:
            task_lines = [f"Active tasks ({len(tasks)}):"]
            budget_remaining = chars_tasks - 20
            for task in tasks[:7]:
                status = f"[{task.task_status}] " if task.task_status else ""
                priority = f"({task.priority}) " if task.priority else ""
                line = f"  • {status}{priority}{task.summary_short or task.name}"
                if len(line) > budget_remaining:
                    break
                task_lines.append(_trunc(line, 120))
                budget_remaining -= len(line)
            lines.append("\n".join(task_lines))
    except Exception:
        pass

    # ── 5. Relevant nodes (only if query_hint provided) ───────────────────────
    if query_hint and query_hint.strip():
        try:
            from .embeddings import semantic_search, is_available
            with Storage(project_root) as storage:
                if is_available():
                    results = semantic_search(storage, query_hint.strip(), limit=3)
                    if results:
                        rel_lines = ["Relevant nodes:"]
                        budget_remaining = chars_relevant - 20
                        for r in results:
                            node = r.node
                            if node.category != "codebase":
                                continue
                            line = f"  • {node.name} ({node.file_path}:{node.line_start}) — {node.summary_short}"
                            if len(line) > budget_remaining:
                                break
                            rel_lines.append(_trunc(line, 180))
                            budget_remaining -= len(line)
                        if len(rel_lines) > 1:
                            lines.append("\n".join(rel_lines))
        except Exception:
            pass

    lines.append("<!-- /neural-memory -->")
    return "\n".join(lines)
