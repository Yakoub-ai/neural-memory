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


def save_session_context(
    project_root: str = ".",
    token_budget: int = 800,
) -> str:
    """Save a rich session context snapshot for cross-session continuity.

    Richer than build_context() — includes code node connections for active
    tasks/bugs and the last 5 git commits. Writes to:
        {project_root}/.neural-memory/session_context.md

    Returns the path written, or empty string on failure.
    """
    import subprocess
    from pathlib import Path

    scale = token_budget / 800.0
    lines: list[str] = ["# Neural Memory — Session Context\n"]

    try:
        from .storage import Storage
        from .models import NodeType, EdgeType, NeuralNode

        with Storage(project_root) as storage:
            # ── Active tasks with code connections ──────────────────────────
            tasks = storage.get_active_items("tasks")
            if tasks:
                lines.append(f"## Active Tasks ({len(tasks)})\n")
                for task in tasks[:8]:
                    status = f"[{task.task_status}]" if task.task_status else ""
                    priority = f"({task.priority})" if task.priority else ""
                    lines.append(f"- {status}{priority} **{task.name}**")
                    # Include RELATES_TO code node connections
                    edges = storage.get_edges_from(task.id)
                    code_refs = []
                    for edge in edges:
                        if edge.edge_type == EdgeType.RELATES_TO:
                            target = storage.get_node(edge.target_id)
                            if target and target.category == "codebase":
                                code_refs.append(f"`{target.name}` ({target.file_path}:{target.line_start})")
                    if code_refs:
                        lines.append(f"  → {', '.join(code_refs[:3])}")
                lines.append("")

            # ── Active bugs with code connections ───────────────────────────
            bugs = storage.get_active_items("bugs")
            if bugs:
                lines.append(f"## Active Bugs ({len(bugs)})\n")
                for bug in bugs[:5]:
                    severity = f"[{bug.severity}]" if bug.severity else ""
                    lines.append(f"- {severity} **{bug.name}**")
                    edges = storage.get_edges_from(bug.id)
                    code_refs = []
                    for edge in edges:
                        if edge.edge_type == EdgeType.RELATES_TO:
                            target = storage.get_node(edge.target_id)
                            if target and target.category == "codebase":
                                code_refs.append(f"`{target.name}` ({target.file_path}:{target.line_start})")
                    if code_refs:
                        lines.append(f"  → {', '.join(code_refs[:3])}")
                lines.append("")

            # ── Top important code nodes ─────────────────────────────────────
            import json as _json
            top_rows = storage.conn.execute(
                """SELECT data FROM nodes
                   WHERE category = 'codebase' AND (archived IS NULL OR archived = 0)
                   ORDER BY importance DESC LIMIT 5"""
            ).fetchall()
            top_code_nodes = [NeuralNode.from_dict(_json.loads(r["data"])) for r in top_rows]
            if top_code_nodes:
                lines.append("## Key Nodes\n")
                for node in top_code_nodes:
                    summary = node.summary_short or ""
                    lines.append(f"- `{node.name}` ({node.file_path}:{node.line_start}) — {summary[:80]}")
                lines.append("")

    except Exception:
        pass

    # ── Recent git commits ───────────────────────────────────────────────────
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "-5"],
            capture_output=True, text=True, cwd=project_root, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            lines.append("## Recent Commits\n")
            for commit_line in result.stdout.strip().splitlines():
                lines.append(f"- {commit_line}")
            lines.append("")
    except Exception:
        pass

    if len(lines) <= 1:
        return ""  # Nothing to save

    content = "\n".join(lines)

    # Write to .neural-memory/session_context.md
    try:
        from .config import get_memory_dir
        memory_dir = get_memory_dir(project_root)
        memory_dir.mkdir(parents=True, exist_ok=True)
        out_path = memory_dir / "session_context.md"
        out_path.write_text(content, encoding="utf-8")
        return str(out_path)
    except Exception:
        return ""
