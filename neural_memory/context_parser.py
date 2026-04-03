"""Parse .claude/ context log files into neural graph nodes + edges.

Supported log formats:
  context-log-gotchas.md   → BUG nodes (category="bugs")
  context-log-tasks-XX.md  → PHASE + TASK nodes (category="tasks")

Each entry is uniquely identified by a deterministic hash so re-parsing the
same log is safe (idempotent upsert).
"""

from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .models import NeuralNode, NeuralEdge, NodeType, EdgeType, SummaryMode

if TYPE_CHECKING:
    from .storage import Storage

# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_id(parts: str) -> str:
    return hashlib.sha256(parts.encode()).hexdigest()[:12]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _find_code_nodes_for_file(storage: "Storage", file_path: str) -> list[NeuralNode]:
    """Return code nodes whose file_path matches the given path (partial match)."""
    if not file_path:
        return []
    # Normalize: strip leading path separators, try basename match
    target = file_path.strip().lstrip("/\\")
    all_nodes = storage.get_all_nodes()
    matches = []
    for n in all_nodes:
        if n.category == "codebase" and (
            n.file_path == target
            or n.file_path.endswith(target)
            or target.endswith(n.file_path)
            or os.path.basename(n.file_path) == os.path.basename(target)
        ):
            matches.append(n)
    # Return at most 3 best matches (prefer exact path, then basename)
    exact = [n for n in matches if n.file_path == target or n.file_path.endswith(target)]
    return (exact or matches)[:3]


# ── Gotchas parser → BUG nodes ─────────────────────────────────────────────────

# em-dash (U+2014) is used in these log headings
_GOTCHA_HEADING = re.compile(r"^## (\d{4}-\d{2}-\d{2}) \u2014 (.+)$", re.MULTILINE)
_BOLD_KV = re.compile(r"\*\*(\w[\w\s]*?)\*\*:\s*(.+?)(?=\n\*\*|\n\n|$)", re.DOTALL)
_BACKTICK_FILE = re.compile(r"`([^`]+\.py)(?::[:\d\-]+)?`")


def parse_gotchas(
    content: str,
    storage: Optional["Storage"] = None,
    source_file: str = "context-log-gotchas.md",
) -> list[tuple[NeuralNode, list[NeuralEdge]]]:
    """Parse a gotchas log into BUG nodes.

    Each `## YYYY-MM-DD — title` entry becomes one BUG node.
    If storage is provided, creates RELATES_TO edges to matching code nodes.
    """
    results: list[tuple[NeuralNode, list[NeuralEdge]]] = []

    # Split into entries by heading
    entry_starts = [(m.start(), m.group(1), m.group(2)) for m in _GOTCHA_HEADING.finditer(content)]

    for i, (start, date_str, title) in enumerate(entry_starts):
        end = entry_starts[i + 1][0] if i + 1 < len(entry_starts) else len(content)
        body = content[start:end]

        # Extract bold key-values
        kv: dict[str, str] = {}
        for m in _BOLD_KV.finditer(body):
            kv[m.group(1).strip().lower()] = m.group(2).strip()

        # Extract linked file from **File**: `path`
        linked_file = ""
        file_match = _BACKTICK_FILE.search(kv.get("file", ""))
        if file_match:
            linked_file = file_match.group(1)

        root_cause = kv.get("root cause", "")
        fix_desc = kv.get("fix", "")

        summary_short = f"[{date_str}] {title}"
        summary_detailed = "\n".join(filter(None, [
            f"Root cause: {root_cause}" if root_cause else "",
            f"Fix: {fix_desc}" if fix_desc else "",
        ]))

        node_id = _make_id(f"bug::{source_file}::{date_str}::{title}")

        node = NeuralNode(
            id=node_id,
            name=f"bug/{date_str}: {title[:60]}",
            node_type=NodeType.BUG,
            file_path=linked_file or source_file,
            line_start=0,
            line_end=0,
            summary_short=summary_short[:200],
            summary_detailed=summary_detailed[:1000],
            summary_mode=SummaryMode.HEURISTIC,
            category="bugs",
            severity="medium",
            bug_status="fixed" if fix_desc else "open",
            content_hash=_make_id(body),
        )

        edges: list[NeuralEdge] = []
        if storage and linked_file:
            for code_node in _find_code_nodes_for_file(storage, linked_file):
                edges.append(NeuralEdge(
                    source_id=node_id,
                    target_id=code_node.id,
                    edge_type=EdgeType.RELATES_TO,
                    context=f"Bug from {date_str}",
                    weight=0.8,
                ))

        results.append((node, edges))

    return results


# ── Tasks parser → PHASE + TASK nodes ─────────────────────────────────────────

_H1 = re.compile(r"^# (.+)$", re.MULTILINE)
_TASK_HEADING = re.compile(r"^## (Fix \d+|Task \d+|Step \d+) \u2014 (.+)$", re.MULTILINE)
_TASK_HEADING_GENERIC = re.compile(r"^## (.+?)(?:\s*\u2014\s*(.+))?$", re.MULTILINE)
_STATUS_LINE = re.compile(r"\*\*Status\*\*:\s*\[([xX ])\]\s*(.+?)(?:\n|$)")
_FILE_REF = re.compile(r"\*\*File\*\*:\s*`([^`]+\.py)`(?:\s+lines?\s+(\d+)[–\-]+(\d+))?")
_OVERALL_STATUS = re.compile(r"##\s+Status:\s*(COMPLETE|IN PROGRESS|PENDING)", re.IGNORECASE)


def parse_tasks(
    content: str,
    storage: Optional["Storage"] = None,
    source_file: str = "context-log-tasks.md",
) -> list[tuple[NeuralNode, list[NeuralEdge]]]:
    """Parse a tasks log into PHASE + TASK nodes.

    The H1 heading becomes a PHASE node; each `## Fix N — title` entry
    becomes a TASK node with PHASE_CONTAINS + RELATES_TO edges.
    If storage is provided, creates RELATES_TO edges to matching code nodes.
    """
    results: list[tuple[NeuralNode, list[NeuralEdge]]] = []

    # Overall phase name from H1
    h1_match = _H1.search(content)
    phase_title = h1_match.group(1).strip() if h1_match else source_file
    phase_id = _make_id(f"phase::{source_file}::{phase_title}")

    # Determine overall phase status
    status_match = _OVERALL_STATUS.search(content)
    phase_task_status = "done" if status_match and "COMPLETE" in status_match.group(1).upper() else "in_progress"

    phase_node = NeuralNode(
        id=phase_id,
        name=f"phase: {phase_title[:80]}",
        node_type=NodeType.PHASE,
        file_path=source_file,
        line_start=0,
        line_end=0,
        summary_short=f"Phase: {phase_title}"[:200],
        summary_mode=SummaryMode.HEURISTIC,
        category="tasks",
        task_status=phase_task_status,
        content_hash=_make_id(content[:200]),
    )
    results.append((phase_node, []))

    # Find task entries — try structured heading first, then any H2
    task_matches = list(_TASK_HEADING.finditer(content))
    if not task_matches:
        # Fallback: any H2 heading (skip the Status line)
        task_matches = [m for m in _TASK_HEADING_GENERIC.finditer(content)
                        if not m.group(0).startswith("## Status")]

    for i, m in enumerate(task_matches):
        start = m.start()
        end = task_matches[i + 1].start() if i + 1 < len(task_matches) else len(content)
        body = content[start:end]

        # Task title
        if m.lastindex and m.lastindex >= 2 and m.group(2):
            task_title = m.group(2).strip()
        else:
            task_title = m.group(1).strip()

        # Status
        status_m = _STATUS_LINE.search(body)
        if status_m:
            done = status_m.group(1).lower() == "x"
            task_status = "done" if done else "pending"
        else:
            task_status = "pending"

        # File reference
        file_m = _FILE_REF.search(body)
        linked_file = file_m.group(1) if file_m else ""
        line_start = int(file_m.group(2)) if file_m and file_m.group(2) else 0
        line_end = int(file_m.group(3)) if file_m and file_m.group(3) else 0

        # Summary from body (trim code blocks)
        clean_body = re.sub(r"```[\s\S]*?```", "", body).strip()
        summary_short = f"{task_title}"[:200]
        summary_detailed = clean_body[:600]

        task_id = _make_id(f"task::{source_file}::{task_title}")

        task_node = NeuralNode(
            id=task_id,
            name=f"task: {task_title[:80]}",
            node_type=NodeType.TASK,
            file_path=linked_file or source_file,
            line_start=line_start,
            line_end=line_end,
            summary_short=summary_short,
            summary_detailed=summary_detailed,
            summary_mode=SummaryMode.HEURISTIC,
            category="tasks",
            task_status=task_status,
            content_hash=_make_id(body),
        )

        edges: list[NeuralEdge] = [
            NeuralEdge(
                source_id=phase_id,
                target_id=task_id,
                edge_type=EdgeType.PHASE_CONTAINS,
                context=f"Phase: {phase_title[:60]}",
                weight=1.0,
            )
        ]

        if storage and linked_file:
            for code_node in _find_code_nodes_for_file(storage, linked_file):
                edges.append(NeuralEdge(
                    source_id=task_id,
                    target_id=code_node.id,
                    edge_type=EdgeType.RELATES_TO,
                    context=f"Task: {task_title[:60]}",
                    weight=0.7,
                ))

        results.append((task_node, edges))

    return results


# ── Discover and import all context logs ───────────────────────────────────────

def import_context_logs(
    storage: "Storage",
    project_root: str = ".",
    force: bool = False,
) -> dict:
    """Scan for .claude/ context log files and import them into the graph.

    Uses file mtime via file_hashes table to avoid re-parsing unchanged logs.
    Returns stats: {bugs_imported, tasks_imported, phases_imported, files_scanned}.
    """
    import os
    stats = {"bugs_imported": 0, "tasks_imported": 0, "phases_imported": 0, "files_scanned": 0}

    claude_dir = Path(project_root) / ".claude"
    if not claude_dir.exists():
        return stats

    # Gather candidate log files
    log_files: list[Path] = []
    for pattern in ["context-log-gotchas.md", "context-log-tasks-*.md"]:
        log_files.extend(claude_dir.glob(pattern))

    for log_path in sorted(log_files):
        stats["files_scanned"] += 1
        rel_path = str(log_path.relative_to(project_root))

        # Mtime-based skip
        try:
            mtime = str(int(log_path.stat().st_mtime))
        except OSError:
            continue

        if not force:
            stored_hash = storage.get_file_hash(rel_path)
            if stored_hash == mtime:
                continue  # File unchanged since last import

        try:
            content = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        is_gotchas = "gotchas" in log_path.name
        entries = (
            parse_gotchas(content, storage=storage, source_file=rel_path)
            if is_gotchas
            else parse_tasks(content, storage=storage, source_file=rel_path)
        )

        for node, edges in entries:
            storage.upsert_node(node)
            for edge in edges:
                storage.upsert_edge(edge)
            if node.node_type == NodeType.BUG:
                stats["bugs_imported"] += 1
            elif node.node_type == NodeType.PHASE:
                stats["phases_imported"] += 1
            elif node.node_type in (NodeType.TASK, NodeType.SUBTASK):
                stats["tasks_imported"] += 1

        # Store mtime so we don't re-parse next time
        storage.save_file_hash(rel_path, mtime, _now())

    return stats
