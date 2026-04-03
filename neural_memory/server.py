"""Neural Memory MCP Server — exposes tools to Claude Code."""

from __future__ import annotations

import json
import os

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

from .agent import check_staleness, format_agent_report
from .config import load_config, save_config, NeuralConfig, RedactionConfig, get_memory_dir
from .context_parser import parse_gotchas, parse_tasks
from .embeddings import semantic_search, is_available as embeddings_available
from .visualize import generate_hierarchy_html, generate_vector_space_html
from .dashboard import generate_dashboard_html
from .graph import (
    get_neighborhood, format_neighborhood, format_node_summary,
    trace_call_chain, compute_importance
)
from .indexer import full_index, incremental_update
from .models import IndexMode, NodeType, NeuralNode, NeuralEdge, EdgeType, SummaryMode
from .storage import Storage

mcp = FastMCP("neural_memory_mcp")


# ── Input Models ──

class IndexInput(BaseModel):
    """Input for full indexing."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    project_root: str = Field(default=".", description="Project root directory path")
    mode: Optional[str] = Field(default=None, description="Index mode: 'ast_only', 'api_only', or 'both'. Uses config default if not set.")


class UpdateInput(BaseModel):
    """Input for incremental update."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    project_root: str = Field(default=".", description="Project root directory path")


class QueryInput(BaseModel):
    """Input for searching the neural graph."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    query: str = Field(..., description="Search term — function name, class name, or keyword", min_length=1)
    project_root: str = Field(default=".", description="Project root directory path")
    limit: int = Field(default=10, description="Max results to return", ge=1, le=50)


class InspectInput(BaseModel):
    """Input for deep-inspecting a node."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    node_id: Optional[str] = Field(default=None, description="Node ID to inspect (from query results)")
    node_name: Optional[str] = Field(default=None, description="Node name to search and inspect")
    project_root: str = Field(default=".", description="Project root directory path")
    show_code: bool = Field(default=False, description="Include raw source code in output")
    trace_calls: bool = Field(default=False, description="Show call chain (who calls this, what it calls)")


class StatusInput(BaseModel):
    """Input for status check."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    project_root: str = Field(default=".", description="Project root directory path")


class ConfigInput(BaseModel):
    """Input for configuration changes."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    project_root: str = Field(default=".", description="Project root directory path")
    action: str = Field(default="view", description="Action: 'view', 'set_mode', 'add_exclude', 'add_redaction_pattern', 'set_staleness_threshold'")
    value: Optional[str] = Field(default=None, description="Value for the action")


class VisualizeInput(BaseModel):
    """Input for visualization generation."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    project_root: str = Field(default=".", description="Project root directory path")
    mode: str = Field(
        default="both",
        description="Which visualization to generate: 'hierarchy', 'vectors', or 'both'",
    )
    dimensions: int = Field(
        default=2,
        description="Dimensions for vector space view: 2 or 3",
        ge=2, le=3,
    )
    color_by: str = Field(
        default="node_type",
        description="Color grouping for vector view: 'node_type' or 'file'",
    )


# ── Tools ──

@mcp.tool(
    name="neural_index",
    annotations={
        "title": "Neural Memory — Full Index",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def neural_index(params: IndexInput) -> str:
    """Build the complete neural memory graph for a codebase.

    Parses all Python files, extracts functions/classes/modules, builds a
    knowledge graph with call relationships, computes importance scores,
    redacts sensitive content, and optionally generates AI summaries.

    Use this on first run or when you want a fresh re-index.
    """
    config = load_config(params.project_root)
    if params.mode:
        config.index_mode = IndexMode(params.mode)
        save_config(config)

    stats = full_index(config, params.project_root)

    lines = [
        "# Neural Memory — Full Index Complete",
        "",
        f"**Files processed**: {stats['files_processed']}",
        f"**Files skipped**: {stats['files_skipped']}",
        f"**Nodes created**: {stats['nodes_created']}",
        f"**Edges created**: {stats['edges_created']}",
        f"**Redactions applied**: {stats['redactions']}",
        f"**Embeddings computed**: {stats.get('embeddings_computed', 0)}",
        f"**Index mode**: {config.index_mode.value}",
    ]

    if stats["errors"]:
        lines.append(f"\n**Errors** ({len(stats['errors'])}):")
        for err in stats["errors"][:5]:
            lines.append(f"  - {err}")

    return "\n".join(lines)


@mcp.tool(
    name="neural_update",
    annotations={
        "title": "Neural Memory — Incremental Update",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def neural_update(params: UpdateInput) -> str:
    """Incrementally update neural memory based on git changes and file modifications.

    Only re-processes files that have changed since the last index,
    saving time and API tokens.
    """
    stats = incremental_update(project_root=params.project_root)

    lines = [
        "# Neural Memory — Incremental Update Complete",
        "",
        f"**Files updated**: {stats['files_updated']}",
        f"**Files added**: {stats['files_added']}",
        f"**Files removed**: {stats['files_removed']}",
        f"**Nodes updated**: {stats['nodes_updated']}",
    ]

    if stats["errors"]:
        lines.append(f"\n**Errors** ({len(stats['errors'])}):")
        for err in stats["errors"][:5]:
            lines.append(f"  - {err}")

    return "\n".join(lines)


@mcp.tool(
    name="neural_query",
    annotations={
        "title": "Neural Memory — Search",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def neural_query(params: QueryInput) -> str:
    """Search the neural memory graph for functions, classes, modules, or concepts.

    Uses semantic vector search (if embeddings are built) for concept-level
    matching, with graph-guided branch expansion to surface connected nodes.
    Falls back to substring search if embeddings are unavailable.

    Returns layered summaries — short overview first, with the ability to
    dig deeper using neural_inspect on any result.
    """
    with Storage(params.project_root) as storage:
        sem_results = semantic_search(storage, params.query, limit=params.limit)

        if sem_results:
            search_mode = "semantic"
            lines = [
                f"# Neural Query: '{params.query}'",
                f"Found {len(sem_results)} result(s) _(semantic + graph search)_:",
                "",
            ]
            for i, r in enumerate(sem_results, 1):
                node = r.node
                lines.append(f"{i}. {format_node_summary(node, 'short')}")
                lines.append(
                    f"   ID: `{node.id}` | File: {node.file_path}:{node.line_start}"
                    f" | score: {r.score:.2f} [{r.match_type}] | {r.connections_summary}"
                )
                lines.append("")
        else:
            # Fallback: substring LIKE search
            nodes = storage.search_nodes(params.query, limit=params.limit)
            if not nodes:
                hint = (
                    " Run `/neural-index` to build the index."
                    if not storage.get_embedding_meta()
                    else ""
                )
                return (
                    f"No results found for '{params.query}'.{hint}"
                )
            search_mode = "substring"
            lines = [
                f"# Neural Query: '{params.query}'",
                f"Found {len(nodes)} result(s) _(substring search — run `/neural-index` with vectors extra for semantic search)_:",
                "",
            ]
            for i, node in enumerate(nodes, 1):
                lines.append(f"{i}. {format_node_summary(node, 'short')}")
                lines.append(f"   ID: `{node.id}` | File: {node.file_path}:{node.line_start}")
                lines.append("")

    lines.append("---")
    lines.append("Use `neural_inspect` with a node ID or name to see full details, connections, and source code.")

    return "\n".join(lines)


@mcp.tool(
    name="neural_inspect",
    annotations={
        "title": "Neural Memory — Deep Inspect",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def neural_inspect(params: InspectInput) -> str:
    """Deep-dive into a specific neural node — see its full summary,
    who calls it, what it calls, sibling functions, parent module/class,
    and optionally the raw source code.

    This is the "one step deeper" in the neural graph.
    """
    with Storage(params.project_root) as storage:
        node = None

        if params.node_id:
            node = storage.get_node(params.node_id)
        elif params.node_name:
            results = storage.search_nodes(params.node_name, limit=1)
            node = results[0] if results else None

        if not node:
            return f"Node not found. Use `neural_query` to search first."

        # Get neighborhood
        neighborhood = get_neighborhood(storage, node.id)
        output = format_neighborhood(neighborhood)

        # Optionally trace calls
        if params.trace_calls:
            output += "\n\n## Call Chain (Upstream — who calls this?)\n"
            up_chains = trace_call_chain(storage, node.id, direction="up", max_depth=4)
            if up_chains:
                for chain in up_chains[:3]:
                    chain_str = " → ".join(f"{n.name}" for n in chain)
                    output += f"  {chain_str}\n"
            else:
                output += "  (no upstream callers found)\n"

            output += "\n## Call Chain (Downstream — what does this call?)\n"
            down_chains = trace_call_chain(storage, node.id, direction="down", max_depth=4)
            if down_chains:
                for chain in down_chains[:3]:
                    chain_str = " → ".join(f"{n.name}" for n in chain)
                    output += f"  {chain_str}\n"
            else:
                output += "  (no downstream calls found)\n"

        # Optionally include source
        if params.show_code and node.raw_code:
            output += f"\n\n## Source Code\n```python\n{node.raw_code}\n```"

    return output


@mcp.tool(
    name="neural_status",
    annotations={
        "title": "Neural Memory — Status",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def neural_status(params: StatusInput) -> str:
    """Check the health and staleness of neural memory.

    Shows index stats, how far behind the index is, and suggests
    actions if needed.
    """
    report = check_staleness(params.project_root)
    output = format_agent_report(report)

    if report.details:
        output += "\n\n### Index Details"
        for k, v in report.details.items():
            output += f"\n  {k}: {v}"

    return output


@mcp.tool(
    name="neural_config",
    annotations={
        "title": "Neural Memory — Configuration",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def neural_config(params: ConfigInput) -> str:
    """View or modify neural memory configuration.

    Actions:
    - view: Show current config
    - set_mode: Change index mode (ast_only, api_only, both)
    - add_exclude: Add a file exclusion pattern
    - add_redaction_pattern: Add a custom redaction regex
    - set_staleness_threshold: Set commits-behind threshold for staleness warnings
    """
    config = load_config(params.project_root)

    if params.action == "view":
        return json.dumps(config.to_dict(), indent=2)

    elif params.action == "set_mode" and params.value:
        try:
            config.index_mode = IndexMode(params.value)
            save_config(config)
            return f"Index mode set to: **{config.index_mode.value}**"
        except ValueError:
            return f"Invalid mode: {params.value}. Use: ast_only, api_only, both"

    elif params.action == "add_exclude" and params.value:
        config.exclude_patterns.append(params.value)
        save_config(config)
        return f"Added exclusion pattern: `{params.value}`"

    elif params.action == "add_redaction_pattern" and params.value:
        config.redaction.custom_patterns.append(params.value)
        save_config(config)
        return f"Added custom redaction pattern: `{params.value}`"

    elif params.action == "set_staleness_threshold" and params.value:
        try:
            config.staleness_threshold = int(params.value)
            save_config(config)
            return f"Staleness threshold set to: **{config.staleness_threshold}** commits"
        except ValueError:
            return f"Invalid threshold: {params.value}. Must be an integer."

    else:
        return "Unknown action or missing value. Actions: view, set_mode, add_exclude, add_redaction_pattern, set_staleness_threshold"


@mcp.tool(
    name="neural_visualize",
    annotations={
        "title": "Neural Memory — Visualize",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def neural_visualize(params: VisualizeInput) -> str:
    """Generate interactive HTML visualizations of the neural memory graph.

    Two views available:
    - hierarchy: treemap of module→class→function containment, sized by importance
    - vectors: PCA scatter of composite embeddings — semantic similarity clusters

    Output files are written to .neural-memory/ and can be opened in any browser.
    Requires: pip install neural-memory[viz]
    """
    memory_dir = get_memory_dir(params.project_root)
    outputs = []

    with Storage(params.project_root) as storage:
        if params.mode in ("hierarchy", "both"):
            out_path = str(memory_dir / "viz_hierarchy.html")
            result = generate_hierarchy_html(storage, out_path)
            outputs.append(f"**Hierarchy view**: {result}")

        if params.mode in ("vectors", "both"):
            out_path = str(memory_dir / "viz_vectors.html")
            result = generate_vector_space_html(
                storage, out_path,
                dimensions=params.dimensions,
                color_by=params.color_by,
            )
            outputs.append(f"**Vector space view** ({params.dimensions}D): {result}")

    if not outputs:
        return "Invalid mode. Use: hierarchy, vectors, or both"

    lines = ["# Neural Memory — Visualization Generated", ""]
    lines.extend(outputs)
    lines.append("")
    lines.append("Open the HTML files in your browser for interactive exploration.")
    return "\n".join(lines)


class DashboardInput(BaseModel):
    """Input for the interactive dashboard tool."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    project_root: str = Field(default=".", description="Project root directory")
    output_path: Optional[str] = Field(
        default=None,
        description="Output HTML file path. Defaults to .neural-memory/dashboard.html",
    )


@mcp.tool(
    name="neural_visualize_dashboard",
    annotations={
        "title": "Neural Memory — Interactive Dashboard",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def neural_visualize_dashboard(params: DashboardInput) -> str:
    """Generate the interactive D3 knowledge-graph dashboard.

    Produces a single self-contained HTML file with:
    - Hierarchy treemap (module → class → function, sized by importance)
    - Vector space scatter (PCA-projected embeddings, nodes clustered by semantics)
    - Force-directed graph (nodes + edges, drag/zoom/pan)
    - Sidebar filters: category (codebase/bugs/tasks), node type, importance, status, search
    - Click any node to inspect its full detail panel

    Output is written to .neural-memory/dashboard.html (or the path you specify).
    Open the file in any browser — no server required.
    """
    memory_dir = get_memory_dir(params.project_root)
    out_path = params.output_path or str(memory_dir / "dashboard.html")

    with Storage(params.project_root) as storage:
        generate_dashboard_html(
            storage,
            output_path=out_path,
            project_root=params.project_root,
        )

    return (
        "# Neural Memory — Dashboard Generated\n\n"
        f"**File**: `{out_path}`\n\n"
        "Open in your browser for the interactive 3-tab knowledge graph view.\n\n"
        "**Views**: Hierarchy treemap | Vector space | Force graph\n"
        "**Filters**: Category · Node type · Importance · Status · Search"
    )


class ServeInput(BaseModel):
    """Input for neural_serve."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    project_root: str = Field(default=".", description="Project root directory")
    port: int = Field(default=7891, description="Port to listen on (default: 7891)")
    open_browser: bool = Field(default=True, description="Open dashboard in browser automatically")
    regenerate: bool = Field(default=True, description="Regenerate dashboard HTML before serving")


@mcp.tool(
    name="neural_serve",
    annotations={
        "title": "Neural Memory — Start Dashboard Server",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def neural_serve(params: ServeInput) -> str:
    """Start a local HTTP server and open the neural memory dashboard in your browser.

    Regenerates the dashboard HTML from the current index, then serves it at
    http://localhost:PORT/dashboard.html and opens the URL in the default browser.

    Idempotent: calling again while the server is already running on the same port
    just re-opens the browser tab. Call neural_stop_serve to shut it down.
    """
    from .serve import start_server, is_running, get_url

    memory_dir = get_memory_dir(params.project_root)
    out_path = str(memory_dir / "dashboard.html")

    if params.regenerate:
        with Storage(params.project_root) as storage:
            generate_dashboard_html(
                storage,
                output_path=out_path,
                project_root=params.project_root,
            )

    url = start_server(
        project_root=params.project_root,
        port=params.port,
        open_browser=params.open_browser,
    )

    return (
        "# Neural Memory Dashboard\n\n"
        f"**URL**: <{url}>\n\n"
        f"Serving `.neural-memory/` on port **{params.port}**.\n\n"
        "The dashboard opens automatically in your default browser.\n\n"
        "**Views**: Hierarchy treemap · Semantic radial tree · Force-directed graph\n"
        "**Filters**: Category · Node type · Importance · Status · Search\n\n"
        "Run `neural_stop_serve` to shut down the server."
    )


class StopServeInput(BaseModel):
    """Input for neural_stop_serve (no required fields)."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


@mcp.tool(
    name="neural_stop_serve",
    annotations={
        "title": "Neural Memory — Stop Dashboard Server",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def neural_stop_serve(params: StopServeInput) -> str:
    """Stop the running neural memory dashboard HTTP server."""
    from .serve import stop_server, is_running, get_url

    url = get_url()
    stopped = stop_server()

    if stopped:
        return f"Dashboard server stopped (was running at {url})."
    return "No dashboard server is currently running."


class AddBugInput(BaseModel):
    """Input for manually adding a bug node."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    description: str = Field(..., description="Bug description / what went wrong", min_length=3)
    severity: str = Field(default="medium", description="Severity: low / medium / high / critical")
    file_path: Optional[str] = Field(default=None, description="Source file this bug relates to")
    line_start: Optional[int] = Field(default=None, description="Starting line number")
    line_end: Optional[int] = Field(default=None, description="Ending line number")
    root_cause: str = Field(default="", description="Root cause description")
    fix_description: str = Field(default="", description="How it was / should be fixed")
    project_root: str = Field(default=".", description="Project root directory path")


class AddTaskInput(BaseModel):
    """Input for manually adding a task node."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    title: str = Field(..., description="Task title", min_length=3)
    phase_name: Optional[str] = Field(default=None, description="Parent phase name (creates phase if new)")
    priority: str = Field(default="medium", description="Priority: low / medium / high")
    task_status: str = Field(default="pending", description="Status: pending / in_progress / done")
    related_files: list[str] = Field(default_factory=list, description="Source files this task relates to")
    project_root: str = Field(default=".", description="Project root directory path")


@mcp.tool(
    name="neural_add_bug",
    annotations={
        "title": "Neural Memory — Add Bug",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
async def neural_add_bug(params: AddBugInput) -> str:
    """Manually log a bug as a neural graph node connected to the affected code.

    The bug node is linked to the specified file via a RELATES_TO edge and
    becomes searchable alongside code nodes in neural_query.
    """
    import hashlib
    from datetime import datetime, timezone

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    node_id = hashlib.sha256(
        f"bug::manual::{now_str}::{params.description}".encode()
    ).hexdigest()[:12]

    summary_short = f"[{now_str}] {params.description}"[:200]
    summary_detailed = "\n".join(filter(None, [
        f"Root cause: {params.root_cause}" if params.root_cause else "",
        f"Fix: {params.fix_description}" if params.fix_description else "",
    ]))

    node = NeuralNode(
        id=node_id,
        name=f"bug/{now_str}: {params.description[:60]}",
        node_type=NodeType.BUG,
        file_path=params.file_path or "__manual__",
        line_start=params.line_start or 0,
        line_end=params.line_end or 0,
        summary_short=summary_short,
        summary_detailed=summary_detailed,
        summary_mode=SummaryMode.HEURISTIC,
        category="bugs",
        severity=params.severity,
        bug_status="open",
        content_hash=node_id,
    )

    with Storage(params.project_root) as storage:
        storage.upsert_node(node)

        edges_created = 0
        if params.file_path:
            # Link to code nodes in that file
            code_nodes = storage.get_nodes_by_file(params.file_path)
            if not code_nodes:
                # Try basename match
                from .context_parser import _find_code_nodes_for_file
                code_nodes = _find_code_nodes_for_file(storage, params.file_path)
            for cn in code_nodes[:3]:
                storage.upsert_edge(NeuralEdge(
                    source_id=node_id,
                    target_id=cn.id,
                    edge_type=EdgeType.RELATES_TO,
                    context=f"Bug: {params.description[:60]}",
                    weight=0.8,
                ))
                edges_created += 1

    return (
        f"# Bug Logged\n\n"
        f"**ID**: `{node_id}`\n"
        f"**Description**: {params.description}\n"
        f"**Severity**: {params.severity}\n"
        f"**File**: {params.file_path or '(unlinked)'}\n"
        f"**Edges created**: {edges_created}\n\n"
        f"Use `neural_query` to find this bug or `neural_inspect` with ID `{node_id}`."
    )


@mcp.tool(
    name="neural_add_task",
    annotations={
        "title": "Neural Memory — Add Task",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
async def neural_add_task(params: AddTaskInput) -> str:
    """Manually log a task as a neural graph node, optionally under a phase.

    Creates PHASE_CONTAINS edges if a phase is specified, and RELATES_TO
    edges for any related files. Tasks appear in neural_query results.
    """
    import hashlib
    from datetime import datetime, timezone

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    task_id = hashlib.sha256(
        f"task::manual::{params.title}".encode()
    ).hexdigest()[:12]

    task_node = NeuralNode(
        id=task_id,
        name=f"task: {params.title[:80]}",
        node_type=NodeType.TASK,
        file_path="__manual__",
        line_start=0,
        line_end=0,
        summary_short=params.title[:200],
        summary_mode=SummaryMode.HEURISTIC,
        category="tasks",
        task_status=params.task_status,
        priority=params.priority,
        content_hash=task_id,
    )

    with Storage(params.project_root) as storage:
        storage.upsert_node(task_node)
        edges_created = 0

        # Link to parent phase
        if params.phase_name:
            phase_id = hashlib.sha256(
                f"phase::manual::{params.phase_name}".encode()
            ).hexdigest()[:12]
            # Create phase node if it doesn't exist
            if not storage.get_node(phase_id):
                phase_node = NeuralNode(
                    id=phase_id,
                    name=f"phase: {params.phase_name[:80]}",
                    node_type=NodeType.PHASE,
                    file_path="__manual__",
                    line_start=0,
                    line_end=0,
                    summary_short=f"Phase: {params.phase_name}"[:200],
                    summary_mode=SummaryMode.HEURISTIC,
                    category="tasks",
                    task_status="in_progress",
                    content_hash=phase_id,
                )
                storage.upsert_node(phase_node)
            storage.upsert_edge(NeuralEdge(
                source_id=phase_id,
                target_id=task_id,
                edge_type=EdgeType.PHASE_CONTAINS,
                context=f"Phase: {params.phase_name[:60]}",
                weight=1.0,
            ))
            edges_created += 1

        # Link to related files
        from .context_parser import _find_code_nodes_for_file
        for rel_file in params.related_files[:5]:
            for cn in _find_code_nodes_for_file(storage, rel_file)[:2]:
                storage.upsert_edge(NeuralEdge(
                    source_id=task_id,
                    target_id=cn.id,
                    edge_type=EdgeType.RELATES_TO,
                    context=f"Task: {params.title[:60]}",
                    weight=0.7,
                ))
                edges_created += 1

    return (
        f"# Task Logged\n\n"
        f"**ID**: `{task_id}`\n"
        f"**Title**: {params.title}\n"
        f"**Status**: {params.task_status} | **Priority**: {params.priority}\n"
        f"**Phase**: {params.phase_name or '(none)'}\n"
        f"**Edges created**: {edges_created}\n\n"
        f"Use `neural_query` to find this task or `neural_inspect` with ID `{task_id}`."
    )


# Entry point
if __name__ == "__main__":
    mcp.run()
