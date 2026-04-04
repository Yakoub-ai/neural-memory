"""Neural Memory MCP Server — exposes tools to Claude Code."""

from __future__ import annotations

import json
import logging
import os
import sys

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

mcp = FastMCP("neural_memory_mcp")

# Ensure library logging never writes to stdout (would corrupt stdio MCP transport)
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)


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
    language: Optional[str] = Field(default=None, description="Filter results to a specific language (e.g. 'python', 'typescript', 'rust')")


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
    from .config import load_config, save_config
    from .indexer import full_index
    from .models import IndexMode
    config = load_config(params.project_root)
    if params.mode:
        config.index_mode = IndexMode(params.mode)
        save_config(config)

    stats = full_index(config, params.project_root)

    # Auto-install the agent hook into the project's CLAUDE.md so Claude
    # checks index staleness automatically on every future session.
    try:
        from pathlib import Path
        from .cli import _append_hook
        _append_hook(Path(params.project_root).resolve() / "CLAUDE.md")
    except Exception:
        pass  # non-fatal — indexing succeeded regardless

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
    from .indexer import incremental_update
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
    from .embeddings import semantic_search
    from .graph import format_node_summary
    from .storage import Storage
    with Storage(params.project_root) as storage:
        sem_results = semantic_search(storage, params.query, limit=params.limit * 3 if params.language else params.limit)

        # Apply language filter
        if params.language and sem_results:
            sem_results = [r for r in sem_results if r.node.language == params.language]
            sem_results = sem_results[:params.limit]

        if sem_results:
            lang_label = f" [{params.language}]" if params.language else ""
            search_mode = "semantic"
            lines = [
                f"# Neural Query: '{params.query}'{lang_label}",
                f"Found {len(sem_results)} result(s) _(semantic + graph search)_:",
                "",
            ]
            for i, r in enumerate(sem_results, 1):
                node = r.node
                lang_tag = f" `{node.language}`" if node.language else ""
                lines.append(f"{i}. {format_node_summary(node, 'short')}{lang_tag}")
                lines.append(
                    f"   ID: `{node.id}` | File: {node.file_path}:{node.line_start}"
                    f" | score: {r.score:.2f} [{r.match_type}] | {r.connections_summary}"
                )
                lines.append("")
        else:
            # Fallback: substring LIKE search
            nodes = storage.search_nodes(params.query, limit=params.limit)
            if params.language and nodes:
                nodes = [n for n in nodes if n.language == params.language]
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
                lang_tag = f" `{node.language}`" if node.language else ""
                lines.append(f"{i}. {format_node_summary(node, 'short')}{lang_tag}")
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
    from .graph import get_neighborhood, format_neighborhood, trace_call_chain
    from .storage import Storage
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
            from .languages import detect_language
            lang = detect_language(node.file_path)
            fence = (lang.code_fence if lang else None) or node.language or "text"
            output += f"\n\n## Source Code\n```{fence}\n{node.raw_code}\n```"

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
    from .agent import check_staleness, format_agent_report
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
    from .config import load_config, save_config
    from .models import IndexMode
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
    from .config import get_memory_dir
    from .storage import Storage
    from .visualize import generate_hierarchy_html, generate_vector_space_html
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
    from .config import get_memory_dir
    from .dashboard import generate_dashboard_html
    from .storage import Storage
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
    from .config import get_memory_dir
    from .dashboard import generate_dashboard_html
    from .serve import start_server, is_running, get_url
    from .storage import Storage

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
    from .models import NeuralNode, NodeType, NeuralEdge, EdgeType, SummaryMode
    from .storage import Storage
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
    from .models import NeuralNode, NodeType, NeuralEdge, EdgeType, SummaryMode
    from .storage import Storage
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


# ── DB Schema Tool ──

class IndexDbInput(BaseModel):
    """Input for live database schema indexing."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    connection_string: str = Field(
        default="auto",
        description="DB connection string (sqlite:///path, postgresql://..., mysql://...) or 'auto' to detect from env/config",
    )
    project_root: str = Field(default=".", description="Project root for auto-detection")
    db_name: Optional[str] = Field(default=None, description="Override database name label")


@mcp.tool(
    name="neural_index_db",
    annotations={
        "title": "Neural Memory — Index Database Schema",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    }
)
async def neural_index_db(params: IndexDbInput) -> str:
    """Index a live database schema into the knowledge graph.

    Connects to a SQLite, PostgreSQL, or MySQL database and introspects its
    schema (tables, columns, foreign keys), storing them as DATABASE/TABLE/COLUMN
    nodes in the knowledge graph.

    Use 'auto' for connection_string to detect DATABASE_URL from environment
    or project .env file.
    """
    from .db.connector import detect_connection_string, fetch_schema
    from .db.schema_indexer import index_db_schema
    from .storage import Storage

    # Resolve connection string
    cs = params.connection_string.strip()
    if cs == "auto":
        cs = detect_connection_string(params.project_root) or ""
        if not cs:
            return (
                "# Neural Memory — DB Index Failed\n\n"
                "Could not auto-detect a database connection string.\n\n"
                "**Tried:** `DATABASE_URL` env var, `.env` file, `docker-compose.yml`\n\n"
                "Please provide an explicit `connection_string` parameter."
            )

    # Derive db_name from connection string if not provided
    db_name = params.db_name
    if not db_name:
        # Use filename for sqlite, or database part for others
        if cs.startswith("sqlite:///"):
            db_name = cs.split("/")[-1].replace(".db", "").replace(".sqlite", "") or "local"
        else:
            db_name = cs.split("/")[-1].split("?")[0] or "database"

    try:
        schemas = fetch_schema(cs)
    except Exception as e:
        return (
            f"# Neural Memory — DB Index Failed\n\n"
            f"**Connection string:** `{cs}`\n\n"
            f"**Error:** {e}"
        )

    if not schemas:
        return (
            f"# Neural Memory — DB Index\n\n"
            f"**Database:** {db_name}\n\n"
            "No tables found in the database."
        )

    with Storage(params.project_root) as storage:
        stats = index_db_schema(storage, schemas, db_name, source="live_db")

    return (
        f"# Neural Memory — DB Schema Indexed\n\n"
        f"**Database:** {db_name}\n"
        f"**Tables indexed:** {stats['tables_indexed']}\n"
        f"**Columns indexed:** {stats['columns_indexed']}\n"
        f"**FK edges created:** {stats['fk_edges']}\n"
        f"**Total edges:** {stats['edges_created']}\n\n"
        f"Use `neural_query` to search the schema, e.g. `neural_query` with query = `{db_name}`."
    )


# ── Documentation Fetching Tool ──

class FetchDocsInput(BaseModel):
    """Input for fetching package documentation."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    package_name: str = Field(
        default="auto",
        description="Package name or 'auto' to fetch for all detected imports",
    )
    registry: Optional[str] = Field(
        default=None,
        description="Registry: 'pypi', 'npm', 'go', 'crates', or None for auto-detect",
    )
    project_root: str = Field(default=".", description="Project root directory path")


@mcp.tool(
    name="neural_fetch_docs",
    annotations={
        "title": "Neural Memory — Fetch Package Docs",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    }
)
async def neural_fetch_docs(params: FetchDocsInput) -> str:
    """Fetch package documentation from registries (PyPI, npm, Go, crates.io) and cache
    it in the knowledge graph.

    Use package_name='auto' to discover all external imports in the indexed codebase
    and fetch their docs in bulk. Or specify a single package name to fetch on demand.
    Supports auto-detection of registry from package naming conventions, or explicit
    registry selection.
    """
    import neural_memory.docs  # triggers registration of all fetchers
    from neural_memory.docs.registry import fetch_docs
    from datetime import datetime, timezone
    from .models import EdgeType
    from .storage import Storage

    project_root = params.project_root or "."

    with Storage(project_root) as storage:
        if params.package_name == "auto":
            # Find all unresolved IMPORTS edges and extract package names
            edges = storage.get_all_edges()
            packages: set[str] = set()
            for edge in edges:
                if edge.edge_type == EdgeType.IMPORTS:
                    if edge.target_id.startswith("__unresolved__"):
                        pkg = (
                            edge.target_id
                            .replace("__unresolved__", "")
                            .split(".")[0]
                            .split("/")[0]
                        )
                        if pkg:
                            packages.add(pkg)

            fetched = []
            failed = []
            for pkg in sorted(packages):
                try:
                    doc = fetch_docs(pkg, params.registry)
                    if doc:
                        storage.upsert_package_doc(
                            doc.package_name, doc.registry,
                            doc.to_storage_dict(), doc.fetched_at,
                        )
                        fetched.append(f"{pkg} ({doc.registry})")
                    else:
                        failed.append(pkg)
                except Exception as e:
                    failed.append(f"{pkg} (error: {e})")

            lines = ["## Documentation Fetch Results\n"]
            lines.append(f"**Fetched**: {len(fetched)} packages")
            if fetched:
                lines.append("\n".join(f"- {p}" for p in fetched))
            if failed:
                lines.append(f"\n**Not found**: {', '.join(failed[:20])}")
            return "\n".join(lines)

        else:
            try:
                doc = fetch_docs(params.package_name, params.registry)
            except Exception as e:
                return (
                    f"Error fetching documentation for `{params.package_name}`: {e}"
                )

            if not doc:
                return (
                    f"No documentation found for `{params.package_name}`. "
                    f"Try specifying the registry with `registry='pypi'` (or 'npm', 'go', 'crates')."
                )

            try:
                storage.upsert_package_doc(
                    doc.package_name, doc.registry,
                    doc.to_storage_dict(), doc.fetched_at,
                )
            except Exception as e:
                return (
                    f"Fetched documentation for `{params.package_name}` but failed to store it: {e}"
                )

            lines = [
                f"## {doc.package_name} ({doc.registry})",
                f"**Version**: {doc.version or 'unknown'}",
                f"**Summary**: {doc.summary or 'No summary available'}",
            ]
            if doc.homepage_url:
                lines.append(f"**Homepage**: {doc.homepage_url}")
            if doc.doc_url:
                lines.append(f"**Docs**: {doc.doc_url}")
            if doc.description and doc.description != doc.summary:
                lines.append(f"\n{doc.description[:1000]}")
            return "\n".join(lines)


# ── Healthcheck Tool ──

class PingInput(BaseModel):
    """Input for healthcheck ping (no fields required)."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


@mcp.tool(
    name="neural_ping",
    annotations={
        "title": "Neural Memory — Ping",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def neural_ping(params: PingInput) -> str:
    """Lightweight healthcheck — confirms the neural memory MCP server is running."""
    from . import __version__
    return f"neural-memory v{__version__} — MCP server is running."


# Entry point
if __name__ == "__main__":
    try:
        mcp.run()
    except Exception as exc:
        print(f"neural-memory MCP server failed: {exc}", file=sys.stderr)
        sys.exit(1)
