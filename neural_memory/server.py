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
    include_archived: bool = Field(default=False, description="Include archived (completed/fixed) bugs and tasks in results")


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



class ContextInput(BaseModel):
    """Input for the compact context tool."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    project_root: str = Field(default=".", description="Project root directory path")
    query_hint: Optional[str] = Field(default=None, description="Optional prompt or keyword to drive semantic pre-fetch of relevant nodes")
    token_budget: int = Field(default=500, description="Approximate token budget for the response", ge=100, le=2000)


class ArchiveInput(BaseModel):
    """Input for manual archive/unarchive operations."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    node_id: str = Field(..., description="Node ID to archive or unarchive")
    action: str = Field(default="archive", description="Action: 'archive' or 'unarchive'")
    project_root: str = Field(default=".", description="Project root directory path")



class ListTasksInput(BaseModel):
    """Input for listing tasks."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    project_root: str = Field(default=".", description="Project root directory path")
    status: Optional[str] = Field(default=None, description="Filter by status: pending / in_progress / testing / done")
    priority: Optional[str] = Field(default=None, description="Filter by priority: low / medium / high")
    include_archived: bool = Field(default=False, description="Include archived (done) tasks")


class UpdateTaskInput(BaseModel):
    """Input for updating a task's status, priority, or related files."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    title_or_id: str = Field(..., description="Task name substring or full node ID to match", min_length=1)
    task_status: Optional[str] = Field(default=None, description="New status: pending / new / in_progress / testing / done")
    priority: Optional[str] = Field(default=None, description="New priority: low / medium / high")
    related_files: list[str] = Field(default_factory=list, description="Additional file paths to link to this task")
    project_root: str = Field(default=".", description="Project root directory path")


def _parse_filter(key: str, query: str) -> tuple[Optional[str], str]:
    """Extract a `key:value` structured filter token from a query string.

    Returns (value, remainder_query). If the key is not present, returns (None, query).
    Example: _parse_filter("type", "type:class auth") -> ("class", "auth")
    """
    import re
    m = re.search(rf'\b{key}:(\S+)', query)
    if m:
        return m.group(1).lower(), (query[:m.start()] + query[m.end():]).strip()
    return None, query


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

    result = "\n".join(lines)

    # One-time RTK suggestion (token compression tool)
    import shutil
    from pathlib import Path
    rtk_flag = Path(params.project_root).resolve() / ".neural-memory" / "rtk_prompted"
    if not shutil.which("rtk") and not rtk_flag.exists():
        try:
            rtk_flag.parent.mkdir(parents=True, exist_ok=True)
            rtk_flag.touch()
            result += (
                "\n\n---\n💡 **Token tip**: RTK is not installed. "
                "RTK compresses command outputs before they reach Claude, saving 60-90% tokens. "
                "Install: `npm i -g rtk` — https://github.com/rtk-ai/rtk"
            )
        except Exception:
            pass

    return result


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

    Supports structured filters in the query string:
      type:function    — filter by node type (function, class, method, module, etc.)
      file:storage     — filter to files containing this substring
      layer:bugs       — filter to a specific layer (codebase, bugs, tasks, insights)
      edge:calls       — filter to nodes with a specific edge type

    Examples:
      "auth"                  — semantic search for anything auth-related
      "type:class auth"       — auth-related classes only
      "file:storage"          — all nodes in storage-related files
      "layer:bugs"            — all tracked bugs
      "edge:calls storage"    — nodes that call storage-related things

    Every response includes a freshness stamp — if indexed files have changed on disk
    since the last index, the stale files are listed so you know the data may be outdated.
    """
    from .embeddings import semantic_search
    from .graph import format_node_summary
    from .storage import Storage

    # Parse structured filters from query
    raw_query = params.query
    filter_type, raw_query = _parse_filter("type", raw_query)
    filter_file, raw_query = _parse_filter("file", raw_query)
    filter_layer, raw_query = _parse_filter("layer", raw_query)
    filter_edge, raw_query = _parse_filter("edge", raw_query)

    # Use cleaned query for search (or full query if no filters extracted)
    search_query = raw_query.strip() or params.query

    with Storage(params.project_root) as storage:
        # Run semantic search (or FTS fallback)
        fetch_limit = params.limit * 4  # over-fetch to allow for filter narrowing
        sem_results = semantic_search(storage, search_query, limit=fetch_limit)

        # Get result nodes (either from semantic search or FTS/LIKE fallback)
        result_nodes_with_scores = []
        if sem_results:
            result_nodes_with_scores = [(r.node, r.score, r.match_type, getattr(r, "connections_summary", "")) for r in sem_results]
        else:
            fallback_nodes = storage.search_nodes(search_query, limit=fetch_limit)
            result_nodes_with_scores = [(n, 0.0, "substring", "") for n in fallback_nodes]

        # Apply structured filters
        if filter_type:
            result_nodes_with_scores = [
                t for t in result_nodes_with_scores
                if t[0].node_type.value == filter_type
            ]
        if filter_file:
            result_nodes_with_scores = [
                t for t in result_nodes_with_scores
                if filter_file in t[0].file_path
            ]
        if filter_layer:
            result_nodes_with_scores = [
                t for t in result_nodes_with_scores
                if t[0].category == filter_layer
            ]
        if filter_edge:
            # Filter to nodes that have at least one edge of this type
            edge_filtered = []
            for t in result_nodes_with_scores:
                node = t[0]
                out_edges = storage.get_edges_from(node.id)
                in_edges = storage.get_edges_to(node.id)
                all_types = {e.edge_type.value for e in out_edges + in_edges}
                if filter_edge in all_types:
                    edge_filtered.append(t)
            result_nodes_with_scores = edge_filtered

        # Apply language + archive filters
        if params.language:
            result_nodes_with_scores = [
                t for t in result_nodes_with_scores if t[0].language == params.language
            ]
        if not params.include_archived:
            result_nodes_with_scores = [
                t for t in result_nodes_with_scores if not t[0].archived
            ]

        result_nodes_with_scores = result_nodes_with_scores[:params.limit]

        if not result_nodes_with_scores:
            hint = (
                " Run `/neural-index` to build the index."
                if not storage.get_embedding_meta()
                else ""
            )
            active_filters = [f"{k}:{v}" for k, v in [
                ("type", filter_type), ("file", filter_file),
                ("layer", filter_layer), ("edge", filter_edge)
            ] if v]
            filter_note = f" (filters: {', '.join(active_filters)})" if active_filters else ""
            return f"No results found for '{params.query}'{filter_note}.{hint}"

        # Freshness check: verify result files haven't changed since indexing
        unique_files = list({t[0].file_path for t in result_nodes_with_scores if t[0].file_path})
        freshness = storage.check_file_freshness(unique_files, params.project_root)
        stale_files = [fp for fp, status in freshness.items() if status == "stale"]
        deleted_files = [fp for fp, status in freshness.items() if status == "deleted"]

        # Build output
        used_semantic = bool(sem_results)
        lang_label = f" [{params.language}]" if params.language else ""
        search_mode_label = "semantic + graph" if used_semantic else "BM25 text"
        lines = [
            f"# Neural Query: '{params.query}'{lang_label}",
            f"Found {len(result_nodes_with_scores)} result(s) _({search_mode_label} search)_:",
            "",
        ]

        for i, (node, score, match_type, conn_summary) in enumerate(result_nodes_with_scores, 1):
            lang_tag = f" `{node.language}`" if node.language else ""
            staleness_tag = " ⚠️stale" if node.file_path in stale_files else ""
            lines.append(f"{i}. {format_node_summary(node, 'short')}{lang_tag}{staleness_tag}")
            score_str = f" | score: {score:.2f} [{match_type}]" if score > 0 else ""
            conn_str = f" | {conn_summary}" if conn_summary else ""
            lines.append(f"   ID: `{node.id}` | File: `{node.file_path}:{node.line_start}`{score_str}{conn_str}")
            lines.append("")

        lines.append("---")

        # Freshness footer
        if stale_files or deleted_files:
            stale_list = ", ".join(f"`{f}`" for f in (stale_files + deleted_files)[:5])
            lines.append(
                f"⚠️ **Freshness warning**: {len(stale_files + deleted_files)} file(s) changed since last index: {stale_list}. "
                f"Run `neural_update` for fresh data."
            )
        else:
            lines.append("✓ All result files verified fresh.")

        lines.append("Use `neural_inspect` with a node ID or name for full details and call chains.")

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

    Automatically refreshes the target file if it has changed on disk since
    the last index, so inspect always returns current data for the specific file.

    This is the "one step deeper" in the neural graph.
    """
    from .graph import get_neighborhood, format_neighborhood, trace_call_chain
    from .storage import Storage

    # First pass: locate the node
    with Storage(params.project_root) as storage:
        node = None
        if params.node_id:
            node = storage.get_node(params.node_id)
        elif params.node_name:
            results = storage.search_nodes(params.node_name, limit=1)
            node = results[0] if results else None

        if not node:
            return "Node not found. Use `neural_query` to search first."

        # Check if the file has changed — if so, micro-update it before inspecting
        if node.file_path and not node.file_path.startswith("__"):
            freshness = storage.check_file_freshness([node.file_path], params.project_root)
            file_status = freshness.get(node.file_path, "fresh")
        else:
            file_status = "fresh"

    freshness_note = ""
    if file_status == "stale":
        # Micro-update the single file so we return fresh AST data
        from .indexer import micro_update
        mu_stats = micro_update(node.file_path, params.project_root)
        if mu_stats.get("error"):
            freshness_note = f"\n> ⚠️ File changed but refresh failed: {mu_stats['error']}\n"
        else:
            freshness_note = f"\n> ✓ File changed — auto-refreshed ({mu_stats['nodes_updated']} nodes updated)\n"
    elif file_status == "deleted":
        freshness_note = "\n> ⚠️ Source file has been deleted — showing last-indexed data\n"

    # Second pass: read the (potentially refreshed) node
    with Storage(params.project_root) as storage:
        if params.node_id:
            node = storage.get_node(params.node_id)
        elif params.node_name:
            results = storage.search_nodes(params.node_name, limit=1)
            node = results[0] if results else None

        if not node:
            return "Node not found after refresh. The file may have been restructured."

        neighborhood = get_neighborhood(storage, node.id)
        output = format_neighborhood(neighborhood)

        if freshness_note:
            output = freshness_note + output

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


class ImpactInput(BaseModel):
    """Input for impact analysis."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    node_name: Optional[str] = Field(default=None, description="Function or class name to analyze")
    node_id: Optional[str] = Field(default=None, description="Node ID to analyze (from query results)")
    project_root: str = Field(default=".", description="Project root directory path")
    max_depth: int = Field(default=3, description="Maximum transitive depth to trace (1-5)", ge=1, le=5)


@mcp.tool(
    name="neural_impact",
    annotations={
        "title": "Neural Memory — Impact Analysis",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def neural_impact(params: ImpactInput) -> str:
    """Analyze the blast radius of changing a function or class.

    Answers: "If I change this, what else might break?"

    Traces all transitive callers, inheritors, and implementors up to max_depth hops.
    Results are grouped by file so you know exactly which files need review/testing.

    This is something Claude cannot efficiently do natively — it would require reading
    dozens of files. The pre-built graph makes this a single sub-second call.

    Examples:
      neural_impact(node_name="Storage")       — what depends on Storage class?
      neural_impact(node_name="incremental_update", max_depth=2)
    """
    from .graph import get_impact_radius, format_impact_report
    from .storage import Storage

    with Storage(params.project_root) as storage:
        node = None
        if params.node_id:
            node = storage.get_node(params.node_id)
        elif params.node_name:
            results = storage.search_nodes(params.node_name, limit=1)
            node = results[0] if results else None

        if not node:
            return (
                f"Node '{params.node_name or params.node_id}' not found. "
                "Use `neural_query` to find the correct name first."
            )

        impact = get_impact_radius(storage, node.id, max_depth=params.max_depth)
        return format_impact_report(impact)


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


class AddInsightInput(BaseModel):
    """Input for saving a technical insight into the knowledge graph."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    content: str = Field(..., description="The insight text — educational point about implementation choices, patterns, or architecture", min_length=10)
    topic: str = Field(..., description="Topic area, e.g. 'authentication', 'database', 'hooks', 'caching', 'storage'", min_length=2)
    related_files: list[str] = Field(default_factory=list, description="File paths this insight relates to (creates RELATES_TO edges to code nodes)")
    project_root: str = Field(default=".", description="Project root directory path")


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
    task_status: str = Field(default="pending", description="Status: new (alias: pending) / pending / in_progress / testing / done")
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
    from .models import NeuralNode, NodeType, NeuralEdge, EdgeType, SummaryMode, TASK_STATUS_ALIASES, VALID_TASK_STATUSES
    from .storage import Storage
    import hashlib
    from datetime import datetime, timezone
    # Normalize status aliases (e.g., "new" → "pending")
    status = TASK_STATUS_ALIASES.get(params.task_status, params.task_status)
    if status not in VALID_TASK_STATUSES:
        return f"Invalid task_status '{params.task_status}'. Valid: {sorted(VALID_TASK_STATUSES)} (or 'new' as alias for 'pending')"

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
        task_status=status,
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
        f"**Status**: {status} | **Priority**: {params.priority}\n"
        f"**Phase**: {params.phase_name or '(none)'}\n"
        f"**Edges created**: {edges_created}\n\n"
        f"Use `neural_query` to find this task or `neural_inspect` with ID `{task_id}`."
    )


# ── Insight Tools ──

@mcp.tool(
    name="neural_add_insight",
    annotations={
        "title": "Neural Memory — Add Insight",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def neural_add_insight(params: AddInsightInput) -> str:
    """Save a technical insight into the knowledge graph.

    Insights capture educational points about implementation choices, architecture
    decisions, and technical patterns discovered during development. They accumulate
    over time and can be synthesized into full documentation via neural_generate_docs.

    Insights are deduplicated by topic + content — re-saving a near-identical insight
    updates the existing one rather than creating a duplicate. Search insights with
    `neural_query` using `layer:insights`, or browse a specific one via `neural_inspect`.
    """
    from .models import NeuralNode, NodeType, NeuralEdge, EdgeType, SummaryMode
    from .storage import Storage
    import hashlib

    topic_normalized = params.topic.strip().lower()
    content_key = params.content.strip().lower()[:100]
    node_id = hashlib.sha256(
        f"insight::{topic_normalized}::{content_key}".encode()
    ).hexdigest()[:12]

    node = NeuralNode(
        id=node_id,
        name=f"insight/{topic_normalized}: {params.content[:60]}",
        node_type=NodeType.INSIGHT,
        file_path="__manual__",
        line_start=0,
        line_end=0,
        summary_short=params.content[:200],
        summary_detailed=params.content,
        summary_mode=SummaryMode.HEURISTIC,
        category="insights",
        importance=0.5,
        content_hash=hashlib.sha256(params.content.encode()).hexdigest()[:16],
    )

    with Storage(params.project_root) as storage:
        storage.upsert_node(node)
        edges_created = 0

        for rel_file in params.related_files[:5]:
            code_nodes = storage.get_nodes_by_file(rel_file)
            if not code_nodes:
                from .context_parser import _find_code_nodes_for_file
                code_nodes = _find_code_nodes_for_file(storage, rel_file)
            for cn in code_nodes[:2]:
                storage.upsert_edge(NeuralEdge(
                    source_id=node_id,
                    target_id=cn.id,
                    edge_type=EdgeType.RELATES_TO,
                    context=f"Insight: {topic_normalized}",
                    weight=0.6,
                ))
                edges_created += 1

    return (
        f"# Insight Saved\n\n"
        f"**ID**: `{node_id}`\n"
        f"**Topic**: {topic_normalized}\n"
        f"**Content**: {params.content[:120]}{'...' if len(params.content) > 120 else ''}\n"
        f"**Edges created**: {edges_created}\n\n"
        f"Use `neural_query` with `layer:insights` to browse, or `neural_inspect` with ID `{node_id}` for details."
    )


# ── Context Tool ──

async def neural_context(params: ContextInput) -> str:
    """Get a token-budgeted context snapshot for the current project.

    Returns a compact summary (~300–500 tokens) covering:
    - Index health and staleness status
    - Project overview (what this codebase does)
    - Active bugs (non-archived, open)
    - Active tasks (non-archived, pending/in_progress)
    - Nodes semantically relevant to your query (if query_hint provided)

    Designed for quick orientation at the start of a task or between steps.
    Use neural_query / neural_inspect for deeper exploration.
    """
    from .context import build_context
    return build_context(
        project_root=params.project_root,
        query_hint=params.query_hint,
        token_budget=params.token_budget,
    )


# ── Archive Tool ──

async def neural_archive(params: ArchiveInput) -> str:
    """Archive or unarchive a bug/task node.

    Archived nodes are excluded from neural_context and neural_query by default.
    Their importance score is decayed by 0.3× so they sink in semantic search.
    They remain fully accessible via neural_query with include_archived=true.

    Use this to:
    - Mark a bug as resolved: archive after setting bug_status='fixed'
    - Mark a task as done: archive after setting task_status='done'
    - Restore an archived item: unarchive to bring it back to active
    """
    from .storage import Storage

    with Storage(params.project_root) as storage:
        if params.action == "archive":
            success = storage.archive_node(params.node_id)
            if success:
                return f"Node `{params.node_id}` archived. It will no longer appear in active context fetches."
            return f"Node `{params.node_id}` not found."
        elif params.action == "unarchive":
            success = storage.unarchive_node(params.node_id)
            if success:
                return f"Node `{params.node_id}` unarchived. It will appear in active context fetches again."
            return f"Node `{params.node_id}` not found."
        else:
            return "Invalid action. Use: 'archive' or 'unarchive'"


# ── Task Management Tools ──

@mcp.tool(
    name="neural_list_tasks",
    annotations={
        "title": "Neural Memory — List Tasks",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def neural_list_tasks(params: ListTasksInput) -> str:
    """List tasks in the knowledge graph, grouped by status with code connections.

    Returns all non-archived tasks by default. Use status/priority filters to
    narrow results. Each task shows its linked code nodes (RELATES_TO edges).
    """
    from .storage import Storage
    from .models import EdgeType

    with Storage(params.project_root) as storage:
        tasks = storage.get_tasks(
            status=params.status,
            priority=params.priority,
            include_archived=params.include_archived,
        )

        if not tasks:
            filter_desc = []
            if params.status:
                filter_desc.append(f"status={params.status}")
            if params.priority:
                filter_desc.append(f"priority={params.priority}")
            filters = f" (filters: {', '.join(filter_desc)})" if filter_desc else ""
            return f"No tasks found{filters}. Use `neural_add_task` to create tasks."

        # Group by status
        by_status: dict[str, list] = {}
        for task in tasks:
            s = task.task_status or "pending"
            by_status.setdefault(s, []).append(task)

        STATUS_ORDER = ["in_progress", "testing", "pending", "done"]
        lines = [f"# Tasks ({len(tasks)} total)\n"]

        for status_key in STATUS_ORDER + [s for s in by_status if s not in STATUS_ORDER]:
            group = by_status.get(status_key, [])
            if not group:
                continue
            lines.append(f"## {status_key.replace('_', ' ').title()} ({len(group)})\n")
            for task in group:
                pri = f" ({task.priority})" if task.priority else ""
                lines.append(f"- **{task.name}**{pri} `id:{task.id}`")
                # Show code connections
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

    return "\n".join(lines)


@mcp.tool(
    name="neural_update_task",
    annotations={
        "title": "Neural Memory — Update Task",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def neural_update_task(params: UpdateTaskInput) -> str:
    """Update a task's status, priority, or add related file connections.

    Resolves the task by name substring or exact node ID. Updates are applied
    to the first matching task. Normalizes 'new' status alias to 'pending'.
    """
    from .storage import Storage
    from .models import TASK_STATUS_ALIASES, VALID_TASK_STATUSES, VALID_PRIORITIES, NeuralEdge, EdgeType
    from .context_parser import _find_code_nodes_for_file

    # Validate inputs
    if params.task_status is not None:
        normalized_status = TASK_STATUS_ALIASES.get(params.task_status, params.task_status)
        if normalized_status not in VALID_TASK_STATUSES:
            return f"Invalid task_status '{params.task_status}'. Valid: {sorted(VALID_TASK_STATUSES)} (or 'new' as alias for 'pending')"
    else:
        normalized_status = None

    if params.priority is not None and params.priority not in VALID_PRIORITIES:
        return f"Invalid priority '{params.priority}'. Valid: {sorted(VALID_PRIORITIES)}"

    with Storage(params.project_root) as storage:
        # Try exact ID match first
        task = storage.get_node(params.title_or_id)

        # Fall back to name substring search among task nodes
        if task is None or task.category != "tasks":
            all_tasks = storage.get_tasks(include_archived=True)
            search = params.title_or_id.lower()
            matches = [t for t in all_tasks if search in t.name.lower()]
            if not matches:
                return f"No task found matching '{params.title_or_id}'. Use `neural_list_tasks` to see all tasks."
            task = matches[0]

        changes: list[str] = []

        if normalized_status is not None:
            storage.update_node_field(task.id, "task_status", normalized_status)
            changes.append(f"status → {normalized_status}")

        if params.priority is not None:
            storage.update_node_field(task.id, "priority", params.priority)
            changes.append(f"priority → {params.priority}")

        # Add new file relations
        edges_added = 0
        if params.related_files:
            for rel_file in params.related_files[:5]:
                for cn in _find_code_nodes_for_file(storage, rel_file)[:2]:
                    storage.upsert_edge(NeuralEdge(
                        source_id=task.id,
                        target_id=cn.id,
                        edge_type=EdgeType.RELATES_TO,
                        context=f"Task: {task.name[:60]}",
                        weight=0.7,
                    ))
                    edges_added += 1
            if edges_added:
                changes.append(f"linked {edges_added} new code node(s)")

    if not changes:
        return f"No changes requested for task '{task.name}'."

    return (
        f"# Task Updated\n\n"
        f"**Task**: {task.name}\n"
        f"**ID**: `{task.id}`\n"
        f"**Changes**: {', '.join(changes)}\n\n"
        f"Use `neural_list_tasks` to see all tasks or `neural_inspect` with ID `{task.id}` for details."
    )


# Entry point
if __name__ == "__main__":
    try:
        mcp.run()
    except Exception as exc:
        print(f"neural-memory MCP server failed: {exc}", file=sys.stderr)
        sys.exit(1)
