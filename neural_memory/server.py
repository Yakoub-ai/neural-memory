"""Neural Memory MCP Server — exposes tools to Claude Code."""

from __future__ import annotations

import json
import os

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

from .agent import check_staleness, format_agent_report
from .config import load_config, save_config, NeuralConfig, RedactionConfig
from .graph import (
    get_neighborhood, format_neighborhood, format_node_summary,
    trace_call_chain, compute_importance
)
from .indexer import full_index, incremental_update
from .models import IndexMode, NodeType
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

    Returns layered summaries — short overview first, with the ability to
    dig deeper using neural_inspect on any result.
    """
    with Storage(params.project_root) as storage:
        results = storage.search_nodes(params.query, limit=params.limit)

    if not results:
        return f"No results found for '{params.query}'. Try a different search term or run `/neural-index` if the codebase hasn't been indexed."

    lines = [f"# Neural Query: '{params.query}'", f"Found {len(results)} result(s):", ""]

    for i, node in enumerate(results, 1):
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


# Entry point
if __name__ == "__main__":
    mcp.run()
