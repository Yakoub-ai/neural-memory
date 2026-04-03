"""Generate tiered overview nodes for the neural memory graph.

Two levels:
  project_overview  — one per project, summarizes the entire codebase
  directory_overview — one per directory with Python files

These sit at the top of the CONTAINS hierarchy and give agents an immediate
structural understanding without reading any source code.
"""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from .models import NeuralNode, NeuralEdge, NodeType, EdgeType, SummaryMode

if TYPE_CHECKING:
    from .storage import Storage


def _make_id(parts: str) -> str:
    return hashlib.sha256(parts.encode()).hexdigest()[:12]


# ── Project overview ───────────────────────────────────────────────────────────

def generate_project_overview(storage: "Storage") -> NeuralNode:
    """Create a single project-level overview node.

    Summarises total counts, top modules, and key entities by importance.
    """
    all_nodes = [n for n in storage.get_all_nodes() if n.category == "codebase"
                 and n.node_type not in (NodeType.PROJECT_OVERVIEW, NodeType.DIRECTORY_OVERVIEW)]

    stats = storage.get_stats()
    total_files = stats.get("total_files", 0)

    # Count by type
    by_type: dict[str, int] = defaultdict(int)
    for n in all_nodes:
        by_type[n.node_type.value] += 1

    # Top modules by importance
    modules = sorted(
        [n for n in all_nodes if n.node_type == NodeType.MODULE],
        key=lambda n: n.importance, reverse=True
    )
    top_module_names = [m.name.rsplit(".", 1)[0] if "." in m.name else m.name for m in modules[:6]]

    # Top classes / functions by importance
    top_entities = sorted(
        [n for n in all_nodes if n.node_type in (NodeType.CLASS, NodeType.FUNCTION)],
        key=lambda n: n.importance, reverse=True
    )[:5]
    top_entity_names = [e.name for e in top_entities]

    summary_parts = [
        f"Project with {total_files} files, "
        f"{by_type.get('class', 0)} classes, "
        f"{by_type.get('function', 0)} functions, "
        f"{by_type.get('method', 0)} methods.",
    ]
    if top_module_names:
        summary_parts.append(f"Key modules: {', '.join(top_module_names)}.")
    if top_entity_names:
        summary_parts.append(f"Key entities: {', '.join(top_entity_names)}.")

    summary_short = " ".join(summary_parts)[:200]
    summary_detailed = (
        f"Node counts: {dict(by_type)}\n"
        f"Top modules by importance: {top_module_names}\n"
        f"Top entities by importance: {top_entity_names}"
    )[:1000]

    node_id = _make_id("__project__::project_overview::project")

    return NeuralNode(
        id=node_id,
        name="project_overview",
        node_type=NodeType.PROJECT_OVERVIEW,
        file_path="__project__",
        line_start=0,
        line_end=0,
        summary_short=summary_short,
        summary_detailed=summary_detailed,
        summary_mode=SummaryMode.HEURISTIC,
        category="codebase",
        importance=1.0,   # Highest importance — it's the entry point
        content_hash=_make_id(summary_short),
    )


# ── Directory overviews ────────────────────────────────────────────────────────

def generate_directory_overviews(storage: "Storage") -> list[NeuralNode]:
    """Create one directory_overview node per directory that contains code nodes."""
    all_nodes = [n for n in storage.get_all_nodes() if n.category == "codebase"
                 and n.node_type not in (NodeType.PROJECT_OVERVIEW, NodeType.DIRECTORY_OVERVIEW)]

    # Group by directory
    by_dir: dict[str, list[NeuralNode]] = defaultdict(list)
    for n in all_nodes:
        d = str(Path(n.file_path).parent)
        by_dir[d].append(n)

    results: list[NeuralNode] = []
    for dir_path, nodes in by_dir.items():
        files = {n.file_path for n in nodes}
        classes = [n for n in nodes if n.node_type == NodeType.CLASS]
        functions = [n for n in nodes if n.node_type == NodeType.FUNCTION]

        top_entities = sorted(
            classes + functions, key=lambda n: n.importance, reverse=True
        )[:4]
        top_names = [e.name for e in top_entities]

        display_dir = dir_path if dir_path != "." else "(root)"
        summary_short = (
            f"{display_dir}: {len(files)} file(s), "
            f"{len(classes)} class(es), {len(functions)} function(s)."
            + (f" Key: {', '.join(top_names)}." if top_names else "")
        )[:200]

        node_id = _make_id(f"__dir__::directory_overview::{dir_path}")

        results.append(NeuralNode(
            id=node_id,
            name=f"dir: {display_dir}",
            node_type=NodeType.DIRECTORY_OVERVIEW,
            file_path=dir_path,
            line_start=0,
            line_end=0,
            summary_short=summary_short,
            summary_mode=SummaryMode.HEURISTIC,
            category="codebase",
            importance=0.9,
            content_hash=_make_id(summary_short),
        ))

    return results


# ── Edges connecting overview hierarchy ────────────────────────────────────────

def generate_overview_edges(
    storage: "Storage",
    project_node: NeuralNode,
    dir_nodes: list[NeuralNode],
) -> list[NeuralEdge]:
    """Create CONTAINS edges connecting the overview hierarchy.

    project_overview → directory_overview (one per dir)
    directory_overview → module nodes (in that directory)
    """
    edges: list[NeuralEdge] = []
    dir_by_path = {n.file_path: n for n in dir_nodes}

    # project → directories
    for dir_node in dir_nodes:
        edges.append(NeuralEdge(
            source_id=project_node.id,
            target_id=dir_node.id,
            edge_type=EdgeType.CONTAINS,
            context="project overview",
            weight=1.0,
        ))

    # directories → modules
    code_nodes = [n for n in storage.get_all_nodes()
                  if n.category == "codebase"
                  and n.node_type == NodeType.MODULE]

    for module_node in code_nodes:
        dir_path = str(Path(module_node.file_path).parent)
        dir_node = dir_by_path.get(dir_path)
        if dir_node:
            edges.append(NeuralEdge(
                source_id=dir_node.id,
                target_id=module_node.id,
                edge_type=EdgeType.CONTAINS,
                context="directory overview",
                weight=0.9,
            ))

    return edges


# ── Public entry point ─────────────────────────────────────────────────────────

def generate_and_store_overviews(storage: "Storage") -> dict:
    """Generate all overview nodes and store them. Returns stats.

    Deletes stale overview nodes first so removed files don't linger.
    """
    # Remove all existing overview nodes before regenerating
    for node_type in (NodeType.PROJECT_OVERVIEW, NodeType.DIRECTORY_OVERVIEW):
        for old_node in storage.get_nodes_by_type(node_type):
            storage.delete_nodes_by_file(old_node.file_path)

    project_node = generate_project_overview(storage)
    dir_nodes = generate_directory_overviews(storage)
    edges = generate_overview_edges(storage, project_node, dir_nodes)

    storage.upsert_node(project_node)
    for dn in dir_nodes:
        storage.upsert_node(dn)
    for e in edges:
        storage.upsert_edge(e)

    return {
        "project_overviews": 1,
        "directory_overviews": len(dir_nodes),
        "overview_edges": len(edges),
    }
