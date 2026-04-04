"""Graph operations for neural memory — traversal, scoring, querying."""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

from .models import NeuralNode, NeuralEdge, EdgeType, NodeType
from .storage import Storage


def compute_importance(
    storage: Storage,
    nodes: Optional[list[NeuralNode]] = None,
) -> None:
    """Compute importance scores for all nodes based on connectivity.

    Importance = normalized(in_degree + 0.5 * out_degree + type_weight)

    Uses bulk SQL queries to avoid N+1 round-trips to SQLite.

    Args:
        storage: open Storage instance.
        nodes: pre-fetched node list to avoid an extra get_all_nodes() call.
               If None, nodes are fetched from storage.
    """
    type_weights = {
        NodeType.MODULE: 0.3,
        NodeType.CLASS: 0.4,
        NodeType.FUNCTION: 0.2,
        NodeType.METHOD: 0.1,
        NodeType.CONFIG: 0.3,
        NodeType.EXPORT: 0.2,
        NodeType.TYPE_DEF: 0.15,
        NodeType.OTHER: 0.05,
    }

    all_nodes = nodes if nodes is not None else storage.get_all_nodes()
    if not all_nodes:
        return

    degree_counts = storage.get_all_degree_counts()
    scores: dict[str, float] = {}
    node_map: dict[str, NeuralNode] = {}

    for node in all_nodes:
        node_map[node.id] = node
        in_deg, out_deg = degree_counts.get(node.id, (0, 0))
        raw_score = (
            in_deg * 1.0 +
            out_deg * 0.5 +
            type_weights.get(node.node_type, 0.05)
        )
        if node.is_public:
            raw_score *= 1.2
        scores[node.id] = raw_score

    # Normalize to 0-1
    max_score = max(scores.values()) if scores else 1.0
    if max_score == 0:
        max_score = 1.0

    for nid, raw in scores.items():
        node_map[nid].importance = round(raw / max_score, 3)

    storage.batch_upsert_nodes(list(node_map.values()))


def get_neighborhood(
    storage: Storage,
    node_id: str,
    depth: int = 1,
    include_types: Optional[list[EdgeType]] = None
) -> dict:
    """Get a node and its neighborhood up to `depth` hops.

    Returns:
        {
            "center": NeuralNode,
            "callers": [NeuralNode, ...],      # who calls this
            "callees": [NeuralNode, ...],      # what this calls
            "siblings": [NeuralNode, ...],     # same parent
            "parent": NeuralNode | None,
            "children": [NeuralNode, ...],     # contained nodes
        }
    """
    center = storage.get_node(node_id)
    if not center:
        return {"error": f"Node {node_id} not found"}

    result = {
        "center": center,
        "callers": [],
        "callees": [],
        "siblings": [],
        "parent": None,
        "children": [],
    }

    # Incoming edges
    for edge in storage.get_edges_to(node_id):
        if include_types and edge.edge_type not in include_types:
            continue
        source = storage.get_node(edge.source_id)
        if not source:
            continue

        if edge.edge_type == EdgeType.CALLS:
            result["callers"].append(source)
        elif edge.edge_type == EdgeType.CONTAINS:
            result["parent"] = source

    # Outgoing edges
    for edge in storage.get_edges_from(node_id):
        if include_types and edge.edge_type not in include_types:
            continue
        target = storage.get_node(edge.target_id)
        if not target:
            continue

        if edge.edge_type == EdgeType.CALLS:
            result["callees"].append(target)
        elif edge.edge_type == EdgeType.CONTAINS:
            result["children"].append(target)

    # Siblings: nodes with the same parent
    if result["parent"]:
        for edge in storage.get_edges_from(result["parent"].id):
            if edge.edge_type == EdgeType.CONTAINS and edge.target_id != node_id:
                sibling = storage.get_node(edge.target_id)
                if sibling:
                    result["siblings"].append(sibling)

    return result


def trace_call_chain(
    storage: Storage,
    node_id: str,
    direction: str = "up",
    max_depth: int = 5
) -> list[list[NeuralNode]]:
    """Trace call chains up (who calls this?) or down (what does this call?).

    Returns list of chains, each chain is a list of nodes from start to end.
    """
    chains: list[list[NeuralNode]] = []
    visited: set[str] = set()

    def _trace(current_id: str, chain: list[NeuralNode], depth: int):
        if depth >= max_depth or current_id in visited:
            if chain:
                chains.append(list(chain))
            return

        visited.add(current_id)
        node = storage.get_node(current_id)
        if not node:
            return
        chain.append(node)

        if direction == "up":
            edges = storage.get_edges_to(current_id)
            next_edges = [e for e in edges if e.edge_type == EdgeType.CALLS]
        else:
            edges = storage.get_edges_from(current_id)
            next_edges = [e for e in edges if e.edge_type == EdgeType.CALLS]

        if not next_edges:
            chains.append(list(chain))
        else:
            for edge in next_edges:
                next_id = edge.source_id if direction == "up" else edge.target_id
                _trace(next_id, chain, depth + 1)

        chain.pop()
        visited.discard(current_id)

    _trace(node_id, [], 0)
    return chains


def format_node_summary(node: NeuralNode, level: str = "short") -> str:
    """Format a node for display."""
    icon = {
        NodeType.MODULE: "📦",
        NodeType.CLASS: "🏗️",
        NodeType.FUNCTION: "⚡",
        NodeType.METHOD: "🔧",
        NodeType.CONFIG: "⚙️",
        NodeType.EXPORT: "📤",
        NodeType.TYPE_DEF: "📝",
        NodeType.OTHER: "📎",
    }.get(node.node_type, "📎")

    if level == "short":
        return f"{icon} **{node.name}** ({node.node_type.value}) — {node.summary_short}"
    else:
        lines = [
            f"{icon} **{node.name}** ({node.node_type.value})",
            f"   File: {node.file_path}:{node.line_start}-{node.line_end}",
        ]
        if node.signature:
            lines.append(f"   Signature: `{node.signature}`")
        lines.append(f"   Importance: {node.importance:.2f} | Complexity: {node.complexity}")
        lines.append(f"   {node.summary_detailed}")
        if node.has_redacted_content:
            lines.append("   ⚠️ Contains redacted sensitive content")
        return "\n".join(lines)


def format_neighborhood(neighborhood: dict) -> str:
    """Format a neighborhood result for display."""
    if "error" in neighborhood:
        return neighborhood["error"]

    center = neighborhood["center"]
    lines = [
        "# Neural Node Inspection",
        "",
        format_node_summary(center, level="detailed"),
        "",
    ]

    if neighborhood["parent"]:
        lines.append(f"## Parent: {format_node_summary(neighborhood['parent'], 'short')}")
        lines.append("")

    if neighborhood["callers"]:
        lines.append(f"## Called by ({len(neighborhood['callers'])})")
        for n in neighborhood["callers"]:
            lines.append(f"  - {format_node_summary(n, 'short')}")
        lines.append("")

    if neighborhood["callees"]:
        lines.append(f"## Calls ({len(neighborhood['callees'])})")
        for n in neighborhood["callees"]:
            lines.append(f"  - {format_node_summary(n, 'short')}")
        lines.append("")

    if neighborhood["children"]:
        lines.append(f"## Contains ({len(neighborhood['children'])})")
        for n in neighborhood["children"]:
            lines.append(f"  - {format_node_summary(n, 'short')}")
        lines.append("")

    if neighborhood["siblings"]:
        lines.append(f"## Siblings ({len(neighborhood['siblings'])})")
        for n in sorted(neighborhood["siblings"], key=lambda x: -x.importance)[:10]:
            lines.append(f"  - {format_node_summary(n, 'short')}")

    return "\n".join(lines)
