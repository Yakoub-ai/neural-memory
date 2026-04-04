"""Multi-language parser dispatcher for neural memory.

Delegates file parsing to the tree-sitter engine (ts_parser.py) which
handles all supported languages via grammar packages and .scm query files.

Public interface (unchanged from the original Python-only version):
  parse_file(file_path, source=None) -> (nodes, edges)
  resolve_edges(all_nodes, edges) -> edges
"""

from __future__ import annotations

from typing import Optional

from .models import NeuralNode, NeuralEdge

_UNRESOLVED = "__unresolved__"

# Module-level singleton — one TreeSitterParser shared per process
_ts_parser_instance = None


def _get_ts_parser():
    global _ts_parser_instance
    if _ts_parser_instance is None:
        from .ts_parser import TreeSitterParser
        _ts_parser_instance = TreeSitterParser()
    return _ts_parser_instance


def parse_file(
    file_path: str,
    source: Optional[str] = None,
) -> tuple[list[NeuralNode], list[NeuralEdge]]:
    """Parse any supported source file into nodes and edges.

    Language is auto-detected from the file extension. Files with unsupported
    extensions are silently skipped (returns empty lists).

    Args:
        file_path: Relative path used as the node's file_path and for language detection.
        source: File source text. If None, read from disk using file_path as an absolute path.

    Returns:
        (nodes, edges) — edge targets may be unresolved; call resolve_edges() after
        all files are parsed.
    """
    return _get_ts_parser().parse_file(file_path, source)


def resolve_edges(all_nodes: dict[str, NeuralNode], edges: list[NeuralEdge]) -> list[NeuralEdge]:
    """Resolve unresolved edge targets to actual node IDs where possible.

    Prefers same-file matches to avoid cross-file name collisions (e.g. two
    files each defining a function called ``helper``). Works across all
    languages since matching is purely name-based.

    Args:
        all_nodes: All parsed nodes keyed by node ID.
        edges: All collected edges, possibly with ``__unresolved__<name>`` targets.

    Returns:
        Edges with resolved target IDs. Edges that cannot be resolved are dropped.
    """
    # Build name -> [(node_id, file_path)] for disambiguation
    name_to_candidates: dict[str, list[tuple[str, str]]] = {}
    for node in all_nodes.values():
        name_to_candidates.setdefault(node.name, []).append((node.id, node.file_path))
        # Also index short name (e.g., "MyClass.method" -> index "method" too)
        if "." in node.name:
            short = node.name.split(".")[-1]
            name_to_candidates.setdefault(short, []).append((node.id, node.file_path))

    resolved = []
    for edge in edges:
        if not edge.target_id.startswith(_UNRESOLVED):
            resolved.append(edge)
            continue
        target_name = edge.target_id[len(_UNRESOLVED):]
        candidates = name_to_candidates.get(target_name, [])
        if not candidates:
            continue
        # Prefer same-file match; fall back to first candidate
        source_node = all_nodes.get(edge.source_id)
        source_file = source_node.file_path if source_node else None
        same_file_match = next((cid for cid, fp in candidates if fp == source_file), None)
        edge.target_id = same_file_match if same_file_match else candidates[0][0]
        resolved.append(edge)

    return resolved
