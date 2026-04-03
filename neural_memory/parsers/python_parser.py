"""Python parser — wraps the existing AST-based neural_memory.parser module."""

from __future__ import annotations

from typing import Optional

from ..models import NeuralEdge, NeuralNode
from ..parser import parse_file as _parse_file
from ..parser import resolve_edges as _resolve_edges


class PythonParser:
    """Delegates to the battle-tested Python AST parser."""

    language_id = "python"
    file_extensions = [".py"]

    def parse_file(
        self,
        file_path: str,
        source: Optional[str] = None,
    ) -> tuple[list[NeuralNode], list[NeuralEdge]]:
        nodes, edges = _parse_file(file_path, source)
        for node in nodes:
            node.language = "python"
        return nodes, edges

    def resolve_edges(
        self,
        all_nodes: dict[str, NeuralNode],
        edges: list[NeuralEdge],
    ) -> list[NeuralEdge]:
        return _resolve_edges(all_nodes, edges)
