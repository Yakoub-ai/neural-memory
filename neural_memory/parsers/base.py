"""Base types for the multi-language parser system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

from ..models import NeuralEdge, NeuralNode, NodeType


class LanguageParser(Protocol):
    """Interface all language parsers must satisfy."""

    @property
    def language_id(self) -> str:
        """Canonical language identifier, e.g. 'python', 'typescript'."""
        ...

    @property
    def file_extensions(self) -> list[str]:
        """File extensions handled by this parser, e.g. ['.ts', '.tsx']."""
        ...

    def parse_file(
        self,
        file_path: str,
        source: Optional[str] = None,
    ) -> tuple[list[NeuralNode], list[NeuralEdge]]:
        """Parse a source file and return nodes + unresolved edges."""
        ...

    def resolve_edges(
        self,
        all_nodes: dict[str, NeuralNode],
        edges: list[NeuralEdge],
    ) -> list[NeuralEdge]:
        """Resolve unresolved edge targets against the known node set."""
        ...


@dataclass
class ContainerSpec:
    """A node type that contains nested definitions (class body, impl block)."""
    # tree-sitter type of the container node (e.g. "class_declaration")
    ts_type: str
    # The NeuralNode type this container becomes (usually CLASS)
    model_type: NodeType
    # Field name on the container node that yields the body node
    body_field: str
    # tree-sitter types of nested items -> their NeuralNode types
    nested_type_map: dict[str, NodeType]
    # Optional: how to get name — defaults to child_by_field_name("name")
    name_field: str = "name"


@dataclass
class TypeSpecConfig:
    """Config for languages that nest types inside a wrapper declaration.

    Example (Go):
        type_declaration         <- outer_type
          type_spec              <- spec_type
            type_identifier      <- name (via name_field)
            struct_type          <- value_field, mapped to NodeType.CLASS
    """
    outer_type: str          # e.g. "type_declaration"
    spec_type: str           # e.g. "type_spec"
    name_field: str = "name"
    value_field: str = "type"
    # Maps the value node type -> NeuralNode type
    value_type_map: dict[str, NodeType] = field(default_factory=dict)


@dataclass
class LanguageConfig:
    """All configuration needed for a tree-sitter-based parser."""
    language_id: str
    extensions: list[str]
    # Callable that returns the tree-sitter Language object's capsule
    language_fn: Callable

    # Top-level definition nodes: tree-sitter type -> NeuralNode type
    # Name is extracted via child_by_field_name(name_field)
    definition_types: dict[str, NodeType]
    # Nodes whose "name" is accessed via a different field
    definition_name_fields: dict[str, str] = field(default_factory=dict)

    # Container nodes (class, impl block) with nested methods
    containers: list[ContainerSpec] = field(default_factory=list)

    # Import statement node types
    import_node_types: list[str] = field(default_factory=list)

    # Wrapper node that holds the real declaration (TypeScript export_statement)
    wrapper_type: str = ""
    wrapper_field: str = "declaration"

    # Go-style nested type declarations
    type_spec: Optional[TypeSpecConfig] = None
