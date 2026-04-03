"""Generic tree-sitter parser for any language with a LanguageConfig."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

from ..models import NeuralEdge, NeuralNode, NodeType, EdgeType
from .base import LanguageConfig


def _node_id(file_path: str, name: str, line: int) -> str:
    raw = f"{file_path}::{name}::{line}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _module_id(file_path: str) -> str:
    return hashlib.sha256(f"module::{file_path}".encode()).hexdigest()[:12]


def _get_source_segment(source_bytes: bytes, node) -> str:
    try:
        return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
    except Exception:
        return ""


def _get_name(ts_node, name_field: str = "name") -> Optional[str]:
    """Extract name from a tree-sitter node via a named field."""
    name_node = ts_node.child_by_field_name(name_field)
    if name_node and name_node.text:
        return name_node.text.decode("utf-8", errors="replace")
    return None


def _extract_calls(ts_node, call_types: frozenset[str]) -> list[str]:
    """Recursively collect call target names from a subtree."""
    calls: list[str] = []
    stack = list(ts_node.children)
    while stack:
        child = stack.pop()
        if child.type in call_types:
            fn_node = child.child_by_field_name("function")
            if fn_node:
                # Simple identifier or attribute access
                if fn_node.type in ("identifier", "property_identifier",
                                    "type_identifier", "field_identifier"):
                    if fn_node.text:
                        calls.append(fn_node.text.decode("utf-8", errors="replace"))
                elif fn_node.type in ("member_expression", "selector_expression",
                                      "field_expression", "qualified_identifier"):
                    attr = fn_node.child_by_field_name("field") or fn_node.child_by_field_name("alternative")
                    if attr and attr.text:
                        calls.append(attr.text.decode("utf-8", errors="replace"))
        stack.extend(child.children)
    return calls


def _line_at(ts_node) -> int:
    return ts_node.start_point[0] + 1


def _end_line(ts_node) -> int:
    return ts_node.end_point[0] + 1


_CALL_TYPES = frozenset([
    "call_expression",    # TS/JS/Rust
    "function_call_expression",  # Go (gopls sometimes uses this)
    "call",              # some grammars
])


class TreeSitterParser:
    """Parses a source file using tree-sitter, configured by a LanguageConfig."""

    def __init__(self, config: LanguageConfig) -> None:
        from tree_sitter import Language, Parser
        self._config = config
        self._ts_language = Language(config.language_fn())
        self._parser = Parser(self._ts_language)

    @property
    def language_id(self) -> str:
        return self._config.language_id

    @property
    def file_extensions(self) -> list[str]:
        return self._config.extensions

    def parse_file(
        self,
        file_path: str,
        source: Optional[str] = None,
    ) -> tuple[list[NeuralNode], list[NeuralEdge]]:
        if source is None:
            try:
                source = Path(file_path).read_text(encoding="utf-8", errors="replace")
            except OSError:
                return [], []

        source_bytes = source.encode("utf-8")
        tree = self._parser.parse(source_bytes)

        nodes: list[NeuralNode] = []
        edges: list[NeuralEdge] = []

        line_count = source.count("\n") + 1
        mod_id = _module_id(file_path)
        mod_node = NeuralNode(
            id=mod_id,
            name=Path(file_path).stem,
            node_type=NodeType.MODULE,
            file_path=file_path,
            line_start=1,
            line_end=line_count,
            language=self._config.language_id,
            raw_code=source[:500],  # first 500 chars for module summary
        )
        nodes.append(mod_node)

        cfg = self._config
        top_def_types = set(cfg.definition_types.keys())
        container_types = {c.ts_type: c for c in cfg.containers}
        import_types = set(cfg.import_node_types)

        for ts_node in tree.root_node.children:
            # Handle export wrapper (TypeScript: export_statement)
            effective_node = ts_node
            if cfg.wrapper_type and ts_node.type == cfg.wrapper_type:
                inner = ts_node.child_by_field_name(cfg.wrapper_field)
                if inner:
                    effective_node = inner
                else:
                    continue

            ntype = effective_node.type

            # Import — may produce multiple edges (e.g. Go grouped imports)
            if ntype in import_types:
                for import_name in self._collect_import_names(effective_node):
                    edges.append(NeuralEdge(
                        source_id=mod_id,
                        target_id=f"__unresolved__{import_name}",
                        edge_type=EdgeType.IMPORTS,
                    ))
                continue

            # Go-style type_declaration
            if cfg.type_spec and ntype == cfg.type_spec.outer_type:
                self._extract_type_specs(
                    effective_node, file_path, mod_id,
                    source_bytes, nodes, edges,
                )
                continue

            # Container (class, impl)
            if ntype in container_types:
                container_spec = container_types[ntype]
                self._extract_container(
                    effective_node, file_path, mod_id,
                    source_bytes, nodes, edges, container_spec,
                )
                continue

            # Plain definition (function, enum, typedef, etc.)
            if ntype in top_def_types:
                model_type = cfg.definition_types[ntype]
                name_field = cfg.definition_name_fields.get(ntype, "name")
                name = _get_name(effective_node, name_field)
                if not name:
                    continue
                nid = _node_id(file_path, name, _line_at(effective_node))
                raw = _get_source_segment(source_bytes, effective_node)
                node = NeuralNode(
                    id=nid,
                    name=name,
                    node_type=model_type,
                    file_path=file_path,
                    line_start=_line_at(effective_node),
                    line_end=_end_line(effective_node),
                    language=self._config.language_id,
                    raw_code=raw,
                    is_public=self._is_public(name, effective_node),
                )
                nodes.append(node)
                edges.append(NeuralEdge(
                    source_id=mod_id,
                    target_id=nid,
                    edge_type=EdgeType.CONTAINS,
                ))
                # Extract calls
                for call_name in _extract_calls(effective_node, _CALL_TYPES):
                    edges.append(NeuralEdge(
                        source_id=nid,
                        target_id=f"__unresolved__{call_name}",
                        edge_type=EdgeType.CALLS,
                    ))

        return nodes, edges

    def _extract_container(
        self,
        ts_node,
        file_path: str,
        mod_id: str,
        source_bytes: bytes,
        nodes: list[NeuralNode],
        edges: list[NeuralEdge],
        spec,
    ) -> None:
        name = _get_name(ts_node, spec.name_field)
        # Rust impl: name comes from "type" field
        if not name:
            name = _get_name(ts_node, "type")
        if not name:
            return

        cid = _node_id(file_path, name, _line_at(ts_node))
        raw = _get_source_segment(source_bytes, ts_node)
        class_node = NeuralNode(
            id=cid,
            name=name,
            node_type=spec.model_type,
            file_path=file_path,
            line_start=_line_at(ts_node),
            line_end=_end_line(ts_node),
            language=self._config.language_id,
            raw_code=raw[:1000],
            is_public=self._is_public(name, ts_node),
        )
        nodes.append(class_node)
        edges.append(NeuralEdge(
            source_id=mod_id,
            target_id=cid,
            edge_type=EdgeType.CONTAINS,
        ))

        # Extract methods from the body
        body = ts_node.child_by_field_name(spec.body_field)
        if not body:
            # Fallback: look for the first node of body_field type
            for child in ts_node.children:
                if child.type in ("class_body", "declaration_list",
                                  "block", "field_declaration_list"):
                    body = child
                    break
        if not body:
            return

        for child in body.children:
            if child.type not in spec.nested_type_map:
                continue
            method_model_type = spec.nested_type_map[child.type]
            mname = _get_name(child)
            if not mname:
                continue
            mid = _node_id(file_path, f"{name}.{mname}", _line_at(child))
            mraw = _get_source_segment(source_bytes, child)
            method_node = NeuralNode(
                id=mid,
                name=f"{name}.{mname}",
                node_type=method_model_type,
                file_path=file_path,
                line_start=_line_at(child),
                line_end=_end_line(child),
                language=self._config.language_id,
                raw_code=mraw,
                is_public=self._is_public(mname, child),
            )
            nodes.append(method_node)
            edges.append(NeuralEdge(
                source_id=cid,
                target_id=mid,
                edge_type=EdgeType.CONTAINS,
            ))
            for call_name in _extract_calls(child, _CALL_TYPES):
                edges.append(NeuralEdge(
                    source_id=mid,
                    target_id=f"__unresolved__{call_name}",
                    edge_type=EdgeType.CALLS,
                ))

    def _extract_type_specs(
        self,
        ts_node,
        file_path: str,
        mod_id: str,
        source_bytes: bytes,
        nodes: list[NeuralNode],
        edges: list[NeuralEdge],
    ) -> None:
        """Extract Go-style type declarations (struct, interface, type alias)."""
        spec_cfg = self._config.type_spec
        if not spec_cfg:
            return
        for child in ts_node.children:
            if child.type != spec_cfg.spec_type:
                continue
            name = _get_name(child, spec_cfg.name_field)
            if not name:
                continue
            value_node = child.child_by_field_name(spec_cfg.value_field)
            if not value_node:
                continue
            model_type = spec_cfg.value_type_map.get(value_node.type, NodeType.TYPE_DEF)
            nid = _node_id(file_path, name, _line_at(child))
            raw = _get_source_segment(source_bytes, ts_node)
            node = NeuralNode(
                id=nid,
                name=name,
                node_type=model_type,
                file_path=file_path,
                line_start=_line_at(ts_node),
                line_end=_end_line(ts_node),
                language=self._config.language_id,
                raw_code=raw,
                is_public=name[0].isupper() if name else False,
            )
            nodes.append(node)
            edges.append(NeuralEdge(
                source_id=mod_id,
                target_id=nid,
                edge_type=EdgeType.CONTAINS,
            ))

    def _collect_import_names(self, ts_node) -> list[str]:
        """Collect all imported module/package names from an import statement.

        Handles both single imports and grouped imports (Go-style import_spec_list).
        Returns a list because one import_declaration can import many packages.
        """
        names: list[str] = []
        self._collect_import_names_recursive(ts_node, names)
        return names

    def _collect_import_names_recursive(self, ts_node, names: list[str]) -> None:
        # Direct source/path field (TypeScript, JavaScript)
        for field_name in ("source", "path"):
            field_node = ts_node.child_by_field_name(field_name)
            if field_node and field_node.text:
                raw = field_node.text.decode("utf-8", errors="replace").strip("\"'`")
                name = raw.split("/")[-1].split(".")[-1]
                if name:
                    names.append(name)
                return

        # String literal directly in the node (Go import_spec)
        for child in ts_node.children:
            if child.type in ("interpreted_string_literal", "raw_string_literal",
                              "string", "string_literal", "string_content"):
                raw = child.text.decode("utf-8", errors="replace").strip("\"'`")
                name = raw.split("/")[-1].split(".")[-1]
                if name:
                    names.append(name)
                return

        # Rust: use_declaration -> scoped_identifier or identifier
        for child in ts_node.children:
            if child.type in ("scoped_identifier", "scoped_use_list", "use_list",
                              "use_as_clause"):
                # Get the rightmost name from scoped paths (e.g. std::fmt -> fmt)
                name_node = child.child_by_field_name("name")
                if name_node and name_node.text:
                    names.append(name_node.text.decode("utf-8", errors="replace"))
                    return
                # Fallback: use the whole text and take the last segment
                if child.text:
                    raw = child.text.decode("utf-8", errors="replace").replace("::", "/")
                    name = raw.split("/")[-1].strip("{ }").split(",")[0].strip()
                    if name:
                        names.append(name)
                    return
            elif child.type == "identifier" and child.text:
                names.append(child.text.decode("utf-8", errors="replace"))
                return

        # Recurse into child spec/list nodes (Go import_spec_list, import_spec)
        for child in ts_node.children:
            if child.type in ("import_spec_list", "import_spec",
                              "named_imports", "namespace_import"):
                self._collect_import_names_recursive(child, names)

    def _is_public(self, name: str, ts_node) -> bool:
        """Heuristic: is this definition publicly accessible?"""
        lang = self._config.language_id
        if lang in ("go", "rust"):
            return name[:1].isupper() if name else False
        if lang in ("typescript", "javascript"):
            # Public if not prefixed with private/protected/_ convention
            return not name.startswith("_")
        return True

    def resolve_edges(
        self,
        all_nodes: dict[str, NeuralNode],
        edges: list[NeuralEdge],
    ) -> list[NeuralEdge]:
        """Resolve __unresolved__ edge targets. Reuses the Python parser logic."""
        from ..parser import resolve_edges as _resolve_edges
        return _resolve_edges(all_nodes, edges)
