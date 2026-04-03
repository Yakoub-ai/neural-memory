"""Python AST parser for extracting neural memory nodes and edges."""

from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
from typing import Optional

from .models import NeuralNode, NeuralEdge, NodeType, EdgeType, SummaryMode


def _node_id(file_path: str, name: str, node_type: NodeType) -> str:
    """Generate a deterministic node ID."""
    raw = f"{file_path}::{node_type.value}::{name}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _module_id(file_path: str) -> str:
    return _node_id(file_path, "__module__", NodeType.MODULE)


def _get_source_segment(source_lines: list[str], node: ast.AST) -> str:
    """Extract source code for an AST node."""
    start = node.lineno - 1
    end = getattr(node, "end_lineno", node.lineno)
    return "\n".join(source_lines[start:end])


def _estimate_complexity(node: ast.AST) -> int:
    """Estimate cyclomatic complexity of a function/method."""
    complexity = 1
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += len(child.values) - 1
        elif isinstance(child, (ast.Assert, ast.Raise)):
            complexity += 1
    return complexity


def _get_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Extract function signature as string."""
    args = node.args
    parts = []

    # Regular args
    defaults_offset = len(args.args) - len(args.defaults)
    for i, arg in enumerate(args.args):
        s = arg.arg
        if arg.annotation:
            s += f": {ast.dump(arg.annotation)}" if not hasattr(ast, 'unparse') else f": {ast.unparse(arg.annotation)}"
        default_idx = i - defaults_offset
        if default_idx >= 0 and default_idx < len(args.defaults):
            default = args.defaults[default_idx]
            s += f" = {ast.unparse(default)}" if hasattr(ast, 'unparse') else " = ..."
        parts.append(s)

    # *args
    if args.vararg:
        s = f"*{args.vararg.arg}"
        if args.vararg.annotation and hasattr(ast, 'unparse'):
            s += f": {ast.unparse(args.vararg.annotation)}"
        parts.append(s)

    # **kwargs
    if args.kwarg:
        s = f"**{args.kwarg.arg}"
        if args.kwarg.annotation and hasattr(ast, 'unparse'):
            s += f": {ast.unparse(args.kwarg.annotation)}"
        parts.append(s)

    sig = f"({', '.join(parts)})"

    # Return annotation
    if node.returns and hasattr(ast, 'unparse'):
        sig += f" -> {ast.unparse(node.returns)}"

    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {node.name}{sig}"


def _get_decorators(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> list[str]:
    """Extract decorator names."""
    decorators = []
    for dec in node.decorator_list:
        if hasattr(ast, 'unparse'):
            decorators.append(ast.unparse(dec))
        elif isinstance(dec, ast.Name):
            decorators.append(dec.id)
        elif isinstance(dec, ast.Attribute):
            decorators.append(f"{ast.dump(dec)}")
    return decorators


def _extract_calls(node: ast.AST) -> list[str]:
    """Extract all function/method call names from a node."""
    calls = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name):
                calls.append(child.func.id)
            elif isinstance(child.func, ast.Attribute):
                calls.append(child.func.attr)
    return calls


def _extract_imports(tree: ast.Module) -> list[tuple[str, Optional[str]]]:
    """Extract all imports as (module_or_name, alias) tuples."""
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name, alias.asname))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append((f"{module}.{alias.name}", alias.asname))
    return imports


def _heuristic_summary(node: ast.AST, source_lines: list[str]) -> str:
    """Generate a basic heuristic summary from AST info."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        docstring = ast.get_docstring(node) or ""
        if docstring:
            # Use first sentence of docstring
            first_line = docstring.strip().split('\n')[0].strip()
            return first_line[:200]
        # Fallback: describe from signature
        sig = _get_signature(node)
        calls = _extract_calls(node)
        body_len = (getattr(node, 'end_lineno', node.lineno) - node.lineno) + 1
        desc = f"{sig} — {body_len} lines"
        if calls:
            desc += f", calls: {', '.join(set(calls[:5]))}"
        return desc[:200]

    elif isinstance(node, ast.ClassDef):
        docstring = ast.get_docstring(node) or ""
        if docstring:
            return docstring.strip().split('\n')[0].strip()[:200]
        methods = [n.name for n in node.body
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        bases = [ast.unparse(b) if hasattr(ast, 'unparse') else "..." for b in node.bases]
        desc = f"class {node.name}"
        if bases:
            desc += f"({', '.join(bases)})"
        desc += f" — {len(methods)} methods"
        return desc[:200]

    return ""


def parse_file(file_path: str, source: Optional[str] = None) -> tuple[list[NeuralNode], list[NeuralEdge]]:
    """Parse a Python file and extract nodes + edges.

    Returns (nodes, edges) for the file.
    """
    if source is None:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()

    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError:
        return [], []

    source_lines = source.split("\n")
    nodes: list[NeuralNode] = []
    edges: list[NeuralEdge] = []

    # File content hash
    file_hash = hashlib.sha256(source.encode()).hexdigest()[:16]

    # Module node
    module_docstring = ast.get_docstring(tree) or ""
    mod_node = NeuralNode(
        id=_module_id(file_path),
        name=Path(file_path).stem,
        node_type=NodeType.MODULE,
        file_path=file_path,
        line_start=1,
        line_end=len(source_lines),
        summary_short=module_docstring.split('\n')[0][:200] if module_docstring else f"Module: {Path(file_path).name}",
        docstring=module_docstring,
        content_hash=file_hash,
        is_public=True,
    )
    nodes.append(mod_node)

    # Track names defined at module level for edge resolution
    defined_names: dict[str, str] = {}  # name -> node_id

    # Extract top-level definitions
    for item in ast.iter_child_nodes(tree):
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_id = _node_id(file_path, item.name, NodeType.FUNCTION)
            code = _get_source_segment(source_lines, item)
            n = NeuralNode(
                id=func_id,
                name=item.name,
                node_type=NodeType.FUNCTION,
                file_path=file_path,
                line_start=item.lineno,
                line_end=getattr(item, 'end_lineno', item.lineno),
                signature=_get_signature(item),
                docstring=ast.get_docstring(item) or "",
                summary_short=_heuristic_summary(item, source_lines),
                complexity=_estimate_complexity(item),
                is_public=not item.name.startswith("_"),
                decorators=_get_decorators(item),
                raw_code=code,
            )
            n.compute_hash(code)
            nodes.append(n)
            defined_names[item.name] = func_id

            # CONTAINS edge: module -> function
            edges.append(NeuralEdge(
                source_id=mod_node.id, target_id=func_id,
                edge_type=EdgeType.CONTAINS
            ))

            # CALLS edges
            for call_name in set(_extract_calls(item)):
                if call_name != item.name:  # Skip recursion for now
                    edges.append(NeuralEdge(
                        source_id=func_id,
                        target_id=f"__unresolved__{call_name}",
                        edge_type=EdgeType.CALLS,
                        context=f"called from {item.name}"
                    ))

        elif isinstance(item, ast.ClassDef):
            class_id = _node_id(file_path, item.name, NodeType.CLASS)
            code = _get_source_segment(source_lines, item)
            n = NeuralNode(
                id=class_id,
                name=item.name,
                node_type=NodeType.CLASS,
                file_path=file_path,
                line_start=item.lineno,
                line_end=getattr(item, 'end_lineno', item.lineno),
                docstring=ast.get_docstring(item) or "",
                summary_short=_heuristic_summary(item, source_lines),
                is_public=not item.name.startswith("_"),
                decorators=_get_decorators(item),
                raw_code=code,
            )
            n.compute_hash(code)
            nodes.append(n)
            defined_names[item.name] = class_id

            # CONTAINS edge: module -> class
            edges.append(NeuralEdge(
                source_id=mod_node.id, target_id=class_id,
                edge_type=EdgeType.CONTAINS
            ))

            # INHERITS edges
            for base in item.bases:
                base_name = ast.unparse(base) if hasattr(ast, 'unparse') else "..."
                edges.append(NeuralEdge(
                    source_id=class_id,
                    target_id=f"__unresolved__{base_name}",
                    edge_type=EdgeType.INHERITS,
                    context=f"{item.name} inherits from {base_name}"
                ))

            # Methods
            for method in item.body:
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_id = _node_id(file_path, f"{item.name}.{method.name}", NodeType.METHOD)
                    method_code = _get_source_segment(source_lines, method)
                    m = NeuralNode(
                        id=method_id,
                        name=f"{item.name}.{method.name}",
                        node_type=NodeType.METHOD,
                        file_path=file_path,
                        line_start=method.lineno,
                        line_end=getattr(method, 'end_lineno', method.lineno),
                        signature=_get_signature(method),
                        docstring=ast.get_docstring(method) or "",
                        summary_short=_heuristic_summary(method, source_lines),
                        complexity=_estimate_complexity(method),
                        is_public=not method.name.startswith("_"),
                        decorators=_get_decorators(method),
                        raw_code=method_code,
                    )
                    m.compute_hash(method_code)
                    nodes.append(m)
                    defined_names[f"{item.name}.{method.name}"] = method_id

                    # CONTAINS edge: class -> method
                    edges.append(NeuralEdge(
                        source_id=class_id, target_id=method_id,
                        edge_type=EdgeType.CONTAINS
                    ))

                    # CALLS edges from methods
                    for call_name in set(_extract_calls(method)):
                        edges.append(NeuralEdge(
                            source_id=method_id,
                            target_id=f"__unresolved__{call_name}",
                            edge_type=EdgeType.CALLS,
                            context=f"called from {item.name}.{method.name}"
                        ))

    # Import edges
    for module_name, alias in _extract_imports(tree):
        edges.append(NeuralEdge(
            source_id=mod_node.id,
            target_id=f"__unresolved__{module_name}",
            edge_type=EdgeType.IMPORTS,
            context=f"import {module_name}" + (f" as {alias}" if alias else "")
        ))

    return nodes, edges


def resolve_edges(all_nodes: dict[str, NeuralNode], edges: list[NeuralEdge]) -> list[NeuralEdge]:
    """Resolve __unresolved__ edge targets to actual node IDs where possible."""
    # Build name -> node_id lookup
    name_to_id: dict[str, str] = {}
    for node in all_nodes.values():
        name_to_id[node.name] = node.id
        # Also index short name (e.g., "MyClass.method" -> index "method" too)
        if "." in node.name:
            short = node.name.split(".")[-1]
            if short not in name_to_id:
                name_to_id[short] = node.id

    resolved = []
    for edge in edges:
        if edge.target_id.startswith("__unresolved__"):
            target_name = edge.target_id.replace("__unresolved__", "")
            if target_name in name_to_id:
                edge.target_id = name_to_id[target_name]
                resolved.append(edge)
            # Else: external call, drop the edge (or keep as unresolved)
        else:
            resolved.append(edge)

    return resolved
