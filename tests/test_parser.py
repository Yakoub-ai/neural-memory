"""Tests for neural_memory.parser — parse_file and resolve_edges."""

import pytest

from neural_memory.models import NodeType, EdgeType, NeuralNode
from neural_memory.parser import (
    parse_file,
    resolve_edges,
)


# ── parse_file: basic cases ────────────────────────────────────────────────────

def test_empty_file_produces_one_module_node(tmp_path):
    f = tmp_path / "empty.py"
    f.write_text("", encoding="utf-8")
    nodes, edges = parse_file(str(f))
    assert len(nodes) == 1
    assert nodes[0].node_type == NodeType.MODULE
    assert len(edges) == 0


def test_syntax_error_still_returns_module(tmp_path):
    # tree-sitter is fault-tolerant: always returns at least a module node
    f = tmp_path / "bad.py"
    f.write_text("def (broken:", encoding="utf-8")
    nodes, edges = parse_file(str(f))
    assert any(n.node_type == NodeType.MODULE for n in nodes)


def test_module_docstring_captured(tmp_path):
    f = tmp_path / "mod.py"
    f.write_text('"""Module docstring."""\n\nx = 1\n', encoding="utf-8")
    nodes, edges = parse_file(str(f))
    mod = next(n for n in nodes if n.node_type == NodeType.MODULE)
    assert "Module docstring" in mod.docstring


def test_single_function_creates_correct_nodes(tmp_path):
    src = "def foo(x: int) -> str:\n    return str(x)\n"
    f = tmp_path / "funcs.py"
    f.write_text(src, encoding="utf-8")
    nodes, edges = parse_file(str(f))

    assert len(nodes) == 2  # MODULE + FUNCTION
    func = next(n for n in nodes if n.node_type == NodeType.FUNCTION)
    assert func.name == "foo"
    assert func.line_start == 1
    assert "foo" in func.signature
    assert func.is_public is True
    assert func.content_hash != ""
    assert len(func.content_hash) == 16


def test_private_function_is_not_public(tmp_path):
    src = "def _helper():\n    pass\n"
    f = tmp_path / "priv.py"
    f.write_text(src, encoding="utf-8")
    nodes, _ = parse_file(str(f))
    func = next(n for n in nodes if n.node_type == NodeType.FUNCTION)
    assert func.is_public is False


def test_function_contains_edge(tmp_path):
    src = "def foo():\n    pass\n"
    f = tmp_path / "funcs.py"
    f.write_text(src, encoding="utf-8")
    nodes, edges = parse_file(str(f))
    mod = next(n for n in nodes if n.node_type == NodeType.MODULE)
    func = next(n for n in nodes if n.node_type == NodeType.FUNCTION)
    contains = [e for e in edges if e.edge_type == EdgeType.CONTAINS]
    assert any(e.source_id == mod.id and e.target_id == func.id for e in contains)


def test_class_creates_correct_nodes(tmp_path):
    src = "class MyClass:\n    def method(self):\n        pass\n"
    f = tmp_path / "cls.py"
    f.write_text(src, encoding="utf-8")
    nodes, edges = parse_file(str(f))

    types = {n.node_type for n in nodes}
    assert NodeType.MODULE in types
    assert NodeType.CLASS in types
    assert NodeType.METHOD in types

    cls = next(n for n in nodes if n.node_type == NodeType.CLASS)
    assert cls.name == "MyClass"
    assert cls.is_public is True


def test_method_name_uses_class_prefix(tmp_path):
    src = "class Foo:\n    def bar(self):\n        pass\n"
    f = tmp_path / "cls.py"
    f.write_text(src, encoding="utf-8")
    nodes, _ = parse_file(str(f))
    method = next(n for n in nodes if n.node_type == NodeType.METHOD)
    assert method.name == "Foo.bar"


def test_class_method_contains_edge(tmp_path):
    src = "class Foo:\n    def bar(self):\n        pass\n"
    f = tmp_path / "cls.py"
    f.write_text(src, encoding="utf-8")
    nodes, edges = parse_file(str(f))
    cls = next(n for n in nodes if n.node_type == NodeType.CLASS)
    method = next(n for n in nodes if n.node_type == NodeType.METHOD)
    contains = [e for e in edges if e.edge_type == EdgeType.CONTAINS]
    assert any(e.source_id == cls.id and e.target_id == method.id for e in contains)


def test_inheritance_creates_inherits_edge(tmp_path):
    src = "class Base:\n    pass\n\nclass Child(Base):\n    pass\n"
    f = tmp_path / "inh.py"
    f.write_text(src, encoding="utf-8")
    nodes, edges = parse_file(str(f))
    inherits = [e for e in edges if e.edge_type == EdgeType.INHERITS]
    assert len(inherits) >= 1


def test_async_function_signature_prefix(tmp_path):
    src = "async def fetch():\n    pass\n"
    f = tmp_path / "async_.py"
    f.write_text(src, encoding="utf-8")
    nodes, _ = parse_file(str(f))
    func = next(n for n in nodes if n.node_type == NodeType.FUNCTION)
    assert func.signature.startswith("async def")


def test_function_with_docstring(tmp_path):
    src = 'def foo():\n    """Does something."""\n    pass\n'
    f = tmp_path / "doc.py"
    f.write_text(src, encoding="utf-8")
    nodes, _ = parse_file(str(f))
    func = next(n for n in nodes if n.node_type == NodeType.FUNCTION)
    assert "Does something" in func.docstring
    assert "Does something" in func.summary_short


def test_function_calls_create_unresolved_edges(tmp_path):
    src = "def foo():\n    bar()\n    baz()\n\ndef bar():\n    pass\n\ndef baz():\n    pass\n"
    f = tmp_path / "calls.py"
    f.write_text(src, encoding="utf-8")
    _, edges = parse_file(str(f))
    call_edges = [e for e in edges if e.edge_type == EdgeType.CALLS]
    targets = [e.target_id for e in call_edges]
    assert any("__unresolved__" in t for t in targets)


def test_import_creates_import_edges(tmp_path):
    src = "import os\nfrom pathlib import Path\n"
    f = tmp_path / "imp.py"
    f.write_text(src, encoding="utf-8")
    _, edges = parse_file(str(f))
    import_edges = [e for e in edges if e.edge_type == EdgeType.IMPORTS]
    assert len(import_edges) >= 2


def test_raw_code_is_stored(tmp_path):
    src = "def foo():\n    return 42\n"
    f = tmp_path / "raw.py"
    f.write_text(src, encoding="utf-8")
    nodes, _ = parse_file(str(f))
    func = next(n for n in nodes if n.node_type == NodeType.FUNCTION)
    assert "return 42" in func.raw_code


def test_parse_from_source_string(tmp_path):
    """parse_file can accept source directly without reading from disk."""
    src = "def greet(name: str) -> str:\n    return f'hi {name}'\n"
    f = tmp_path / "greet.py"
    # Write a different file — we'll pass source directly
    f.write_text("# placeholder", encoding="utf-8")
    nodes, _ = parse_file(str(f), source=src)
    func = next(n for n in nodes if n.node_type == NodeType.FUNCTION)
    assert func.name == "greet"


# ── resolve_edges ────────────────────────────────────────────────────────────────

def test_resolve_edges_links_known_nodes(tmp_path):
    src = "def foo():\n    bar()\n\ndef bar():\n    pass\n"
    f = tmp_path / "resolve.py"
    f.write_text(src, encoding="utf-8")
    nodes_list, edges = parse_file(str(f))
    all_nodes = {n.id: n for n in nodes_list}
    resolved = resolve_edges(all_nodes, edges)

    call_edges = [e for e in resolved if e.edge_type == EdgeType.CALLS]
    # After resolution, no target should be unresolved
    for e in call_edges:
        assert not e.target_id.startswith("__unresolved__")


def test_resolve_edges_drops_external_calls(tmp_path):
    src = "import requests\ndef foo():\n    requests.get('http://example.com')\n"
    f = tmp_path / "ext.py"
    f.write_text(src, encoding="utf-8")
    nodes_list, edges = parse_file(str(f))
    all_nodes = {n.id: n for n in nodes_list}
    resolved = resolve_edges(all_nodes, edges)

    # External 'get' call should be dropped (no matching node)
    call_edges = [e for e in resolved if e.edge_type == EdgeType.CALLS]
    for e in call_edges:
        assert not e.target_id.startswith("__unresolved__")


def test_resolve_edges_prefers_same_file_match(tmp_path):
    src_a = "def helper(): pass\ndef caller():\n    helper()\n"
    src_b = "def helper(): pass\n"
    f_a = tmp_path / "a.py"
    f_b = tmp_path / "b.py"
    f_a.write_text(src_a, encoding="utf-8")
    f_b.write_text(src_b, encoding="utf-8")
    nodes_a, edges_a = parse_file(str(f_a))
    nodes_b, edges_b = parse_file(str(f_b))
    all_nodes = {n.id: n for n in nodes_a + nodes_b}
    resolved = resolve_edges(all_nodes, edges_a + edges_b)

    helper_a = next(n for n in nodes_a if n.name == "helper")
    calls_edges = [e for e in resolved if e.edge_type == EdgeType.CALLS]
    # caller in a.py should resolve to helper in a.py, not b.py
    assert any(e.target_id == helper_a.id for e in calls_edges)
