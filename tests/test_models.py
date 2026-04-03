"""Tests for neural_memory.models — NeuralNode, NeuralEdge, IndexState."""

import pytest

from neural_memory.models import (
    NeuralNode,
    NeuralEdge,
    IndexState,
    NodeType,
    EdgeType,
    SummaryMode,
    IndexMode,
)


def _make_node(**kwargs) -> NeuralNode:
    defaults = dict(
        id="abc123def456",
        name="my_func",
        node_type=NodeType.FUNCTION,
        file_path="mymodule.py",
        line_start=10,
        line_end=20,
    )
    defaults.update(kwargs)
    return NeuralNode(**defaults)


def _make_edge(**kwargs) -> NeuralEdge:
    defaults = dict(
        source_id="src000001",
        target_id="tgt000002",
        edge_type=EdgeType.CALLS,
    )
    defaults.update(kwargs)
    return NeuralEdge(**defaults)


# ── NodeType enum ──────────────────────────────────────────────────────────────

def test_node_type_values():
    assert NodeType.MODULE.value == "module"
    assert NodeType.CLASS.value == "class"
    assert NodeType.FUNCTION.value == "function"
    assert NodeType.METHOD.value == "method"


def test_edge_type_values():
    assert EdgeType.CALLS.value == "calls"
    assert EdgeType.IMPORTS.value == "imports"
    assert EdgeType.INHERITS.value == "inherits"
    assert EdgeType.CONTAINS.value == "contains"


# ── NeuralNode round-trip ──────────────────────────────────────────────────────

def test_node_roundtrip_basic():
    node = _make_node()
    d = node.to_dict()
    restored = NeuralNode.from_dict(d)
    assert restored.id == node.id
    assert restored.name == node.name
    assert restored.node_type == NodeType.FUNCTION
    assert restored.file_path == node.file_path
    assert restored.line_start == 10
    assert restored.line_end == 20


def test_node_roundtrip_all_node_types():
    for nt in NodeType:
        node = _make_node(node_type=nt)
        restored = NeuralNode.from_dict(node.to_dict())
        assert restored.node_type == nt


def test_node_roundtrip_optional_fields():
    node = _make_node(
        summary_short="short summary",
        summary_detailed="detailed explanation",
        signature="def my_func(x: int) -> str",
        docstring="Does something.",
        complexity=5,
        importance=0.75,
        is_public=False,
        decorators=["staticmethod", "property"],
        content_hash="abcdef123456",
        raw_code="def my_func(x): return x",
        has_redacted_content=True,
    )
    restored = NeuralNode.from_dict(node.to_dict())
    assert restored.summary_short == "short summary"
    assert restored.complexity == 5
    assert restored.importance == 0.75
    assert restored.is_public is False
    assert restored.decorators == ["staticmethod", "property"]
    assert restored.has_redacted_content is True


def test_node_summary_mode_roundtrip():
    node = _make_node(summary_mode=SummaryMode.API)
    restored = NeuralNode.from_dict(node.to_dict())
    assert restored.summary_mode == SummaryMode.API


def test_node_to_dict_enums_are_strings():
    node = _make_node()
    d = node.to_dict()
    assert isinstance(d["node_type"], str)
    assert isinstance(d["summary_mode"], str)


# ── NeuralNode.compute_hash ────────────────────────────────────────────────────

def test_compute_hash_consistency():
    node = _make_node()
    h1 = node.compute_hash("def foo(): pass")
    h2 = node.compute_hash("def foo(): pass")
    assert h1 == h2
    assert len(h1) == 16


def test_compute_hash_different_inputs():
    node = _make_node()
    h1 = node.compute_hash("def foo(): pass")
    h2 = node.compute_hash("def bar(): pass")
    assert h1 != h2


def test_compute_hash_sets_field():
    node = _make_node()
    node.compute_hash("some source code")
    assert len(node.content_hash) == 16


# ── NeuralEdge round-trip ──────────────────────────────────────────────────────

def test_edge_roundtrip_basic():
    edge = _make_edge()
    restored = NeuralEdge.from_dict(edge.to_dict())
    assert restored.source_id == "src000001"
    assert restored.target_id == "tgt000002"
    assert restored.edge_type == EdgeType.CALLS
    assert restored.weight == 1.0


def test_edge_roundtrip_all_edge_types():
    for et in EdgeType:
        edge = _make_edge(edge_type=et)
        restored = NeuralEdge.from_dict(edge.to_dict())
        assert restored.edge_type == et


def test_edge_to_dict_enum_is_string():
    edge = _make_edge()
    d = edge.to_dict()
    assert isinstance(d["edge_type"], str)


# ── IndexState round-trip ──────────────────────────────────────────────────────

def test_index_state_roundtrip_defaults():
    state = IndexState()
    restored = IndexState.from_dict(state.to_dict())
    assert restored.total_nodes == 0
    assert restored.total_edges == 0
    assert restored.index_mode == IndexMode.BOTH
    assert restored.stale_files == []


def test_index_state_roundtrip_populated():
    state = IndexState(
        last_full_index="2026-01-01T00:00:00+00:00",
        last_commit_hash="abc123",
        total_nodes=42,
        total_edges=100,
        total_files=5,
        index_mode=IndexMode.AST_ONLY,
        stale_files=["a.py", "b.py"],
    )
    restored = IndexState.from_dict(state.to_dict())
    assert restored.total_nodes == 42
    assert restored.index_mode == IndexMode.AST_ONLY
    assert restored.stale_files == ["a.py", "b.py"]
    assert restored.last_commit_hash == "abc123"
