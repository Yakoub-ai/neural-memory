"""Tests for neural_memory.graph — importance scoring, neighborhood, call chains."""

import pytest

from neural_memory.models import NeuralNode, NeuralEdge, NodeType, EdgeType
from neural_memory.graph import (
    compute_importance,
    get_neighborhood,
    trace_call_chain,
    format_node_summary,
)
from neural_memory.storage import Storage


def _node(id, name="func", node_type=NodeType.FUNCTION, file_path="a.py",
          line_start=1, line_end=5, is_public=True) -> NeuralNode:
    return NeuralNode(
        id=id, name=name, node_type=node_type, file_path=file_path,
        line_start=line_start, line_end=line_end, is_public=is_public,
    )


def _edge(src, tgt, edge_type=EdgeType.CALLS) -> NeuralEdge:
    return NeuralEdge(source_id=src, target_id=tgt, edge_type=edge_type)


def _setup_graph(storage: Storage, nodes, edges):
    for n in nodes:
        storage.upsert_node(n)
    for e in edges:
        storage.upsert_edge(e)


# ── compute_importance ─────────────────────────────────────────────────────────

def test_importance_all_scores_in_range(storage):
    _setup_graph(storage, [
        _node("n1"), _node("n2", name="n2"), _node("n3", name="n3"),
    ], [
        _edge("n1", "n2"), _edge("n3", "n1"),
    ])
    compute_importance(storage)
    for nid in ["n1", "n2", "n3"]:
        node = storage.get_node(nid)
        assert 0.0 <= node.importance <= 1.0


def test_importance_most_connected_node_gets_highest(storage):
    # n1 is called by n2 and n3, so it has highest in-degree
    _setup_graph(storage, [
        _node("n1"), _node("n2", name="n2"), _node("n3", name="n3"),
    ], [
        _edge("n2", "n1"), _edge("n3", "n1"),
    ])
    compute_importance(storage)
    n1 = storage.get_node("n1")
    n2 = storage.get_node("n2")
    n3 = storage.get_node("n3")
    assert n1.importance >= n2.importance
    assert n1.importance >= n3.importance


def test_importance_class_outweighs_function_at_same_connectivity(storage):
    _setup_graph(storage, [
        _node("nc", name="MyClass", node_type=NodeType.CLASS),
        _node("nf", name="my_func", node_type=NodeType.FUNCTION),
    ], [])
    compute_importance(storage)
    nc = storage.get_node("nc")
    nf = storage.get_node("nf")
    assert nc.importance >= nf.importance


def test_importance_public_bonus(storage):
    _setup_graph(storage, [
        _node("pub", name="public_func", is_public=True),
        _node("prv", name="_private_func", is_public=False),
    ], [])
    compute_importance(storage)
    pub = storage.get_node("pub")
    prv = storage.get_node("prv")
    assert pub.importance >= prv.importance


def test_importance_single_node_gets_nonzero(storage):
    storage.upsert_node(_node("n1"))
    compute_importance(storage)
    node = storage.get_node("n1")
    assert node.importance > 0.0


def test_importance_updates_stored_nodes(storage):
    storage.upsert_node(_node("n1"))
    compute_importance(storage)
    node = storage.get_node("n1")
    # After compute_importance, the stored node should have been updated
    assert node.importance is not None


# ── get_neighborhood ────────────────────────────────────────────────────────────

def test_neighborhood_center_found(storage):
    storage.upsert_node(_node("n1"))
    result = get_neighborhood(storage, "n1")
    assert "center" in result
    assert result["center"].id == "n1"


def test_neighborhood_missing_node(storage):
    result = get_neighborhood(storage, "does_not_exist")
    assert "error" in result


def test_neighborhood_callers_populated(storage):
    _setup_graph(storage, [
        _node("caller"), _node("callee", name="callee"),
    ], [_edge("caller", "callee", EdgeType.CALLS)])
    result = get_neighborhood(storage, "callee")
    caller_ids = [n.id for n in result["callers"]]
    assert "caller" in caller_ids


def test_neighborhood_callees_populated(storage):
    _setup_graph(storage, [
        _node("caller"), _node("callee", name="callee"),
    ], [_edge("caller", "callee", EdgeType.CALLS)])
    result = get_neighborhood(storage, "caller")
    callee_ids = [n.id for n in result["callees"]]
    assert "callee" in callee_ids


def test_neighborhood_parent_via_contains_edge(storage):
    parent = _node("mod", name="mymodule", node_type=NodeType.MODULE)
    child = _node("func", name="my_func")
    _setup_graph(storage, [parent, child], [_edge("mod", "func", EdgeType.CONTAINS)])
    result = get_neighborhood(storage, "func")
    assert result["parent"] is not None
    assert result["parent"].id == "mod"


def test_neighborhood_children_via_contains_edge(storage):
    parent = _node("mod", name="mymodule", node_type=NodeType.MODULE)
    child = _node("func", name="my_func")
    _setup_graph(storage, [parent, child], [_edge("mod", "func", EdgeType.CONTAINS)])
    result = get_neighborhood(storage, "mod")
    child_ids = [n.id for n in result["children"]]
    assert "func" in child_ids


def test_neighborhood_siblings_share_parent(storage):
    parent = _node("mod", name="mod", node_type=NodeType.MODULE)
    sib1 = _node("s1", name="sibling1")
    sib2 = _node("s2", name="sibling2")
    _setup_graph(storage, [parent, sib1, sib2], [
        _edge("mod", "s1", EdgeType.CONTAINS),
        _edge("mod", "s2", EdgeType.CONTAINS),
    ])
    result = get_neighborhood(storage, "s1")
    sibling_ids = [n.id for n in result["siblings"]]
    assert "s2" in sibling_ids
    assert "s1" not in sibling_ids  # Self not included


# ── trace_call_chain ────────────────────────────────────────────────────────────

def test_trace_call_chain_down(storage):
    _setup_graph(storage, [
        _node("a", name="a"), _node("b", name="b"), _node("c", name="c"),
    ], [
        _edge("a", "b", EdgeType.CALLS), _edge("b", "c", EdgeType.CALLS),
    ])
    chains = trace_call_chain(storage, "a", direction="down")
    all_ids = {n.id for chain in chains for n in chain}
    assert "a" in all_ids
    assert "b" in all_ids


def test_trace_call_chain_up(storage):
    _setup_graph(storage, [
        _node("a", name="a"), _node("b", name="b"), _node("c", name="c"),
    ], [
        _edge("a", "b", EdgeType.CALLS), _edge("b", "c", EdgeType.CALLS),
    ])
    chains = trace_call_chain(storage, "c", direction="up")
    all_ids = {n.id for chain in chains for n in chain}
    assert "c" in all_ids
    assert "b" in all_ids


def test_trace_call_chain_max_depth(storage):
    # Chain: a -> b -> c -> d -> e (5 nodes)
    ids = ["a", "b", "c", "d", "e"]
    nodes = [_node(i, name=i) for i in ids]
    edges = [_edge(ids[i], ids[i + 1], EdgeType.CALLS) for i in range(4)]
    _setup_graph(storage, nodes, edges)

    chains = trace_call_chain(storage, "a", direction="down", max_depth=2)
    # With max_depth=2, chains should not go all the way to "e"
    all_ids = {n.id for chain in chains for n in chain}
    assert "e" not in all_ids


def test_trace_call_chain_no_infinite_loop_on_cycle(storage):
    _setup_graph(storage, [
        _node("x", name="x"), _node("y", name="y"),
    ], [
        _edge("x", "y", EdgeType.CALLS), _edge("y", "x", EdgeType.CALLS),
    ])
    # Should terminate without error
    chains = trace_call_chain(storage, "x", direction="down")
    assert isinstance(chains, list)


# ── format_node_summary ─────────────────────────────────────────────────────────

def test_format_node_summary_short(storage):
    node = _node("n1", name="my_func")
    node.summary_short = "Does something."
    result = format_node_summary(node, level="short")
    assert "my_func" in result
    assert "Does something" in result


def test_format_node_summary_detailed_differs_from_short(storage):
    node = _node("n1", name="my_func")
    node.signature = "def my_func(x: int) -> str"
    node.summary_detailed = "Full explanation here."
    short = format_node_summary(node, level="short")
    detailed = format_node_summary(node, level="detailed")
    assert short != detailed
    assert "my_func(x: int)" in detailed
