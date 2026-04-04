"""Tests for neural_memory.storage — SQLite-backed graph persistence."""

import pytest

from neural_memory.models import NeuralNode, NeuralEdge, NodeType, EdgeType, IndexState, IndexMode
from neural_memory.storage import Storage


def _node(id="n1", name="my_func", node_type=NodeType.FUNCTION, file_path="foo.py",
          line_start=1, line_end=5, importance=0.0) -> NeuralNode:
    return NeuralNode(
        id=id, name=name, node_type=node_type, file_path=file_path,
        line_start=line_start, line_end=line_end, importance=importance,
    )


def _edge(source_id="n1", target_id="n2", edge_type=EdgeType.CALLS) -> NeuralEdge:
    return NeuralEdge(source_id=source_id, target_id=target_id, edge_type=edge_type)


# ── Open / close / context manager ─────────────────────────────────────────────

def test_open_close(tmp_path):
    s = Storage(str(tmp_path))
    s.open()
    assert s.conn is not None
    s.close()
    assert s.conn is None


def test_context_manager(tmp_path):
    with Storage(str(tmp_path)) as s:
        assert s.conn is not None
    assert s.conn is None


# ── Node CRUD ──────────────────────────────────────────────────────────────────

def test_upsert_and_get_node(storage):
    n = _node()
    storage.upsert_node(n)
    got = storage.get_node("n1")
    assert got is not None
    assert got.id == "n1"
    assert got.name == "my_func"
    assert got.node_type == NodeType.FUNCTION


def test_get_node_returns_none_for_missing(storage):
    assert storage.get_node("nonexistent") is None


def test_upsert_is_idempotent(storage):
    n = _node(name="original")
    storage.upsert_node(n)
    n2 = _node(name="updated")
    storage.upsert_node(n2)
    got = storage.get_node("n1")
    assert got.name == "updated"


def test_get_nodes_by_file(storage):
    storage.upsert_node(_node(id="n1", file_path="a.py"))
    storage.upsert_node(_node(id="n2", name="other", file_path="b.py"))
    result = storage.get_nodes_by_file("a.py")
    assert len(result) == 1
    assert result[0].id == "n1"


def test_get_nodes_by_type(storage):
    storage.upsert_node(_node(id="n1", node_type=NodeType.FUNCTION))
    storage.upsert_node(_node(id="n2", name="MyClass", node_type=NodeType.CLASS))
    funcs = storage.get_nodes_by_type(NodeType.FUNCTION)
    assert all(n.node_type == NodeType.FUNCTION for n in funcs)
    assert len(funcs) == 1


def test_delete_nodes_by_file_returns_count(storage):
    storage.upsert_node(_node(id="n1", file_path="a.py"))
    storage.upsert_node(_node(id="n2", name="n2", file_path="a.py"))
    storage.upsert_node(_node(id="n3", name="n3", file_path="b.py"))
    count = storage.delete_nodes_by_file("a.py")
    assert count == 2
    assert storage.get_node("n1") is None
    assert storage.get_node("n3") is not None


def test_get_all_node_ids(storage):
    storage.upsert_node(_node(id="n1"))
    storage.upsert_node(_node(id="n2", name="n2"))
    ids = storage.get_all_node_ids()
    assert set(ids) == {"n1", "n2"}


def test_get_all_nodes_returns_all(storage):
    storage.upsert_node(_node(id="n1"))
    storage.upsert_node(_node(id="n2", name="n2"))
    nodes = storage.get_all_nodes()
    assert {n.id for n in nodes} == {"n1", "n2"}


# ── Search ─────────────────────────────────────────────────────────────────────

def test_search_nodes_by_name(storage):
    storage.upsert_node(_node(id="n1", name="parse_file"))
    storage.upsert_node(_node(id="n2", name="resolve_edges"))
    results = storage.search_nodes("parse")
    assert any(n.id == "n1" for n in results)
    assert all(n.id != "n2" or "resolve" in n.name for n in results)


def test_search_nodes_no_results(storage):
    storage.upsert_node(_node(id="n1", name="my_func"))
    results = storage.search_nodes("xyz_not_found")
    assert results == []


def test_search_nodes_respects_limit(storage):
    for i in range(10):
        storage.upsert_node(_node(id=f"n{i}", name=f"func_{i}"))
    results = storage.search_nodes("func", limit=3)
    assert len(results) <= 3


def test_search_nodes_ordered_by_importance(storage):
    storage.upsert_node(_node(id="n1", name="func_low", importance=0.1))
    storage.upsert_node(_node(id="n2", name="func_high", importance=0.9))
    results = storage.search_nodes("func", limit=10)
    importances = [n.importance for n in results]
    assert importances == sorted(importances, reverse=True)


# ── Edge operations ─────────────────────────────────────────────────────────────

def test_upsert_and_get_edges_from(storage):
    storage.upsert_node(_node(id="n1"))
    storage.upsert_node(_node(id="n2", name="n2"))
    storage.upsert_edge(_edge("n1", "n2", EdgeType.CALLS))
    edges = storage.get_edges_from("n1")
    assert len(edges) == 1
    assert edges[0].target_id == "n2"
    assert edges[0].edge_type == EdgeType.CALLS


def test_upsert_and_get_edges_to(storage):
    storage.upsert_node(_node(id="n1"))
    storage.upsert_node(_node(id="n2", name="n2"))
    storage.upsert_edge(_edge("n1", "n2", EdgeType.CALLS))
    edges = storage.get_edges_to("n2")
    assert len(edges) == 1
    assert edges[0].source_id == "n1"


def test_edge_upsert_no_duplicates(storage):
    storage.upsert_node(_node(id="n1"))
    storage.upsert_node(_node(id="n2", name="n2"))
    storage.upsert_edge(_edge("n1", "n2", EdgeType.CALLS))
    storage.upsert_edge(_edge("n1", "n2", EdgeType.CALLS))
    edges = storage.get_edges_from("n1")
    assert len(edges) == 1


def test_delete_edges_for_node(storage):
    storage.upsert_node(_node(id="n1"))
    storage.upsert_node(_node(id="n2", name="n2"))
    storage.upsert_edge(_edge("n1", "n2"))
    storage.delete_edges_for_node("n1")
    assert storage.get_edges_from("n1") == []
    assert storage.get_edges_to("n2") == []


def test_delete_edges_by_file(storage):
    storage.upsert_node(_node(id="n1", file_path="a.py"))
    storage.upsert_node(_node(id="n2", name="n2", file_path="b.py"))
    storage.upsert_edge(_edge("n1", "n2"))
    storage.delete_edges_by_file("a.py")
    assert storage.get_edges_from("n1") == []


def test_get_all_edges_by_node_structure(storage):
    # Create two nodes and one directed edge
    storage.upsert_node(_node(id="n1", name="caller"))
    storage.upsert_node(_node(id="n2", name="callee"))
    storage.upsert_edge(_edge("n1", "n2", EdgeType.CALLS))

    result = storage.get_all_edges_by_node()

    # Both endpoints must appear with the correct incoming/outgoing buckets
    assert "n1" in result
    assert "n2" in result
    assert len(result["n1"]["outgoing"]) == 1
    assert result["n1"]["outgoing"][0].target_id == "n2"
    assert result["n1"]["incoming"] == []
    assert len(result["n2"]["incoming"]) == 1
    assert result["n2"]["incoming"][0].source_id == "n1"
    assert result["n2"]["outgoing"] == []


def test_get_all_edges_by_node_isolated_node_absent(storage):
    # A node with no edges should not appear in the result at all
    storage.upsert_node(_node(id="lone", name="isolated"))
    result = storage.get_all_edges_by_node()
    assert "lone" not in result


# ── Index state ────────────────────────────────────────────────────────────────

def test_get_index_state_returns_default(storage):
    state = storage.get_index_state()
    assert isinstance(state, IndexState)
    assert state.total_nodes == 0


def test_save_and_get_index_state(storage):
    state = IndexState(total_nodes=42, total_edges=100, index_mode=IndexMode.AST_ONLY)
    storage.save_index_state(state)
    got = storage.get_index_state()
    assert got.total_nodes == 42
    assert got.total_edges == 100
    assert got.index_mode == IndexMode.AST_ONLY


# ── File hashes ────────────────────────────────────────────────────────────────

def test_save_and_get_file_hash(storage):
    storage.save_file_hash("foo.py", "abc123", "2026-01-01T00:00:00Z")
    got = storage.get_file_hash("foo.py")
    assert got == "abc123"


def test_get_file_hash_missing(storage):
    assert storage.get_file_hash("missing.py") is None


def test_get_all_indexed_files(storage):
    storage.save_file_hash("a.py", "hash1", "2026-01-01T00:00:00Z")
    storage.save_file_hash("b.py", "hash2", "2026-01-01T00:00:00Z")
    files = storage.get_all_indexed_files()
    assert files == {"a.py": "hash1", "b.py": "hash2"}


def test_delete_file_hash(storage):
    storage.save_file_hash("a.py", "h1", "2026-01-01T00:00:00Z")
    storage.delete_file_hash("a.py")
    assert storage.get_file_hash("a.py") is None


# ── Stats ──────────────────────────────────────────────────────────────────────

def test_get_stats_accuracy(storage):
    storage.upsert_node(_node(id="n1", file_path="a.py", node_type=NodeType.FUNCTION))
    storage.upsert_node(_node(id="n2", name="n2", file_path="a.py", node_type=NodeType.CLASS))
    storage.upsert_node(_node(id="n3", name="n3", file_path="b.py", node_type=NodeType.FUNCTION))
    storage.upsert_edge(_edge("n1", "n2"))

    stats = storage.get_stats()
    assert stats["total_nodes"] == 3
    assert stats["total_edges"] == 1
    assert stats["total_files"] == 2
    assert stats["nodes_by_type"]["function"] == 2
    assert stats["nodes_by_type"]["class"] == 1


# ── Batch operations ────────────────────────────────────────────────────────────

def test_batch_upsert_nodes(storage):
    nodes = [
        _node(id="b1", name="func1", file_path="batch.py"),
        _node(id="b2", name="func2", file_path="batch.py"),
        _node(id="b3", name="func3", file_path="other.py"),
    ]
    storage.batch_upsert_nodes(nodes)
    assert len(storage.get_all_nodes()) == 3
    assert storage.get_node("b2").name == "func2"


def test_batch_upsert_nodes_empty(storage):
    storage.batch_upsert_nodes([])
    assert storage.get_all_nodes() == []


def test_batch_upsert_edges(storage):
    storage.batch_upsert_nodes([
        _node(id="e1", name="a"), _node(id="e2", name="b"), _node(id="e3", name="c"),
    ])
    edges = [
        _edge("e1", "e2", EdgeType.CALLS),
        _edge("e2", "e3", EdgeType.IMPORTS),
    ]
    storage.batch_upsert_edges(edges)
    assert len(storage.get_all_edges()) == 2


def test_batch_upsert_edges_empty(storage):
    storage.batch_upsert_edges([])
    assert storage.get_all_edges() == []


def test_batch_save_file_hashes(storage):
    entries = [
        ("file_a.py", "abc123", "2024-01-01"),
        ("file_b.py", "def456", "2024-01-01"),
    ]
    storage.batch_save_file_hashes(entries)
    indexed = storage.get_all_indexed_files()
    assert indexed["file_a.py"] == "abc123"
    assert indexed["file_b.py"] == "def456"


def test_batch_save_file_hashes_upsert(storage):
    storage.batch_save_file_hashes([("f.py", "old_hash", "2024-01-01")])
    storage.batch_save_file_hashes([("f.py", "new_hash", "2024-01-02")])
    assert storage.get_all_indexed_files()["f.py"] == "new_hash"


def test_transaction_rollback_on_error(storage):
    storage.upsert_node(_node(id="tx1", name="existing"))
    try:
        with storage.transaction():
            storage.conn.execute(
                "INSERT INTO nodes (id, name, node_type, file_path, line_start, line_end, importance, category, data) "
                "VALUES ('tx2', 'new', 'function', 'f.py', 1, 2, 0.0, 'codebase', '{}')"
            )
            raise ValueError("forced error")
    except ValueError:
        pass
    # tx2 should have been rolled back
    assert storage.get_node("tx2") is None
    assert storage.get_node("tx1") is not None


def test_get_all_degree_counts(storage):
    storage.batch_upsert_nodes([
        _node(id="d1", name="a"), _node(id="d2", name="b"), _node(id="d3", name="c"),
    ])
    storage.batch_upsert_edges([
        _edge("d1", "d2", EdgeType.CALLS),
        _edge("d1", "d3", EdgeType.CALLS),
        _edge("d2", "d3", EdgeType.IMPORTS),
    ])
    counts = storage.get_all_degree_counts()
    assert counts["d1"] == (0, 2)  # in=0, out=2
    assert counts["d2"] == (1, 1)  # in=1, out=1
    assert counts["d3"] == (2, 0)  # in=2, out=0


# ── Package docs ────────────────────────────────────────────────────────────────

def test_upsert_and_get_package_doc(storage):
    data = {
        "version": "2.31.0",
        "summary": "HTTP for Humans",
        "description": "A simple HTTP library.",
        "homepage_url": "https://requests.readthedocs.io",
        "doc_url": "https://docs.python-requests.org",
    }
    storage.upsert_package_doc("requests", "pypi", data, "2024-01-01T00:00:00Z")

    result = storage.get_package_doc("requests", "pypi")
    assert result is not None
    assert result["package_name"] == "requests"
    assert result["registry"] == "pypi"
    assert result["version"] == "2.31.0"
    assert result["summary"] == "HTTP for Humans"
    assert result["homepage_url"] == "https://requests.readthedocs.io"
    assert result["fetched_at"] == "2024-01-01T00:00:00Z"


def test_get_package_doc_missing_returns_none(storage):
    result = storage.get_package_doc("nonexistent-pkg", "pypi")
    assert result is None


def test_get_all_package_docs(storage):
    storage.upsert_package_doc(
        "requests", "pypi",
        {"version": "2.31.0", "summary": "s", "description": "", "homepage_url": "", "doc_url": ""},
        "2024-01-01T00:00:00Z",
    )
    storage.upsert_package_doc(
        "lodash", "npm",
        {"version": "4.17.21", "summary": "s", "description": "", "homepage_url": "", "doc_url": ""},
        "2024-01-01T00:00:00Z",
    )

    docs = storage.get_all_package_docs()
    assert len(docs) == 2
    names = {d["package_name"] for d in docs}
    assert names == {"requests", "lodash"}


def test_upsert_package_doc_updates_existing(storage):
    data_v1 = {
        "version": "1.0.0",
        "summary": "Old summary",
        "description": "",
        "homepage_url": "",
        "doc_url": "",
    }
    storage.upsert_package_doc("mypkg", "pypi", data_v1, "2024-01-01T00:00:00Z")

    data_v2 = {
        "version": "2.0.0",
        "summary": "New summary",
        "description": "",
        "homepage_url": "",
        "doc_url": "",
    }
    storage.upsert_package_doc("mypkg", "pypi", data_v2, "2024-06-01T00:00:00Z")

    result = storage.get_package_doc("mypkg", "pypi")
    assert result is not None
    assert result["version"] == "2.0.0"
    assert result["summary"] == "New summary"
    assert result["fetched_at"] == "2024-06-01T00:00:00Z"

    # Should still be only one row
    all_docs = storage.get_all_package_docs()
    pypi_docs = [d for d in all_docs if d["package_name"] == "mypkg"]
    assert len(pypi_docs) == 1


# ── Archive / lifecycle ────────────────────────────────────────────────────────

def _bug_node(id="b1", bug_status="open", importance=1.0) -> NeuralNode:
    from neural_memory.models import NodeType
    return NeuralNode(
        id=id, name=f"bug_{id}", node_type=NodeType.BUG,
        file_path="src/foo.py", line_start=1, line_end=5,
        category="bugs", bug_status=bug_status, importance=importance,
    )


def _task_node(id="t1", task_status="pending", importance=0.8) -> NeuralNode:
    from neural_memory.models import NodeType
    return NeuralNode(
        id=id, name=f"task_{id}", node_type=NodeType.TASK,
        file_path="src/foo.py", line_start=1, line_end=5,
        category="tasks", task_status=task_status, importance=importance,
    )


def test_get_active_items_returns_non_archived(storage):
    storage.upsert_node(_bug_node(id="b1"))
    results = storage.get_active_items("bugs")
    assert any(n.id == "b1" for n in results)


def test_get_active_items_excludes_archived(storage):
    n = _bug_node(id="b2")
    n.archived = True
    storage.upsert_node(n)
    results = storage.get_active_items("bugs")
    assert not any(n.id == "b2" for n in results)


def test_archive_node_sets_archived_and_decays_importance(storage):
    storage.upsert_node(_bug_node(id="b3", importance=1.0))
    success = storage.archive_node("b3")
    assert success is True
    node = storage.get_node("b3")
    assert node.archived is True
    assert abs(node.importance - 0.3) < 0.001


def test_archive_node_returns_false_for_missing(storage):
    assert storage.archive_node("nonexistent_id") is False


def test_unarchive_node_clears_archived(storage):
    storage.upsert_node(_bug_node(id="b4", importance=1.0))
    storage.archive_node("b4")
    success = storage.unarchive_node("b4")
    assert success is True
    node = storage.get_node("b4")
    assert node.archived is False
    # Importance stays decayed — by design
    assert node.importance < 1.0


def test_archive_completed_archives_done_tasks(storage):
    storage.upsert_node(_task_node(id="t1", task_status="done"))
    count = storage.archive_completed()
    assert count == 1
    node = storage.get_node("t1")
    assert node.archived is True


def test_archive_completed_archives_fixed_bugs(storage):
    storage.upsert_node(_bug_node(id="b5", bug_status="fixed"))
    count = storage.archive_completed()
    assert count == 1
    node = storage.get_node("b5")
    assert node.archived is True


def test_archive_completed_skips_active_items(storage):
    storage.upsert_node(_bug_node(id="b6", bug_status="open"))
    storage.upsert_node(_task_node(id="t2", task_status="in_progress"))
    count = storage.archive_completed()
    assert count == 0


def test_schema_migration_adds_archived_column(tmp_path):
    with Storage(str(tmp_path)) as s:
        cols = s.conn.execute("PRAGMA table_info(nodes)").fetchall()
        col_names = [c["name"] for c in cols]
    assert "archived" in col_names
