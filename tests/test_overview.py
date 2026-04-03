"""Tests for neural_memory.overview — project and directory overview generation."""

import pytest
from neural_memory.models import NeuralNode, NodeType, NeuralEdge, EdgeType
from neural_memory.overview import generate_project_overview, generate_directory_overviews, generate_and_store_overviews
from neural_memory.storage import Storage


def _make_node(name, file_path, node_type=NodeType.FUNCTION, importance=0.5):
    from neural_memory.models import SummaryMode
    n = NeuralNode(
        id=f"test::{file_path}::{name}",
        name=name,
        node_type=node_type,
        file_path=file_path,
        line_start=1,
        line_end=10,
        summary_short=f"Short summary of {name}",
        summary_mode=SummaryMode.HEURISTIC,
        importance=importance,
    )
    return n


@pytest.fixture
def populated_storage(tmp_path):
    s = Storage(str(tmp_path))
    s.open()
    nodes = [
        _make_node("module_a", "pkg/a.py", NodeType.MODULE, importance=0.8),
        _make_node("ClassA", "pkg/a.py", NodeType.CLASS, importance=0.7),
        _make_node("method_x", "pkg/a.py", NodeType.METHOD, importance=0.4),
        _make_node("module_b", "pkg/b.py", NodeType.MODULE, importance=0.6),
        _make_node("helper", "pkg/b.py", NodeType.FUNCTION, importance=0.3),
        _make_node("util", "utils/c.py", NodeType.FUNCTION, importance=0.5),
    ]
    for n in nodes:
        s.upsert_node(n)
    yield s
    s.close()


class TestGenerateProjectOverview:
    def test_returns_node(self, populated_storage):
        node = generate_project_overview(populated_storage)
        assert isinstance(node, NeuralNode)

    def test_node_type(self, populated_storage):
        node = generate_project_overview(populated_storage)
        assert node.node_type == NodeType.PROJECT_OVERVIEW

    def test_category_codebase(self, populated_storage):
        node = generate_project_overview(populated_storage)
        assert node.category == "codebase"

    def test_high_importance(self, populated_storage):
        node = generate_project_overview(populated_storage)
        assert node.importance >= 0.9

    def test_summary_mentions_counts(self, populated_storage):
        node = generate_project_overview(populated_storage)
        summary = (node.summary_short or "") + (node.summary_detailed or "")
        # Should mention some kind of count info
        assert any(c.isdigit() for c in summary)

    def test_deterministic_id(self, populated_storage):
        n1 = generate_project_overview(populated_storage)
        n2 = generate_project_overview(populated_storage)
        assert n1.id == n2.id

    def test_file_path_is_project_marker(self, populated_storage):
        node = generate_project_overview(populated_storage)
        assert node.file_path and "__" in node.file_path


class TestGenerateDirectoryOverviews:
    def test_returns_list(self, populated_storage):
        nodes = generate_directory_overviews(populated_storage)
        assert isinstance(nodes, list)

    def test_one_per_directory(self, populated_storage):
        nodes = generate_directory_overviews(populated_storage)
        dirs = {n.file_path for n in nodes}
        # We have pkg/ and utils/ directories
        assert len(dirs) >= 2

    def test_node_types(self, populated_storage):
        nodes = generate_directory_overviews(populated_storage)
        for n in nodes:
            assert n.node_type == NodeType.DIRECTORY_OVERVIEW

    def test_category_codebase(self, populated_storage):
        nodes = generate_directory_overviews(populated_storage)
        for n in nodes:
            assert n.category == "codebase"

    def test_importance_high(self, populated_storage):
        nodes = generate_directory_overviews(populated_storage)
        for n in nodes:
            assert n.importance > 0.5


class TestGenerateAndStoreOverviews:
    def test_returns_stats(self, populated_storage):
        stats = generate_and_store_overviews(populated_storage)
        assert "project_overviews" in stats
        assert "directory_overviews" in stats

    def test_nodes_written_to_storage(self, populated_storage):
        generate_and_store_overviews(populated_storage)
        all_nodes = populated_storage.get_all_nodes()
        types = {n.node_type for n in all_nodes}
        assert NodeType.PROJECT_OVERVIEW in types
        assert NodeType.DIRECTORY_OVERVIEW in types

    def test_idempotent(self, populated_storage):
        generate_and_store_overviews(populated_storage)
        stats1 = generate_and_store_overviews(populated_storage)
        # Running twice should not duplicate overviews
        overviews = [n for n in populated_storage.get_all_nodes()
                     if n.node_type in (NodeType.PROJECT_OVERVIEW, NodeType.DIRECTORY_OVERVIEW)]
        # Count unique ids
        ids = {n.id for n in overviews}
        assert len(ids) == len(overviews)

    def test_stale_overviews_cleaned(self, populated_storage):
        """After deleting a file's nodes and regenerating, old dir overview removed."""
        generate_and_store_overviews(populated_storage)
        before = len([n for n in populated_storage.get_all_nodes()
                      if n.node_type == NodeType.DIRECTORY_OVERVIEW])
        # Remove nodes for utils/c.py
        populated_storage.delete_nodes_by_file("utils/c.py")
        generate_and_store_overviews(populated_storage)
        after_nodes = populated_storage.get_all_nodes()
        dir_overviews = [n for n in after_nodes if n.node_type == NodeType.DIRECTORY_OVERVIEW]
        dir_files = {n.file_path for n in dir_overviews}
        # utils/ should not appear if no nodes remain in that dir
        assert not any("utils" in (f or "") for f in dir_files)
