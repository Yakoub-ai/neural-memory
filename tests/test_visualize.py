"""Tests for the visualize module: hierarchy and vector space HTML generation."""

from __future__ import annotations

import pytest
from pathlib import Path

from neural_memory.models import NeuralNode, NeuralEdge, NodeType, EdgeType, SummaryMode
from neural_memory.embeddings import compute_all_embeddings, is_available
from neural_memory.visualize import (
    generate_hierarchy_html,
    generate_vector_space_html,
    _build_hierarchy_data,
    _pca_project,
    _viz_available,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _node(name: str, node_type: NodeType = NodeType.FUNCTION,
          file_path: str = "mod.py", importance: float = 0.5,
          summary_short: str = "") -> NeuralNode:
    from neural_memory.ts_parser import _node_id
    nid = _node_id(file_path, name, node_type)
    return NeuralNode(
        id=nid, name=name, node_type=node_type, file_path=file_path,
        line_start=1, line_end=10, importance=importance,
        summary_short=summary_short, summary_mode=SummaryMode.HEURISTIC,
    )


def _populate_storage(storage):
    module = _node("mymodule", NodeType.MODULE, "mymodule.py", importance=0.9)
    cls = _node("MyClass", NodeType.CLASS, "mymodule.py", importance=0.8)
    method = _node("MyClass.do_work", NodeType.METHOD, "mymodule.py", importance=0.5,
                   summary_short="does the work")
    fn = _node("helper", NodeType.FUNCTION, "helpers.py", importance=0.3,
               summary_short="a helper function")

    for n in [module, cls, method, fn]:
        storage.upsert_node(n)

    # CONTAINS edges for hierarchy
    storage.upsert_edge(NeuralEdge(source_id=module.id, target_id=cls.id, edge_type=EdgeType.CONTAINS))
    storage.upsert_edge(NeuralEdge(source_id=cls.id, target_id=method.id, edge_type=EdgeType.CONTAINS))

    return [module, cls, method, fn]


# ── _build_hierarchy_data ──────────────────────────────────────────────────────

class TestBuildHierarchyData:
    def test_returns_all_nodes(self, storage):
        nodes = _populate_storage(storage)
        ids, labels, parents, values, colors, hovers = _build_hierarchy_data(storage)
        assert len(ids) == 4
        assert set(ids) == {n.id for n in nodes}

    def test_contains_edge_sets_parent(self, storage):
        nodes = _populate_storage(storage)
        module, cls, method, fn = nodes
        ids, labels, parents, values, colors, hovers = _build_hierarchy_data(storage)
        idx_map = {nid: i for i, nid in enumerate(ids)}
        # cls parent should be module
        assert parents[idx_map[cls.id]] == module.id
        # method parent should be cls
        assert parents[idx_map[method.id]] == cls.id
        # fn has no parent (root)
        assert parents[idx_map[fn.id]] == ""

    def test_values_floored(self, storage):
        low = _node("low_importance", importance=0.0)
        storage.upsert_node(low)
        _, _, _, values, _, _ = _build_hierarchy_data(storage)
        assert all(v >= 0.05 for v in values)

    def test_empty_storage(self, storage):
        ids, *_ = _build_hierarchy_data(storage)
        assert ids == []


# ── _pca_project ───────────────────────────────────────────────────────────────

@pytest.mark.skipif(not is_available(), reason="numpy not installed")
class TestPCAProject:
    def test_output_shape_2d(self):
        import numpy as np
        matrix = np.random.rand(10, 50).astype(np.float32)
        projected = _pca_project(matrix, dims=2)
        assert projected.shape == (10, 2)

    def test_output_shape_3d(self):
        import numpy as np
        matrix = np.random.rand(8, 30).astype(np.float32)
        projected = _pca_project(matrix, dims=3)
        assert projected.shape == (8, 3)

    def test_fewer_samples_than_dims(self):
        import numpy as np
        matrix = np.random.rand(3, 50).astype(np.float32)
        # SVD will give at most 3 components — should not crash
        projected = _pca_project(matrix, dims=2)
        assert projected.shape == (3, 2)


# ── HTML generation ────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _viz_available(), reason="plotly/numpy not installed")
class TestGenerateHierarchyHTML:
    def test_creates_file(self, storage, tmp_path):
        _populate_storage(storage)
        out = str(tmp_path / "hierarchy.html")
        result = generate_hierarchy_html(storage, out)
        assert result == out
        assert Path(out).exists()

    def test_html_content(self, storage, tmp_path):
        _populate_storage(storage)
        out = str(tmp_path / "hierarchy.html")
        generate_hierarchy_html(storage, out)
        content = Path(out).read_text(encoding="utf-8")
        assert "plotly" in content.lower()
        assert "Neural Memory" in content
        assert "MyClass" in content

    def test_empty_storage_returns_error(self, storage, tmp_path):
        out = str(tmp_path / "h.html")
        result = generate_hierarchy_html(storage, out)
        assert "Error" in result


@pytest.mark.skipif(not _viz_available(), reason="plotly/numpy not installed")
class TestGenerateVectorSpaceHTML:
    def _setup(self, storage):
        nodes = _populate_storage(storage)
        compute_all_embeddings(storage, nodes)
        return nodes

    def test_creates_file_2d(self, storage, tmp_path):
        self._setup(storage)
        out = str(tmp_path / "vectors.html")
        result = generate_vector_space_html(storage, out, dimensions=2)
        assert result == out
        assert Path(out).exists()

    def test_creates_file_3d(self, storage, tmp_path):
        self._setup(storage)
        out = str(tmp_path / "vectors3d.html")
        result = generate_vector_space_html(storage, out, dimensions=3)
        assert result == out

    def test_html_content_mentions_nodes(self, storage, tmp_path):
        self._setup(storage)
        out = str(tmp_path / "vectors.html")
        generate_vector_space_html(storage, out, dimensions=2)
        content = Path(out).read_text(encoding="utf-8")
        assert "plotly" in content.lower()
        assert "Neural Memory" in content

    def test_no_embeddings_returns_error(self, storage, tmp_path):
        # Storage without embeddings
        out = str(tmp_path / "v.html")
        result = generate_vector_space_html(storage, out)
        assert "Error" in result

    def test_color_by_file(self, storage, tmp_path):
        self._setup(storage)
        out = str(tmp_path / "vectors_file.html")
        result = generate_vector_space_html(storage, out, dimensions=2, color_by="file")
        assert result == out
        assert Path(out).exists()
