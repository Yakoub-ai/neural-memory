"""Tests for neural_memory.dashboard — HTML generation."""

import json
import pytest

from neural_memory.dashboard import (
    generate_dashboard_html,
    _extract_data,
    _pca_positions,
    _build_hierarchy,
    _html_head,
    _html_body,
)
from neural_memory.models import NeuralNode, NodeType, NeuralEdge, EdgeType, SummaryMode
from neural_memory.storage import Storage


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_node(name, file_path="mod.py", node_type=NodeType.FUNCTION, importance=0.5, category="codebase"):
    return NeuralNode(
        id=f"test::{file_path}::{name}",
        name=name,
        node_type=node_type,
        file_path=file_path,
        line_start=1,
        line_end=10,
        summary_short=f"Does {name} things",
        summary_mode=SummaryMode.HEURISTIC,
        importance=importance,
        category=category,
    )


@pytest.fixture
def small_storage(tmp_path):
    s = Storage(str(tmp_path))
    s.open()
    nodes = [
        _make_node("main_module", "main.py", NodeType.MODULE, importance=0.9),
        _make_node("MyClass", "main.py", NodeType.CLASS, importance=0.7),
        _make_node("process", "main.py", NodeType.FUNCTION, importance=0.5),
        _make_node("bug_login", "auth.py", NodeType.BUG, importance=0.4, category="bugs"),
        _make_node("task_refactor", "tasks.py", NodeType.TASK, importance=0.3, category="tasks"),
    ]
    for n in nodes:
        s.upsert_node(n)

    # Add an edge
    edge = NeuralEdge(
        source_id=nodes[1].id,
        target_id=nodes[2].id,
        edge_type=EdgeType.CONTAINS,
    )
    s.upsert_edge(edge)

    yield s
    s.close()


# ── _extract_data ─────────────────────────────────────────────────────────────

class TestExtractData:
    def test_returns_dict_with_required_keys(self, small_storage):
        data = _extract_data(small_storage)
        assert "nodes" in data
        assert "edges" in data
        assert "hierarchy" in data
        assert "stats" in data

    def test_nodes_list(self, small_storage):
        data = _extract_data(small_storage)
        assert isinstance(data["nodes"], list)
        assert len(data["nodes"]) == 5

    def test_node_has_required_fields(self, small_storage):
        data = _extract_data(small_storage)
        for n in data["nodes"]:
            assert "id" in n
            assert "name" in n
            assert "node_type" in n
            assert "category" in n
            assert "importance" in n
            assert "px" in n
            assert "py" in n

    def test_no_embedding_in_output(self, small_storage):
        data = _extract_data(small_storage)
        for n in data["nodes"]:
            assert "embedding" not in n

    def test_edges_list(self, small_storage):
        data = _extract_data(small_storage)
        assert isinstance(data["edges"], list)
        assert len(data["edges"]) >= 1

    def test_edge_has_required_fields(self, small_storage):
        data = _extract_data(small_storage)
        for e in data["edges"]:
            assert "source_id" in e
            assert "target_id" in e
            assert "edge_type" in e

    def test_categories_present(self, small_storage):
        data = _extract_data(small_storage)
        cats = {n["category"] for n in data["nodes"]}
        assert "codebase" in cats
        assert "bugs" in cats
        assert "tasks" in cats


# ── _pca_positions ────────────────────────────────────────────────────────────

class TestPcaPositions:
    def test_circular_fallback(self):
        nodes = [{"id": f"n{i}", "embedding": None} for i in range(4)]
        result = _pca_positions(nodes)
        assert len(result) == 4
        for nid, pos in result.items():
            assert len(pos) == 2
            assert -1.0 <= pos[0] <= 1.0
            assert -1.0 <= pos[1] <= 1.0

    def test_returns_dict(self):
        nodes = [{"id": "a", "embedding": None}, {"id": "b", "embedding": None}]
        result = _pca_positions(nodes)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"a", "b"}

    def test_with_numpy_embeddings(self):
        pytest.importorskip("numpy")
        import numpy as np
        rng = np.random.default_rng(42)
        nodes = [{"id": f"n{i}", "embedding": rng.random(32).tolist()} for i in range(10)]
        result = _pca_positions(nodes)
        assert len(result) == 10
        for pos in result.values():
            assert len(pos) == 2


# ── _build_hierarchy ──────────────────────────────────────────────────────────

class TestBuildHierarchy:
    def test_single_node(self):
        nodes = [{"id": "n1", "name": "main", "node_type": "module", "category": "codebase", "importance": 0.8}]
        result = _build_hierarchy(nodes, [])
        assert result["id"] == "n1"
        assert result["name"] == "main"

    def test_parent_child(self):
        nodes = [
            {"id": "parent", "name": "P", "node_type": "module", "category": "codebase", "importance": 0.8},
            {"id": "child", "name": "C", "node_type": "function", "category": "codebase", "importance": 0.4},
        ]
        edges = [{"source_id": "parent", "target_id": "child", "edge_type": "contains"}]
        result = _build_hierarchy(nodes, edges)
        assert result["id"] == "parent"
        assert "children" in result
        assert result["children"][0]["id"] == "child"

    def test_multiple_roots(self):
        nodes = [
            {"id": "a", "name": "A", "node_type": "module", "category": "codebase", "importance": 0.5},
            {"id": "b", "name": "B", "node_type": "module", "category": "codebase", "importance": 0.5},
        ]
        result = _build_hierarchy(nodes, [])
        assert result["id"] == "__root__"
        assert len(result["children"]) == 2


# ── generate_dashboard_html ───────────────────────────────────────────────────

class TestGenerateDashboardHtml:
    def test_returns_string(self, small_storage):
        html = generate_dashboard_html(small_storage)
        assert isinstance(html, str)
        assert len(html) > 1000

    def test_is_valid_html(self, small_storage):
        html = generate_dashboard_html(small_storage)
        assert html.strip().startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_contains_doctype_and_charset(self, small_storage):
        html = generate_dashboard_html(small_storage)
        assert "UTF-8" in html

    def test_node_data_embedded(self, small_storage):
        html = generate_dashboard_html(small_storage)
        # Node names should appear in the embedded JSON
        assert "main_module" in html
        assert "MyClass" in html

    def test_all_categories_in_data(self, small_storage):
        html = generate_dashboard_html(small_storage)
        assert '"codebase"' in html
        assert '"bugs"' in html
        assert '"tasks"' in html

    def test_echarts_reference_present(self, small_storage):
        html = generate_dashboard_html(small_storage, project_root=".")
        # Either inline ECharts or a CDN reference
        assert "echarts" in html.lower()

    def test_tabs_present(self, small_storage):
        html = generate_dashboard_html(small_storage)
        assert "Hierarchy" in html
        assert "Semantic" in html  # renamed from Vectors
        assert "Graph" in html

    def test_writes_file(self, small_storage, tmp_path):
        out = str(tmp_path / "test_dashboard.html")
        generate_dashboard_html(small_storage, output_path=out)
        from pathlib import Path
        assert Path(out).exists()
        content = Path(out).read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content

    def test_data_is_valid_json(self, small_storage):
        html = generate_dashboard_html(small_storage)
        # Extract the JSON between "var RAW = " and ";\n"
        start = html.find("var RAW = ")
        assert start != -1
        start += len("var RAW = ")
        end = html.find(";\n", start)
        assert end != -1
        data_str = html[start:end]
        data = json.loads(data_str)
        assert "nodes" in data
        assert "edges" in data

    def test_empty_storage(self, tmp_path):
        s = Storage(str(tmp_path))
        s.open()
        try:
            html = generate_dashboard_html(s)
            assert "<!DOCTYPE html>" in html
        finally:
            s.close()


# ── HTML structure helpers ────────────────────────────────────────────────────

class TestHtmlHelpers:
    def test_head_has_style(self):
        head = _html_head()
        assert "<style>" in head
        assert "</style>" in head

    def test_body_has_sidebar(self):
        body = _html_body()
        assert "sidebar" in body

    def test_body_has_three_view_panels(self):
        body = _html_body()
        assert "view-hierarchy" in body
        assert "view-vectors" in body
        assert "view-graph" in body
        # ECharts chart containers
        assert "chart-hierarchy" in body
        assert "chart-vectors" in body
        assert "chart-graph" in body

    def test_body_has_detail_panel(self):
        body = _html_body()
        assert "detail-panel" in body
