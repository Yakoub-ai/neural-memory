"""Tests for neural_memory.context — token-budgeted context builder."""

import pytest

from neural_memory.config import NeuralConfig, save_config
from neural_memory.models import IndexMode, NodeType, NeuralNode
from neural_memory.indexer import full_index
from neural_memory.storage import Storage
from neural_memory.context import build_context


@pytest.fixture
def indexed_project(tmp_path):
    """A tmp_path that has been fully indexed."""
    (tmp_path / "sample.py").write_text(
        'def greet(name: str) -> str:\n    """Return a greeting."""\n    return f"Hello, {name}!"\n\n'
        'class Calculator:\n    """A simple calculator."""\n    def add(self, a, b):\n        return a + b\n'
    )
    cfg = NeuralConfig(project_root=str(tmp_path), index_mode=IndexMode.AST_ONLY)
    save_config(cfg)
    full_index(config=cfg)
    return tmp_path


def test_build_context_uninitialized(tmp_path):
    result = build_context(project_root=str(tmp_path))
    # Should not crash and should contain the wrapper
    assert "<!-- neural-memory context -->" in result
    assert "<!-- /neural-memory -->" in result
    # Uninitialized project has no index — status line uses one of these phrases
    assert any(w in result for w in ("uninitialized", "unavailable", "not been initialized", "not indexed"))


def test_build_context_with_index(indexed_project):
    result = build_context(project_root=str(indexed_project))
    assert "<!-- neural-memory context -->" in result
    assert "<!-- /neural-memory -->" in result
    # Should report healthy or at least provide a status line
    assert "Neural memory" in result


def test_build_context_token_budget_scaling(indexed_project):
    small = build_context(project_root=str(indexed_project), token_budget=100)
    large = build_context(project_root=str(indexed_project), token_budget=2000)
    # Larger budget should produce more (or equal) content
    assert len(large) >= len(small)


def test_build_context_active_bugs(indexed_project):
    # Insert an active bug node directly
    bug = NeuralNode(
        id="bug_test_001",
        name="NullPointerBug",
        node_type=NodeType.BUG,
        file_path="src/foo.py",
        line_start=10,
        line_end=10,
        category="bugs",
        bug_status="open",
        severity="high",
        summary_short="Null pointer in parser",
        importance=0.9,
    )
    with Storage(str(indexed_project)) as storage:
        storage.upsert_node(bug)

    result = build_context(project_root=str(indexed_project))
    assert "Active bugs" in result
    assert "NullPointerBug" in result or "Null pointer" in result
