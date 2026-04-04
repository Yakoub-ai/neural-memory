"""Tests for neural_memory.hooks — Claude Code hook entry points."""

import os
import sys
import pytest
from io import StringIO

from neural_memory.config import NeuralConfig, save_config
from neural_memory.models import IndexMode, NodeType, NeuralNode
from neural_memory.indexer import full_index
from neural_memory.storage import Storage
from neural_memory.hooks import prompt_context, session_end


@pytest.fixture
def indexed_project(tmp_path):
    """A tmp_path that has been fully indexed."""
    (tmp_path / "sample.py").write_text(
        'def greet(name: str) -> str:\n    """Return a greeting."""\n    return f"Hello, {name}!"\n'
    )
    cfg = NeuralConfig(project_root=str(tmp_path), index_mode=IndexMode.AST_ONLY)
    save_config(cfg)
    full_index(config=cfg)
    return tmp_path


def test_prompt_context_emits_to_stdout(indexed_project, monkeypatch, capsys):
    monkeypatch.setenv("CLAUDE_PROJECT_ROOT", str(indexed_project))
    monkeypatch.setenv("CLAUDE_USER_PROMPT", "greet function")
    prompt_context()
    captured = capsys.readouterr()
    assert "neural-memory" in captured.out.lower()


def test_prompt_context_uninitialized_emits_hint(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CLAUDE_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("CLAUDE_USER_PROMPT", "some query")
    prompt_context()
    captured = capsys.readouterr()
    # Should emit a hint about indexing, not crash
    assert "neural-index" in captured.out or "not indexed" in captured.out or captured.out == ""


def test_prompt_context_never_crashes(monkeypatch, capsys):
    monkeypatch.setenv("CLAUDE_PROJECT_ROOT", "/totally/nonexistent/path/xyz")
    monkeypatch.setenv("CLAUDE_USER_PROMPT", "anything")
    # Must not raise
    prompt_context()


def test_session_end_archives_completed_tasks(indexed_project, monkeypatch):
    monkeypatch.setenv("CLAUDE_PROJECT_ROOT", str(indexed_project))

    task = NeuralNode(
        id="task_done_001",
        name="FinishedTask",
        node_type=NodeType.TASK,
        file_path="src/foo.py",
        line_start=1,
        line_end=1,
        category="tasks",
        task_status="done",
        importance=0.5,
    )
    with Storage(str(indexed_project)) as storage:
        storage.upsert_node(task)

    session_end()

    with Storage(str(indexed_project)) as storage:
        node = storage.get_node("task_done_001")
    assert node is not None
    assert node.archived is True
