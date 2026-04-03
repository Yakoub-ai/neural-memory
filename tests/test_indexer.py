"""Tests for neural_memory.indexer — full_index, incremental_update, file discovery."""

import pytest
from pathlib import Path

from neural_memory.config import NeuralConfig
from neural_memory.indexer import (
    _discover_files,
    _file_hash,
    full_index,
    incremental_update,
)
from neural_memory.models import IndexMode, NodeType
from neural_memory.storage import Storage


# ── _discover_files ─────────────────────────────────────────────────────────────

def test_discover_finds_py_files(tmp_path):
    (tmp_path / "a.py").write_text("x = 1")
    (tmp_path / "b.py").write_text("y = 2")
    cfg = NeuralConfig(project_root=str(tmp_path), include_patterns=["**/*.py"])
    files = _discover_files(cfg)
    names = [Path(f).name for f in files]
    assert "a.py" in names
    assert "b.py" in names


def test_discover_excludes_pycache(tmp_path):
    (tmp_path / "good.py").write_text("x = 1")
    pycache = tmp_path / "__pycache__"
    pycache.mkdir()
    (pycache / "cached.py").write_text("x = 1")
    cfg = NeuralConfig(
        project_root=str(tmp_path),
        include_patterns=["**/*.py"],
        exclude_patterns=["**/__pycache__/**"],
    )
    files = _discover_files(cfg)
    names = [Path(f).name for f in files]
    assert "good.py" in names
    assert "cached.py" not in names


def test_discover_respects_custom_exclude(tmp_path):
    (tmp_path / "include_me.py").write_text("x = 1")
    skip = tmp_path / "skip_dir"
    skip.mkdir()
    (skip / "skip_me.py").write_text("x = 1")
    cfg = NeuralConfig(
        project_root=str(tmp_path),
        include_patterns=["**/*.py"],
        exclude_patterns=["**/skip_dir/**"],
    )
    files = _discover_files(cfg)
    names = [Path(f).name for f in files]
    assert "include_me.py" in names
    assert "skip_me.py" not in names


def test_discover_returns_sorted_unique(tmp_path):
    (tmp_path / "z.py").write_text("")
    (tmp_path / "a.py").write_text("")
    cfg = NeuralConfig(project_root=str(tmp_path), include_patterns=["**/*.py"])
    files = _discover_files(cfg)
    assert files == sorted(set(files))


# ── _file_hash ─────────────────────────────────────────────────────────────────

def test_file_hash_consistent(tmp_path):
    (tmp_path / "f.py").write_text("x = 1")
    h1 = _file_hash("f.py", str(tmp_path))
    h2 = _file_hash("f.py", str(tmp_path))
    assert h1 == h2
    assert len(h1) == 16


def test_file_hash_changes_on_content_change(tmp_path):
    f = tmp_path / "f.py"
    f.write_text("x = 1")
    h1 = _file_hash("f.py", str(tmp_path))
    f.write_text("x = 2")
    h2 = _file_hash("f.py", str(tmp_path))
    assert h1 != h2


# ── full_index ─────────────────────────────────────────────────────────────────

def test_full_index_creates_nodes(tmp_path):
    (tmp_path / "mod.py").write_text(
        "def foo():\n    pass\n\ndef bar():\n    foo()\n"
    )
    cfg = NeuralConfig(project_root=str(tmp_path), index_mode=IndexMode.AST_ONLY)
    stats = full_index(config=cfg)

    assert stats["files_processed"] == 1
    assert stats["nodes_created"] >= 3  # MODULE + FUNCTION + FUNCTION
    assert stats["errors"] == []


def test_full_index_creates_edges(tmp_path):
    (tmp_path / "mod.py").write_text(
        "def foo():\n    bar()\n\ndef bar():\n    pass\n"
    )
    cfg = NeuralConfig(project_root=str(tmp_path), index_mode=IndexMode.AST_ONLY)
    stats = full_index(config=cfg)
    assert stats["edges_created"] >= 1


def test_full_index_computes_importance(tmp_path):
    (tmp_path / "mod.py").write_text(
        "def foo():\n    pass\n\nclass MyClass:\n    def method(self): pass\n"
    )
    cfg = NeuralConfig(project_root=str(tmp_path), index_mode=IndexMode.AST_ONLY)
    full_index(config=cfg)

    with Storage(str(tmp_path)) as s:
        all_ids = s.get_all_node_ids()
        importances = [s.get_node(nid).importance for nid in all_ids]
    assert any(i > 0.0 for i in importances)


def test_full_index_saves_index_state(tmp_path):
    (tmp_path / "mod.py").write_text("x = 1\n")
    cfg = NeuralConfig(project_root=str(tmp_path), index_mode=IndexMode.AST_ONLY)
    full_index(config=cfg)

    with Storage(str(tmp_path)) as s:
        state = s.get_index_state()
    assert state.last_full_index is not None
    assert state.total_files >= 1


def test_full_index_redacts_secrets(tmp_path):
    src = 'def connect():\n    password = "supersecretpassword123"\n    return password\n'
    (tmp_path / "db.py").write_text(src)
    cfg = NeuralConfig(project_root=str(tmp_path), index_mode=IndexMode.AST_ONLY)
    full_index(config=cfg)

    with Storage(str(tmp_path)) as s:
        results = s.search_nodes("connect")
    assert results, "Expected to find the 'connect' function"
    connect_node = results[0]
    # The raw_code should have been redacted
    assert "supersecretpassword123" not in connect_node.raw_code


def test_full_index_returns_stats_dict(tmp_path):
    (tmp_path / "mod.py").write_text("x = 1\n")
    cfg = NeuralConfig(project_root=str(tmp_path), index_mode=IndexMode.AST_ONLY)
    stats = full_index(config=cfg)

    assert "files_processed" in stats
    assert "nodes_created" in stats
    assert "edges_created" in stats
    assert "errors" in stats


# ── incremental_update ─────────────────────────────────────────────────────────

def test_incremental_detects_new_files(tmp_path):
    cfg = NeuralConfig(project_root=str(tmp_path), index_mode=IndexMode.AST_ONLY)
    # Initial index with one file
    (tmp_path / "a.py").write_text("def foo(): pass\n")
    full_index(config=cfg)

    # Add a new file
    (tmp_path / "b.py").write_text("def bar(): pass\n")
    stats = incremental_update(config=cfg)
    assert stats["files_added"] >= 1

    with Storage(str(tmp_path)) as s:
        results = s.search_nodes("bar")
    assert any(n.name == "bar" for n in results)


def test_incremental_detects_removed_files(tmp_path):
    cfg = NeuralConfig(project_root=str(tmp_path), index_mode=IndexMode.AST_ONLY)
    f = tmp_path / "removable.py"
    f.write_text("def remove_me(): pass\n")
    full_index(config=cfg)

    # Remove the file
    f.unlink()
    stats = incremental_update(config=cfg)
    assert stats["files_removed"] >= 1

    with Storage(str(tmp_path)) as s:
        results = s.search_nodes("remove_me")
    assert results == []


def test_incremental_detects_changed_files(tmp_path):
    cfg = NeuralConfig(project_root=str(tmp_path), index_mode=IndexMode.AST_ONLY)
    f = tmp_path / "change.py"
    f.write_text("def old_func(): pass\n")
    full_index(config=cfg)

    # Change the file
    f.write_text("def new_func(): pass\n")
    stats = incremental_update(config=cfg)
    assert stats["files_updated"] >= 1

    with Storage(str(tmp_path)) as s:
        new_results = s.search_nodes("new_func")
        old_results = s.search_nodes("old_func")
    assert any(n.name == "new_func" for n in new_results)
    assert not any(n.name == "old_func" for n in old_results)
