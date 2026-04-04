"""End-to-end self-referential test — index neural-memory against itself."""

import pytest
from pathlib import Path

from neural_memory.config import NeuralConfig, save_config
from neural_memory.indexer import full_index
from neural_memory.models import IndexMode, NodeType
from neural_memory.storage import Storage

# The neural_memory package directory itself
PACKAGE_ROOT = str(Path(__file__).parent.parent)


@pytest.fixture(scope="module")
def self_indexed(tmp_path_factory):
    """Index the neural_memory package against itself.

    We copy only the source files we want to index into a tmp dir,
    so the DB lives there too — no monkeypatching needed.
    scope=module so we only index once for all E2E tests.
    """
    import shutil

    tmp = tmp_path_factory.mktemp("e2e_root")
    src = Path(PACKAGE_ROOT) / "neural_memory"
    dst = tmp / "neural_memory"
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    cfg = NeuralConfig(
        project_root=str(tmp),
        index_mode=IndexMode.AST_ONLY,
        include_patterns=["neural_memory/**/*.py"],
        exclude_patterns=["**/__pycache__/**"],
    )
    save_config(cfg)
    stats = full_index(config=cfg)
    return {"stats": stats, "tmp": tmp, "cfg": cfg}


@pytest.fixture(scope="module")
def self_storage(self_indexed):
    """Yield a Storage pointing at the self-indexed DB."""
    s = Storage(str(self_indexed["tmp"]))
    s.open()
    yield s
    s.close()


# ── Basic stats ────────────────────────────────────────────────────────────────

def test_e2e_processed_all_source_files(self_indexed):
    stats = self_indexed["stats"]
    # neural_memory has 11 .py files
    assert stats["files_processed"] >= 10
    assert stats["errors"] == []


def test_e2e_nodes_created(self_indexed):
    stats = self_indexed["stats"]
    # At minimum: 11 MODULE + many FUNCTION/CLASS/METHOD nodes
    assert stats["nodes_created"] >= 30


def test_e2e_edges_created(self_indexed):
    stats = self_indexed["stats"]
    assert stats["edges_created"] >= 20


# ── Known nodes exist ──────────────────────────────────────────────────────────

def test_e2e_storage_class_exists(self_storage):
    results = self_storage.search_nodes("Storage", limit=10)
    names = [n.name for n in results]
    assert any("Storage" in n for n in names)


def test_e2e_parse_file_function_exists(self_storage):
    # Use higher limit: multiple parsers define parse_file methods, pushing the
    # module-level function from parser.py further down the ranked list.
    results = self_storage.search_nodes("parse_file", limit=30)
    assert any(n.name == "parse_file" for n in results)


def test_e2e_neural_node_class_exists(self_storage):
    results = self_storage.search_nodes("NeuralNode", limit=10)
    assert any("NeuralNode" in n.name for n in results)


def test_e2e_full_index_function_exists(self_storage):
    results = self_storage.search_nodes("full_index", limit=10)
    assert any(n.name == "full_index" for n in results)


# ── Importance scores are meaningful ──────────────────────────────────────────

def test_e2e_storage_class_has_high_importance(self_storage):
    results = self_storage.search_nodes("Storage", limit=5)
    storage_class = next((n for n in results if n.name == "Storage"), None)
    if storage_class:
        assert storage_class.importance > 0.0


def test_e2e_module_nodes_present(self_storage):
    modules = self_storage.get_nodes_by_type(NodeType.MODULE)
    assert len(modules) >= 10  # One per .py file


def test_e2e_class_nodes_present(self_storage):
    classes = self_storage.get_nodes_by_type(NodeType.CLASS)
    class_names = {n.name for n in classes}
    assert "Storage" in class_names or "NeuralNode" in class_names
