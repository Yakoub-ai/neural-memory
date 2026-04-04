"""Tests for neural_memory.server — MCP tool functions (async, direct invocation)."""

import json
import pytest
import pytest_asyncio

from neural_memory.models import IndexMode, NodeType
from neural_memory.config import NeuralConfig, save_config
from neural_memory.indexer import full_index
from neural_memory.storage import Storage
from neural_memory.server import (
    neural_index,
    neural_query,
    neural_inspect,
    neural_status,
    neural_config,
    neural_context,
    neural_archive,
    IndexInput,
    QueryInput,
    InspectInput,
    StatusInput,
    ConfigInput,
    ContextInput,
    ArchiveInput,
)


@pytest.fixture
def indexed_project(tmp_path):
    """Return a tmp_path that has been fully indexed."""
    (tmp_path / "sample.py").write_text(
        'def greet(name: str) -> str:\n    """Return a greeting."""\n    return f"Hello, {name}!"\n\n'
        'def compute(x: int, y: int = 0) -> int:\n    """Add x and y."""\n    result = greet("world")\n    return x + y\n\n'
        'class Calculator:\n    """A simple calculator."""\n    def add(self, a, b):\n        return compute(a, b)\n'
    )
    cfg = NeuralConfig(project_root=str(tmp_path), index_mode=IndexMode.AST_ONLY)
    save_config(cfg)
    full_index(config=cfg)
    return tmp_path


# ── neural_index ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_neural_index_returns_stats(tmp_path):
    (tmp_path / "mod.py").write_text("def foo():\n    pass\n")
    result = await neural_index(IndexInput(project_root=str(tmp_path), mode="ast_only"))
    assert "Full Index Complete" in result
    assert "Files processed" in result
    assert "Nodes created" in result


@pytest.mark.asyncio
async def test_neural_index_empty_project(tmp_path):
    result = await neural_index(IndexInput(project_root=str(tmp_path), mode="ast_only"))
    assert "Full Index Complete" in result


# ── neural_query ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_neural_query_finds_results(indexed_project):
    result = await neural_query(QueryInput(
        query="greet",
        project_root=str(indexed_project),
    ))
    assert "greet" in result.lower()
    assert "No results" not in result


@pytest.mark.asyncio
async def test_neural_query_no_results(indexed_project):
    result = await neural_query(QueryInput(
        query="xyznonexistent",
        project_root=str(indexed_project),
    ))
    assert "No results" in result


@pytest.mark.asyncio
async def test_neural_query_includes_node_id(indexed_project):
    result = await neural_query(QueryInput(
        query="compute",
        project_root=str(indexed_project),
    ))
    assert "ID:" in result


# ── neural_inspect ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_neural_inspect_by_name(indexed_project):
    result = await neural_inspect(InspectInput(
        node_name="greet",
        project_root=str(indexed_project),
    ))
    assert "greet" in result.lower()


@pytest.mark.asyncio
async def test_neural_inspect_not_found(indexed_project):
    result = await neural_inspect(InspectInput(
        node_name="no_such_function_xyz",
        project_root=str(indexed_project),
    ))
    assert "not found" in result.lower()


@pytest.mark.asyncio
async def test_neural_inspect_show_code(indexed_project):
    result = await neural_inspect(InspectInput(
        node_name="greet",
        project_root=str(indexed_project),
        show_code=True,
    ))
    assert "Source Code" in result
    assert "```" in result


@pytest.mark.asyncio
async def test_neural_inspect_by_id(indexed_project):
    # First query to get a node ID
    query_result = await neural_query(QueryInput(
        query="greet",
        project_root=str(indexed_project),
    ))
    # Extract the ID from the query result (format: "ID: `<id>`")
    import re
    m = re.search(r"ID: `([a-f0-9]+)`", query_result)
    assert m, f"Could not find node ID in: {query_result}"
    node_id = m.group(1)

    result = await neural_inspect(InspectInput(
        node_id=node_id,
        project_root=str(indexed_project),
    ))
    assert "greet" in result.lower()


# ── neural_status ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_neural_status_on_uninitialized_project(tmp_path):
    result = await neural_status(StatusInput(project_root=str(tmp_path)))
    # Should mention it's uninitialized or not indexed
    assert len(result) > 0


@pytest.mark.asyncio
async def test_neural_status_after_index(indexed_project):
    result = await neural_status(StatusInput(project_root=str(indexed_project)))
    assert len(result) > 0


# ── neural_config ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_neural_config_view(tmp_path):
    result = await neural_config(ConfigInput(project_root=str(tmp_path), action="view"))
    data = json.loads(result)
    assert "index_mode" in data
    assert "include_patterns" in data


@pytest.mark.asyncio
async def test_neural_config_set_mode(tmp_path):
    result = await neural_config(ConfigInput(
        project_root=str(tmp_path),
        action="set_mode",
        value="ast_only",
    ))
    assert "ast_only" in result


@pytest.mark.asyncio
async def test_neural_config_add_exclude(tmp_path):
    result = await neural_config(ConfigInput(
        project_root=str(tmp_path),
        action="add_exclude",
        value="**/generated/**",
    ))
    assert "generated" in result


# ── neural_context ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_neural_context_returns_snapshot(indexed_project):
    result = await neural_context(ContextInput(project_root=str(indexed_project)))
    assert "<!-- neural-memory context -->" in result
    assert "<!-- /neural-memory -->" in result


@pytest.mark.asyncio
async def test_neural_context_with_query_hint(indexed_project):
    result = await neural_context(ContextInput(
        project_root=str(indexed_project),
        query_hint="greet",
    ))
    assert "<!-- neural-memory context -->" in result


@pytest.mark.asyncio
async def test_neural_context_uninitialized(tmp_path):
    result = await neural_context(ContextInput(project_root=str(tmp_path)))
    # Should still return wrapped content (uninitialized status), not crash
    assert "neural-memory" in result.lower()


# ── neural_archive ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_neural_archive_and_unarchive(indexed_project):
    # Get a known node ID first
    query_result = await neural_query(QueryInput(
        query="greet",
        project_root=str(indexed_project),
    ))
    import re
    m = re.search(r"ID: `([a-f0-9]+)`", query_result)
    assert m, f"Could not find node ID in: {query_result}"
    node_id = m.group(1)

    # Archive it
    result = await neural_archive(ArchiveInput(
        node_id=node_id,
        action="archive",
        project_root=str(indexed_project),
    ))
    assert "archived" in result.lower()

    # Unarchive it
    result2 = await neural_archive(ArchiveInput(
        node_id=node_id,
        action="unarchive",
        project_root=str(indexed_project),
    ))
    assert "unarchived" in result2.lower()


@pytest.mark.asyncio
async def test_neural_archive_missing_node(indexed_project):
    result = await neural_archive(ArchiveInput(
        node_id="nonexistent_node_id_xyz",
        action="archive",
        project_root=str(indexed_project),
    ))
    assert "not found" in result.lower()


# ── neural_query include_archived ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_neural_query_excludes_archived_by_default(indexed_project):
    # Archive a node then verify it doesn't appear in default query
    query_result = await neural_query(QueryInput(
        query="greet",
        project_root=str(indexed_project),
    ))
    import re
    m = re.search(r"ID: `([a-f0-9]+)`", query_result)
    assert m
    node_id = m.group(1)

    await neural_archive(ArchiveInput(
        node_id=node_id,
        action="archive",
        project_root=str(indexed_project),
    ))

    # Default query should not return the archived node
    result = await neural_query(QueryInput(
        query="greet",
        project_root=str(indexed_project),
    ))
    assert node_id not in result


@pytest.mark.asyncio
async def test_neural_query_includes_archived_when_requested(indexed_project):
    # Archive a node, then query with include_archived=True
    query_result = await neural_query(QueryInput(
        query="greet",
        project_root=str(indexed_project),
    ))
    import re
    m = re.search(r"ID: `([a-f0-9]+)`", query_result)
    assert m
    node_id = m.group(1)

    await neural_archive(ArchiveInput(
        node_id=node_id,
        action="archive",
        project_root=str(indexed_project),
    ))

    result = await neural_query(QueryInput(
        query="greet",
        project_root=str(indexed_project),
        include_archived=True,
    ))
    assert node_id in result
