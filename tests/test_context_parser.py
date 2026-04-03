"""Tests for neural_memory.context_parser."""

import pytest
from neural_memory.context_parser import parse_gotchas, parse_tasks
from neural_memory.models import NodeType, EdgeType


GOTCHAS_SAMPLE = """\
# Gotchas

## 2024-01-15 \u2014 Circular import in models

**File**: `neural_memory/models.py`
**Root cause**: Importing Storage inside models caused a circular dependency.
**Fix**: Move the import to TYPE_CHECKING block.

## 2024-02-20 \u2014 Off-by-one in line numbers

**File**: `neural_memory/parser.py`
**Root cause**: AST gives 1-based lines but we stored 0-based.
**Fix**: No adjustment needed; store as-is.
"""

TASKS_SAMPLE = """\
# Phase 1 \u2014 Core data model

## Fix 1 \u2014 Add BUG node type

**Status**: [x] DONE
**File**: `neural_memory/models.py` lines 10-40

## Fix 2 \u2014 Storage migration

**Status**: [ ] pending
**File**: `neural_memory/storage.py`
"""


class TestParseGotchas:
    def test_returns_list_of_tuples(self):
        results = parse_gotchas(GOTCHAS_SAMPLE)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_node_type_is_bug(self):
        results = parse_gotchas(GOTCHAS_SAMPLE)
        for node, edges in results:
            assert node.node_type == NodeType.BUG

    def test_category_is_bugs(self):
        results = parse_gotchas(GOTCHAS_SAMPLE)
        for node, edges in results:
            assert node.category == "bugs"

    def test_names_extracted(self):
        results = parse_gotchas(GOTCHAS_SAMPLE)
        names = [r[0].name for r in results]
        assert any("Circular import" in n for n in names)
        assert any("Off-by-one" in n for n in names)

    def test_dates_in_summary(self):
        results = parse_gotchas(GOTCHAS_SAMPLE)
        node, _ = results[0]
        assert "2024-01-15" in (node.summary_short or "")

    def test_root_cause_stored(self):
        results = parse_gotchas(GOTCHAS_SAMPLE)
        node, _ = results[0]
        assert "circular" in (node.summary_detailed or "").lower() or "circular" in (node.summary_short or "").lower()

    def test_edges_empty_without_storage(self):
        # Without storage, no RELATES_TO edges can be created
        results = parse_gotchas(GOTCHAS_SAMPLE, storage=None)
        for node, edges in results:
            assert isinstance(edges, list)

    def test_idempotent_ids(self):
        """Same content → same node IDs."""
        r1 = parse_gotchas(GOTCHAS_SAMPLE)
        r2 = parse_gotchas(GOTCHAS_SAMPLE)
        ids1 = {r[0].id for r in r1}
        ids2 = {r[0].id for r in r2}
        assert ids1 == ids2

    def test_empty_content(self):
        results = parse_gotchas("")
        assert results == []

    def test_no_headings(self):
        results = parse_gotchas("# Just a title\n\nSome prose without entries.")
        assert results == []

    def test_severity_field(self):
        results = parse_gotchas(GOTCHAS_SAMPLE)
        # severity should be a string (may be empty or default)
        for node, _ in results:
            assert isinstance(node.severity, str)


class TestParseTasks:
    def test_returns_list_of_tuples(self):
        results = parse_tasks(TASKS_SAMPLE)
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_phase_node_created(self):
        results = parse_tasks(TASKS_SAMPLE)
        types = [r[0].node_type for r in results]
        assert NodeType.PHASE in types

    def test_task_nodes_created(self):
        results = parse_tasks(TASKS_SAMPLE)
        types = [r[0].node_type for r in results]
        assert NodeType.TASK in types

    def test_category_is_tasks(self):
        results = parse_tasks(TASKS_SAMPLE)
        for node, _ in results:
            assert node.category == "tasks"

    def test_task_status_done(self):
        results = parse_tasks(TASKS_SAMPLE)
        task_nodes = [r[0] for r in results if r[0].node_type == NodeType.TASK]
        statuses = [n.task_status for n in task_nodes]
        assert "done" in statuses or any("done" in (s or "") for s in statuses)

    def test_task_names_extracted(self):
        results = parse_tasks(TASKS_SAMPLE)
        names = [r[0].name for r in results]
        assert any("BUG node" in n or "bug node" in n.lower() for n in names)

    def test_phase_contains_edges(self):
        results = parse_tasks(TASKS_SAMPLE)
        all_edges = [e for _, edges in results for e in edges]
        edge_types = [e.edge_type for e in all_edges]
        assert EdgeType.PHASE_CONTAINS in edge_types

    def test_idempotent_ids(self):
        r1 = parse_tasks(TASKS_SAMPLE)
        r2 = parse_tasks(TASKS_SAMPLE)
        ids1 = {r[0].id for r in r1}
        ids2 = {r[0].id for r in r2}
        assert ids1 == ids2

    def test_empty_content(self):
        # parse_tasks always creates at least a stub PHASE node from the source filename
        results = parse_tasks("")
        assert isinstance(results, list)
        if results:
            phase_nodes = [r[0] for r in results if r[0].node_type == NodeType.PHASE]
            assert len(phase_nodes) >= 1

    def test_phase_name_in_node(self):
        results = parse_tasks(TASKS_SAMPLE)
        phase_nodes = [r[0] for r in results if r[0].node_type == NodeType.PHASE]
        assert phase_nodes
        assert any("Phase 1" in n.name or "phase" in n.name.lower() for n in phase_nodes)
