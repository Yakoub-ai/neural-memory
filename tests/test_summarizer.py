"""Tests for neural_memory.summarizer."""

import warnings

import pytest

from neural_memory.models import IndexMode, NeuralNode, NodeType, SummaryMode
from neural_memory.summarizer import summarize_node


def _node() -> NeuralNode:
    return NeuralNode(
        id="x", name="foo", node_type=NodeType.FUNCTION,
        file_path="f.py", line_start=1, line_end=5
    )


def test_summarize_node_ast_only():
    node = _node()
    summarize_node(node, IndexMode.AST_ONLY, context_nodes=[])
    assert node.summary_mode == SummaryMode.HEURISTIC
    assert node.summary_detailed  # non-empty


def test_summarize_falls_back_gracefully():
    """summarize_node must never raise, regardless of CLI availability."""
    node = _node()
    # capture any warnings (emitted when CLI is absent or fails)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        summarize_node(node, IndexMode.API_ONLY, context_nodes=[])
    # No raise; summary_mode is some valid value
    assert node.summary_mode in (SummaryMode.HEURISTIC, SummaryMode.API, SummaryMode.BOTH)
