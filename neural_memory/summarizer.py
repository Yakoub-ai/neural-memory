"""Summary generation for neural memory nodes.

Supports two modes:
- Heuristic: AST-based, fast, no API cost, local-only
- API: Claude API for rich, contextual summaries
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
from typing import Optional

from .models import NeuralNode, SummaryMode, NodeType, IndexMode
from .config import NeuralConfig


def _heuristic_detailed_summary(node: NeuralNode) -> str:
    """Generate a detailed heuristic summary from available node data."""
    parts = []

    if node.node_type == NodeType.FUNCTION or node.node_type == NodeType.METHOD:
        parts.append(f"**{node.signature or node.name}**")
        if node.docstring:
            parts.append(f"Documentation: {node.docstring[:300]}")
        parts.append(f"Location: {node.file_path}:{node.line_start}-{node.line_end}")
        parts.append(f"Complexity: {node.complexity}")
        if node.decorators:
            parts.append(f"Decorators: {', '.join(node.decorators)}")
        if not node.is_public:
            parts.append("Visibility: private/internal")
        body_lines = node.line_end - node.line_start + 1
        parts.append(f"Size: {body_lines} lines")

    elif node.node_type == NodeType.CLASS:
        parts.append(f"**class {node.name}**")
        if node.docstring:
            parts.append(f"Documentation: {node.docstring[:300]}")
        parts.append(f"Location: {node.file_path}:{node.line_start}-{node.line_end}")
        if node.decorators:
            parts.append(f"Decorators: {', '.join(node.decorators)}")

    elif node.node_type == NodeType.MODULE:
        parts.append(f"**Module: {node.name}**")
        if node.docstring:
            parts.append(f"Documentation: {node.docstring[:300]}")
        parts.append(f"File: {node.file_path}")

    else:
        parts.append(f"**{node.node_type.value}: {node.name}**")
        parts.append(f"Location: {node.file_path}:{node.line_start}-{node.line_end}")

    return "\n".join(parts)


def _build_api_prompt(node: NeuralNode, context_nodes: list[NeuralNode] = None) -> str:
    """Build the prompt for Claude API summary generation."""
    lang = node.language or "python"
    prompt = f"""Analyze this code element and provide two summaries:

1. SHORT SUMMARY (max 200 chars): A concise one-liner describing what this does.
2. DETAILED SUMMARY (max 1000 chars): A thorough explanation covering:
   - Purpose and responsibility
   - Key logic / algorithm used
   - Input/output contract
   - Side effects or important caveats
   - How it fits in the broader module

Respond in JSON format:
{{"short": "...", "detailed": "..."}}

Code element ({node.node_type.value}): {node.name}
Language: {lang}
File: {node.file_path}
Signature: {node.signature}
Docstring: {node.docstring}

Source code:
```{lang}
{node.raw_code[:3000]}
```
"""

    if context_nodes:
        prompt += "\n\nRelated code context:\n"
        for ctx in context_nodes[:3]:
            prompt += f"- {ctx.name} ({ctx.node_type.value}): {ctx.summary_short}\n"

    return prompt


def generate_api_summary(node: NeuralNode, context_nodes: list[NeuralNode] = None) -> tuple[str, str]:
    """Generate summary via Claude API using subprocess to call Claude Code.

    Returns (short_summary, detailed_summary).
    Falls back to heuristic if API fails.
    """
    prompt = _build_api_prompt(node, context_nodes)

    try:
        # Use Claude CLI to generate summary
        result = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "json"],
            capture_output=True, text=True, timeout=30
        )

        if result.returncode != 0:
            return node.summary_short, _heuristic_detailed_summary(node)

        # Parse Claude's response
        response = json.loads(result.stdout)
        text = response.get("result", result.stdout)

        # Try to extract JSON from response
        # Handle both raw JSON and text-wrapped JSON
        if isinstance(text, str):
            # Find JSON in response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(text[start:end])
                return (
                    parsed.get("short", node.summary_short)[:200],
                    parsed.get("detailed", _heuristic_detailed_summary(node))[:1000]
                )

    except subprocess.TimeoutExpired:
        warnings.warn(f"Claude CLI timed out summarizing {node.name}")
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        warnings.warn(f"Claude CLI returned unparseable output for {node.name}: {e}")
    except FileNotFoundError:
        warnings.warn("Claude CLI not found — falling back to heuristic summaries")
    except Exception as e:
        warnings.warn(f"Unexpected error summarizing {node.name}: {e}")

    # Fallback to heuristic
    return node.summary_short, _heuristic_detailed_summary(node)


def summarize_node(
    node: NeuralNode,
    mode: IndexMode,
    context_nodes: list[NeuralNode] = None
) -> NeuralNode:
    """Generate summaries for a node based on the configured mode."""

    if mode == IndexMode.AST_ONLY:
        node.summary_detailed = _heuristic_detailed_summary(node)
        node.summary_mode = SummaryMode.HEURISTIC

    elif mode == IndexMode.API_ONLY:
        short, detailed = generate_api_summary(node, context_nodes)
        node.summary_short = short
        node.summary_detailed = detailed
        node.summary_mode = SummaryMode.API

    elif mode == IndexMode.BOTH:
        # Start with heuristic, then enrich with API
        node.summary_detailed = _heuristic_detailed_summary(node)
        short, detailed = generate_api_summary(node, context_nodes)
        node.summary_short = short
        node.summary_detailed = detailed
        node.summary_mode = SummaryMode.BOTH

    return node
