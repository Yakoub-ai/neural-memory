---
name: neural-index
description: Build the complete neural memory knowledge graph for this codebase by parsing AST, resolving relationships, and scoring importance.
---

# Neural Memory — Full Index

Build the complete neural memory knowledge graph for this codebase.

## What this does
1. Discovers all Python files (respecting exclude patterns)
2. Parses AST to extract functions, classes, methods, modules
3. Builds a directed graph of call relationships, imports, and inheritance
4. Redacts sensitive content (secrets, API keys, connection strings)
5. Computes importance scores for each node
6. Optionally generates AI-powered summaries for high-importance nodes

## Usage
Run the `neural_index` tool. Options:
- **mode**: `ast_only` (fast, local), `api_only` (rich summaries), or `both` (default)
- **project_root**: defaults to current directory

First run will take longer. Subsequent runs can use `/neural-update` for incremental changes.

After indexing, use `/neural-query` to search and `/neural-inspect` to deep-dive.
