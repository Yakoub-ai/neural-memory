---
name: neural-query
description: Search the neural knowledge graph for functions, classes, modules, or concepts and get layered results with summaries.
---

# Neural Memory — Query

Search the neural knowledge graph for functions, classes, modules, or concepts.

## What this does
Returns layered results:
- **Short summary**: Understand what a node does at a glance
- **Node ID**: Use with `/neural-inspect` to go deeper
- **Location**: File path and line numbers

## Usage
Run the `neural_query` tool with your search term. Examples:
- Search for a function: `neural_query(query="parse_config")`
- Search for a class: `neural_query(query="UserAuth")`
- Search by concept: `neural_query(query="database connection")`

Results are ranked by importance score — the most connected, public-facing code appears first.

Use `/neural-inspect` on any result to see its full context, callers, callees, and source code.
