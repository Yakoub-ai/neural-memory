# Neural Memory — Full Index

Build the complete neural memory knowledge graph for this codebase.

## What this does
1. Discovers all Python files (respecting exclude patterns)
2. Parses AST to extract functions, classes, methods, modules
3. Builds a directed graph of call relationships, imports, and inheritance
4. Redacts sensitive content (secrets, API keys, connection strings)
5. Computes importance scores for each node
6. Optionally generates AI-powered summaries for high-importance nodes

## How to call

**Via MCP tool** (neural-memory configured as MCP server in Claude Code):
```json
Tool: neural_index
{ "mode": "both" }
```

**Via Python** (working directly in the project):
```python
import asyncio
from neural_memory.server import neural_index, IndexInput

asyncio.run(neural_index(IndexInput(mode="both")))
```

## Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project_root` | str | `"."` | Project root directory |
| `mode` | str | config default | `"ast_only"` (fast/local), `"api_only"` (AI summaries), `"both"` (default) |

First run takes longer. Subsequent runs can use `/neural-update` for incremental changes.

After indexing, use `/neural-query` to search and `/neural-inspect` to deep-dive.
