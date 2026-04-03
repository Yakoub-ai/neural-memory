---
name: neural-status
description: Check the health and freshness of the neural memory index — initialization state, staleness, and graph statistics.
---

# Neural Memory — Status Check

Check the health and freshness of the neural memory index.

## What this checks
- Whether neural memory has been initialized
- How many commits behind the index is
- Which files have changed since last index
- Total nodes, edges, and files in the graph
- Current index mode and configuration

## How to call

**Via MCP tool** (neural-memory configured as MCP server in Claude Code):
```json
Tool: neural_status
{}
```

**Via Python** (working directly in the project):
```python
import asyncio
from neural_memory.server import neural_status, StatusInput

asyncio.run(neural_status(StatusInput()))
```

`StatusInput` has no required fields. Pass `project_root="."` to be explicit.

> **Windows**: run `python -X utf8` (or set `PYTHONUTF8=1`) to avoid cp1252 encoding errors from Unicode output. Applies to all neural commands.

## Agent behavior
The neural agent runs this check automatically and will suggest:
- `/neural-index` if no index exists
- `/neural-update` if the index is stale (default: 5+ commits behind)
