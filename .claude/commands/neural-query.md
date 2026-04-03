# Neural Memory — Query

Search the neural knowledge graph for functions, classes, modules, or concepts.

## What this does
Returns layered results:
- **Short summary**: Understand what a node does at a glance
- **Node ID**: Use with `/neural-inspect` to go deeper
- **Location**: File path and line numbers

## How to call

**Via MCP tool** (neural-memory configured as MCP server in Claude Code):
```json
Tool: neural_query
{ "query": "your search term", "limit": 10 }
```

**Via Python** (working directly in the project):
```python
import asyncio
from neural_memory.server import neural_query, QueryInput

asyncio.run(neural_query(QueryInput(query="your search term")))
```

## Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | str | required | Function name, class name, or concept keyword |
| `project_root` | str | `"."` | Project root directory |
| `limit` | int | `10` | Max results (1–50) |

Results are ranked by importance score — the most connected, public-facing code appears first.

Use `/neural-inspect` on any result's `node_id` to see full context, callers, callees, and source code.
