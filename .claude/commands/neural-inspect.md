# Neural Memory — Deep Inspect

Deep-dive into a specific code element — see its full context in the knowledge graph.

## What you get
- **Full summary**: Detailed explanation of purpose, logic, and interface
- **Parent**: Which module or class contains this
- **Callers**: Who calls this function (upstream)
- **Callees**: What this function calls (downstream)
- **Siblings**: Other functions/methods at the same level
- **Children**: Contained elements (methods in a class, etc.)
- **Call chains**: Trace execution paths up and down the graph
- **Source code**: The actual implementation (optional)

## How to call

**Via MCP tool** (neural-memory configured as MCP server in Claude Code):
```json
Tool: neural_inspect
{ "node_id": "module::ClassName.method_name" }
```
or by name:
```json
{ "node_name": "method_name", "show_code": true, "trace_calls": true }
```

**Via Python** (working directly in the project):
```python
import asyncio
from neural_memory.server import neural_inspect, InspectInput

# By node_id (most precise — get from neural_query results)
asyncio.run(neural_inspect(InspectInput(node_id="module::ClassName.method_name")))

# By name (fuzzy match)
asyncio.run(neural_inspect(InspectInput(node_name="method_name", show_code=True, trace_calls=True)))
```

## Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `node_id` | str | None | Exact node ID from query results (most precise) |
| `node_name` | str | None | Name to fuzzy-search (use if you don't have the ID) |
| `project_root` | str | `"."` | Project root directory |
| `show_code` | bool | `false` | Include raw source code in output |
| `trace_calls` | bool | `false` | Show full upstream/downstream call chains |

Provide either `node_id` or `node_name` — `node_id` is preferred when available.
