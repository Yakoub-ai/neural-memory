---
name: neural-stop
description: Stop the running neural memory dashboard server.
---

# Neural Memory — Stop Dashboard

Stop the running neural memory dashboard server.

## How to call

**Via MCP tool** (neural-memory configured as MCP server in Claude Code):
```json
Tool: neural_stop_serve
{}
```

**Via Python** (working directly in the project):
```python
import asyncio
from neural_memory.server import neural_stop_serve, StopServeInput

asyncio.run(neural_stop_serve(StopServeInput()))
```

`StopServeInput` has no required fields — always pass an empty instance.

> **Same-process requirement**: `neural_stop_serve` only stops a server started in the **same** Python process. It cannot stop a server started by a separate script invocation. For standalone serve/stop, use the `neural-memory-viz` CLI instead.

## Notes
- Returns a confirmation with the URL that was stopped, or a message if no server was running
- The server is a lightweight local HTTP server (Python stdlib, no extra dependencies)
- It only serves files from `.neural-memory/` — no external access
- The server is automatically stopped when the Claude Code session ends
