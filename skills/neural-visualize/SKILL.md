---
name: neural-visualize
description: Generate and open the interactive knowledge graph dashboard in your browser — hierarchy treemap, semantic radial tree, and force-directed graph.
---

# Neural Memory — Start Dashboard

Generate the interactive knowledge graph dashboard and open it in your browser.

## What this does
1. Regenerates `dashboard.html` from the current index
2. Starts a local HTTP server at `http://localhost:7891`
3. Opens the dashboard automatically in your default browser

## How to call

**Via MCP tool** (neural-memory configured as MCP server in Claude Code):
```json
Tool: neural_serve
{ "port": 7891, "open_browser": true }
```

**Via Python** (working directly in the project):
```python
import asyncio
from neural_memory.server import neural_serve, ServeInput

asyncio.run(neural_serve(ServeInput(port=7891, open_browser=True)))
```

> **Windows**: run `python -X utf8` (or set `PYTHONUTF8=1`) to avoid cp1252 encoding errors from Unicode output.

> **Same-process requirement**: `neural_serve` starts the HTTP server in a background thread within the running Python process. Stopping it requires calling `neural_stop_serve(StopServeInput())` in the **same** process before exit — not from a separate script invocation.

**Standalone server** (recommended for direct CLI use):
```
neural-memory-viz
neural-memory-viz --port 8080
neural-memory-viz --no-browser
```

## Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project_root` | str | `"."` | Project root directory |
| `port` | int | `7891` | HTTP port (change if 7891 is taken) |
| `open_browser` | bool | `true` | Auto-open dashboard in browser |
| `regenerate` | bool | `true` | Regenerate HTML from current index |

## Dashboard views
- **Hierarchy** — Treemap: project → module → class → function (sized by importance)
- **Semantic** — Radial tree with vector-space nearest-neighbor hints on hover
- **Graph** — Force-directed: all nodes + edges, hover-highlight, 2-hop focus, drag/zoom

## Stop the server
Run `/neural-stop` or call `neural_stop_serve(StopServeInput())` **in the same process**.
