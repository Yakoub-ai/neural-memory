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

## Usage

Call the `neural_serve` tool:
```
neural_serve(project_root=".", port=7891, open_browser=True)
```

Optional parameters:
- `port` — change the port if 7891 is taken
- `open_browser=False` — serve without auto-opening
- `regenerate=False` — skip regenerating HTML (use cached version)

## Dashboard views
- **Hierarchy** — Treemap: project → module → class → function (sized by importance)
- **Semantic** — Radial tree with vector-space nearest-neighbor hints on hover
- **Graph** — Force-directed: all nodes + edges, hover-highlight, 2-hop focus, drag/zoom

## Stop the server
Run `neural_stop_serve()` or use `/neural-stop`.

## Direct CLI (outside Claude Code)
```
neural-memory-viz
neural-memory-viz --port 8080
neural-memory-viz --no-browser
```
