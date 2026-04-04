# Neural Memory Plugin

Neural Memory is a knowledge graph that maps your codebase into layered, navigable nodes — so you can understand any function without reading the entire file.

## Quick Start

1. **First time?** Run `/neural-index` to build the knowledge graph
2. **Search**: `/neural-query` to find functions, classes, or concepts
3. **Deep dive**: `/neural-inspect` to see a node's full context, callers, and callees
4. **Stay fresh**: `/neural-update` after code changes
5. **Check health**: `/neural-status` to see index freshness

## How It Works

The plugin parses your codebase into a three-layer directed graph:

**Codebase layer** — AST-parsed nodes, LSP-enriched:
- Nodes: modules, classes, functions, methods, project/directory overviews
- Edges: calls, imports, inheritance, containment

**Bugs layer** — auto-imported from `.claude/context-log-gotchas.md` or via MCP tool:
- Nodes: bug entries with severity, status, root cause, fix description
- Edges: `RELATES_TO` code nodes by file path

**Tasks layer** — auto-imported from `.claude/context-log-tasks-XX.md` or via MCP tool:
- Nodes: phases, tasks with status and priority
- Edges: `PHASE_CONTAINS` task, `RELATES_TO` code nodes

**Embeddings**: 138-dim composite (100-dim TF-IDF+SVD content + 38-dim structural graph features)
**Search**: three-phase branch search — seed by cosine similarity → graph expansion → weighted rank
**Security**: Secrets, API keys, and sensitive values are automatically redacted
**LSP**: Pyright/pylsp enrichment for high-importance nodes (type signatures, diagnostics)

## MCP Server

This plugin runs as an MCP server. Add to your Claude Code config:

```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "python",
      "args": ["-m", "neural_memory.server"],
      "cwd": "/path/to/neural-memory"
    }
  }
}
```

## Agent Hook

On each invocation, check neural memory staleness by running the `neural_status` tool.
If the index is stale or uninitialized, inform the user and suggest the appropriate action.

## Available Commands

| Command | Tool | Purpose |
|---------|------|---------|
| `/neural-index` | `neural_index` | Full codebase index (all three layers) |
| `/neural-update` | `neural_update` | Incremental git-based update |
| `/neural-query` | `neural_query` | Semantic search (code + bugs + tasks) |
| `/neural-inspect` | `neural_inspect` | Deep-dive into a node |
| `/neural-status` | `neural_status` | Health and staleness check |
| `/neural-config` | `neural_config` | View/modify settings |
| — | `neural_visualize_dashboard` | Interactive D3 dashboard (3 views, filters) |
| — | `neural_add_bug` | Manually add a bug node |
| — | `neural_add_task` | Manually add a task node |
| `/neural-visualize` | `neural_serve` | Start interactive dashboard server |
| `/neural-stop` | `neural_stop_serve` | Stop dashboard server |
| — | `neural_index_db` | Index live database schema |
| — | `neural_fetch_docs` | Fetch external package documentation |
