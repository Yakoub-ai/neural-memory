# Neural Memory Plugin

Neural Memory is a knowledge graph that maps your codebase into layered, navigable nodes — so you can understand any function without reading the entire file.

## Quick Start

1. **First time?** Run `/neural-index` to build the knowledge graph
2. **Search**: `/neural-query` to find functions, classes, or concepts
3. **Deep dive**: `/neural-inspect` to see a node's full context, callers, and callees
4. **Stay fresh**: `/neural-update` after code changes
5. **Check health**: `/neural-status` to see index freshness

## How It Works

The plugin parses your Python codebase into a directed graph:
- **Nodes**: Functions, classes, methods, modules (with layered summaries)
- **Edges**: Calls, imports, inheritance, containment
- **Summaries**: Short (1-liner) and detailed — understand code without reading it
- **Security**: Secrets, API keys, and sensitive values are automatically redacted

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
| `/neural-index` | `neural_index` | Full codebase index |
| `/neural-update` | `neural_update` | Incremental git-based update |
| `/neural-query` | `neural_query` | Search the knowledge graph |
| `/neural-inspect` | `neural_inspect` | Deep-dive into a node |
| `/neural-status` | `neural_status` | Health and staleness check |
| `/neural-config` | `neural_config` | View/modify settings |
