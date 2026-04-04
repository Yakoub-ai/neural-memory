# Neural Memory Plugin

Neural Memory is a knowledge graph that maps your codebase into four layered, navigable layers — code, bugs, tasks, and insights — so Claude always has deep context without re-reading files.

## Quick Start

1. **First time?** Run `/neural-index` to build the knowledge graph
2. **Search**: `/neural-query` to find functions, classes, or concepts
3. **Deep dive**: `/neural-inspect` to see a node's full context, callers, and callees
4. **Stay fresh**: `/neural-update` after code changes
5. **Check health**: `/neural-status` to see index freshness
6. **Save knowledge**: `/neural-insight` to generate docs from accumulated insights

## How It Works

The plugin parses your codebase into a four-layer directed graph:

**Codebase layer** — AST-parsed nodes, LSP-enriched:
- Nodes: modules, classes, functions, methods, project/directory overviews
- Edges: calls, imports, inheritance, containment
- Languages: Python, JavaScript/TypeScript, Go, Rust, Ruby, PHP, Java, C/C++

**Bugs layer** — auto-imported from `.claude/context-log-gotchas.md` or via MCP tool:
- Nodes: bug entries with severity, status, root cause, fix description
- Edges: `RELATES_TO` code nodes by file path

**Tasks layer** — auto-imported from `.claude/context-log-tasks-XX.md` or via MCP tool:
- Nodes: phases, tasks with status and priority
- Edges: `PHASE_CONTAINS` task, `RELATES_TO` code nodes
- Task statuses: `pending` → `in_progress` → `testing` → `done` (auto-archived)

**Insights layer** — accumulated technical knowledge across sessions:
- Nodes: design decisions, architecture patterns, performance tradeoffs
- Edges: `RELATES_TO` code nodes
- Synthesized into docs via `neural_generate_docs` / `/neural-insight`

**Embeddings**: 138-dim composite (100-dim TF-IDF+SVD content + 38-dim structural graph features)
**Search**: three-phase branch search — seed by cosine similarity → graph expansion → weighted rank
**Security**: Secrets, API keys, and sensitive values are automatically redacted
**LSP**: Pyright/pylsp enrichment for high-importance nodes (type signatures, diagnostics)

## MCP Server

This plugin runs as an MCP server. The recommended way is via the plugin marketplace (auto-configured), or add manually to `.mcp.json`:

```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "uvx",
      "args": ["--from", "neural-memory-mcp", "neural-memory"]
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
| `/neural-index` | `neural_index` | Full codebase index (all four layers) |
| `/neural-update` | `neural_update` | Incremental git-based update |
| `/neural-query` | `neural_query` | Semantic search (code + bugs + tasks + insights) |
| `/neural-inspect` | `neural_inspect` | Deep-dive into a node |
| `/neural-status` | `neural_status` | Health and staleness check |
| `/neural-config` | `neural_config` | View/modify settings |
| `/neural-visualize` | `neural_serve` | Start interactive dashboard server |
| `/neural-stop` | `neural_stop_serve` | Stop dashboard server |
| `/neural-add-task` | `neural_add_task` | Create a task node manually |
| `/neural-add-bug` | `neural_add_bug` | Create a bug node manually |
| `/neural-tasks` | `neural_list_tasks` | List/filter tasks by status or priority |
| `/neural-context` | `neural_context` | Token-budgeted context snapshot |
| `/neural-insight` | `neural_generate_docs` | Generate technical docs from insight bank |
| — | `neural_add_insight` | Save a technical insight |
| — | `neural_list_insights` | Browse insights by topic |
| — | `neural_save_context` | Save session state to session_context.md |
| — | `neural_update_task` | Update task status or priority |
| — | `neural_visualize_dashboard` | Generate D3 dashboard HTML |
| — | `neural_index_db` | Index live database schema |
| — | `neural_fetch_docs` | Fetch external package documentation |

## Bundled Agents

Three agents ship with the plugin and are pre-wired to the MCP tools:

| Agent | Purpose |
|-------|---------|
| `neural-memory:neural-explorer` | Codebase exploration via `neural_query` / `neural_inspect` — prefer over generic Explore |
| `neural-memory:neural-insight-collector` | Captures technical insights into the bank after implementations |
| `neural-memory:neural-doc-writer` | Synthesizes all insights into structured documentation |

## Hooks (Automatic)

Two hooks run automatically after installation:

**`UserPromptSubmit`** — fires before each message, injects compact context (~500 tokens): index health, project overview, active bugs/tasks, insight bank summary. On first prompt of each session, also loads the previous session's context from `.neural-memory/session_context.md`.

**`Stop`** — fires at session end: saves session state to `session_context.md`, archives completed tasks/bugs, and runs incremental update if the index is slightly stale.
