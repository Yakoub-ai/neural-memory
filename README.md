# Neural Memory

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Claude Code Plugin](https://img.shields.io/badge/Claude%20Code-Plugin-green.svg)](https://github.com/Yakoub-ai/neural-memory)

> A persistent knowledge graph plugin for Claude Code that maps your codebase into four layers — code, bugs, tasks, and insights — so Claude always has deep context without burning tokens re-reading files.
![Screenshot of semantic graph.](https://i.imgur.com/M4IPfg4.png)
---

## What It Does

Neural Memory builds and maintains a **four-layer directed knowledge graph** of your project:

| Layer | Nodes | Purpose |
|-------|-------|---------|
| **Codebase** | modules, classes, functions, methods, overviews | Understands structure, calls, imports, inheritance |
| **Bugs** | bug entries with severity and status | Tracks known issues linked to code nodes |
| **Tasks** | phases and tasks with lifecycle | Tracks work items linked to code + each other |
| **Insights** | technical knowledge accumulated over sessions | Builds persistent project documentation |

Every node gets:
- **Layered summaries** — heuristic one-liner from AST, optionally enriched via Claude API
- **138-dim composite embeddings** — 100-dim TF-IDF+SVD content vector + 38-dim structural graph features
- **LSP enrichment** — type signatures and diagnostics from Pyright/pylsp
- **Importance scores** — graph-centrality weighted by node type

Search uses a **three-phase branch algorithm**: seed by cosine similarity → expand through graph edges with decay/pruning → re-rank by combined score. Sensitive values (API keys, tokens, passwords) are automatically redacted before storage.

**Supports 8 languages**: Python, JavaScript/TypeScript, Go, Rust, Ruby, PHP, Java, C/C++.

---

## Quick Start

### Install via Claude Code plugin marketplace (recommended)

```
/plugin marketplace add Yakoub-ai/neural-memory
/plugin install neural-memory@Yakoub-ai
```

The plugin automatically registers:
- **MCP server** — 26 tools accessible to Claude and all bundled agents
- **13 skills** — `/neural-index`, `/neural-query`, `/neural-insight`, etc.
- **3 agents** — `neural-explorer`, `neural-insight-collector`, `neural-doc-writer`
- **2 hooks** — automatic context injection + session persistence

### Build your knowledge graph

```
/neural-memory:neural-index
```

**Done.** Claude now has persistent, token-efficient context about your entire codebase.

> **Manual setup?** See [Installation Options](#installation-options) below.

---

## Installation Options

### Option A — Claude Code Plugin (Recommended)

No pip install required. The MCP server runs via `uvx` directly from PyPI.

```
/plugin marketplace add Yakoub-ai/neural-memory
/plugin install neural-memory@Yakoub-ai
```

Skills are namespaced: `/neural-memory:neural-index`, `/neural-memory:neural-query`, etc.

What gets installed:
- MCP server registered in your `settings.json` (runs via `uvx`, zero config)
- 13 skills copied to `.claude/commands/`
- 3 agents registered for autonomous tasks
- 2 hooks wired to `UserPromptSubmit` and `Stop` events
- `CLAUDE.md` agent hook so Claude always checks index freshness

### Option B — Interactive Setup

Install the package and run the setup wizard:

```bash
# Install
pip install "neural-memory-mcp[all]"

# Run setup wizard (registers MCP, installs skills, hooks, offers RTK)
neural-memory-setup install
```

The wizard walks through each step and confirms before making changes. Run `neural-memory-setup doctor` to diagnose any issues.

### Option C — Manual MCP Setup

**1. Install:**
```bash
pip install neural-memory-mcp          # core only
pip install "neural-memory-mcp[vectors]"  # + semantic search (recommended)
pip install "neural-memory-mcp[all]"      # + dashboard, all features
```

**2. Add to `.mcp.json`:**
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

**3. Add agent hook to `CLAUDE.md`:**
```markdown
## Agent Hook
On each invocation, check neural memory staleness by running the `neural_status` tool.
If the index is stale or uninitialized, inform the user and suggest the appropriate action.
```

### Option D — From source

```bash
git clone https://github.com/Yakoub-ai/neural-memory.git
cd neural-memory
pip install -e ".[all]"
neural-memory-setup install
```

---

## Commands Reference

### Skills (slash commands in Claude Code)

When installed via plugin, skills are namespaced as `/neural-memory:<skill>`. When used standalone, they run as `/neural-<skill>`.

| Skill | MCP Tool | Description |
|-------|----------|-------------|
| `neural-index` | `neural_index` | Full index — AST parse all files, import bugs/tasks, compute embeddings, generate overviews |
| `neural-update` | `neural_update` | Incremental update via git diff — only re-parses changed files |
| `neural-query` | `neural_query` | Semantic search across all four layers |
| `neural-inspect` | `neural_inspect` | Deep-dive into a node: callers, callees, LSP data, full summary |
| `neural-status` | `neural_status` | Index health — nodes/edges, staleness, freshness verdict |
| `neural-config` | `neural_config` | View or modify settings |
| `neural-visualize` | `neural_serve` | Start interactive dashboard at `http://localhost:7891` |
| `neural-stop` | `neural_stop_serve` | Stop the dashboard server |
| `neural-add-task` | `neural_add_task` | Create a task node manually |
| `neural-add-bug` | `neural_add_bug` | Create a bug node manually |
| `neural-tasks` | `neural_list_tasks` | List tasks filtered by status or priority |
| `neural-context` | `neural_context` | Get a compact token-budgeted context snapshot |
| `neural-insight` | `neural_generate_docs` | Generate full technical documentation from accumulated insights |

### Bundled Agents

Three agents ship with the plugin and are pre-wired to the MCP tools:

| Agent | Trigger | Purpose |
|-------|---------|---------|
| `neural-memory:neural-explorer` | Codebase exploration tasks | Semantic graph search + `neural_query` / `neural_inspect` instead of raw grep |
| `neural-memory:neural-insight-collector` | After significant implementations or "remember this" | Captures technical insights into the insight bank |
| `neural-memory:neural-doc-writer` | `/neural-insight` or "generate docs" | Synthesizes all insights into structured technical documentation |

---

## MCP Tools Reference

All tools accept `project_root` (default `"."`).

---

### Core: Index & Search

#### `neural_index`
Full codebase index.

```
project_root  string   Project root (default: ".")
mode          string   "ast_only" | "api" | "both" (default: from config)
```

What it does:
1. Discovers all files matching `include_patterns` (8 languages)
2. Parses AST → nodes + call/import/inheritance edges
3. Imports bugs from `.claude/context-log-gotchas.md`
4. Imports phases/tasks from `.claude/context-log-tasks-*.md`
5. Computes graph importance scores
6. Enriches high-importance nodes with LSP hover/diagnostics
7. Generates project + directory overview nodes
8. Computes 138-dim composite embeddings
9. Optionally generates Claude API summaries for key nodes

---

#### `neural_update`
Incremental update — only processes changed files.

```
project_root  string   Project root (default: ".")
```

Uses git diff and file-hash comparison. Re-parses only modified files, preserves all bug/task/insight/overview nodes.

---

#### `neural_query`
Semantic search across all four layers.

```
query         string   Natural language or keyword query (required)
project_root  string   Project root (default: ".")
limit         int      Max results (default: 10)
node_type     string   Filter: "function" | "class" | "bug" | "task" | "insight" | ...
category      string   Filter: "codebase" | "bugs" | "tasks" | "insights"
```

Returns ranked results with node type, file path, importance score, summary, and edge counts.

---

#### `neural_inspect`
Full deep-dive into a specific node.

```
node_id       string   Node ID from neural_query results
node_name     string   Node name (alternative to node_id)
project_root  string   Project root (default: ".")
show_code     bool     Include raw source code (default: false)
trace_calls   bool     Trace full call chain (default: false)
```

Returns: parent, children, callers (up to 50), callees, LSP hover doc, diagnostics, importance, summary.

---

#### `neural_status`
Index health and staleness check.

```
project_root  string   Project root (default: ".")
```

Returns: total nodes/edges/files, last index time, git commits behind, freshness verdict.

---

### Core: Node Management

#### `neural_add_bug`
Manually create a bug node.

```
description      string   What went wrong (required, min 3 chars)
severity         string   "low" | "medium" | "high" | "critical" (default: "medium")
file_path        string   Source file this bug relates to
line_start       int      Starting line number
line_end         int      Ending line number
root_cause       string   Root cause description
fix_description  string   How it was / should be fixed
project_root     string   Project root (default: ".")
```

---

#### `neural_add_task`
Manually create a task node.

```
title           string      Task title (required, min 3 chars)
phase_name      string      Phase to attach this task to (created if missing)
priority        string      "low" | "medium" | "high" (default: "medium")
task_status     string      "new" | "pending" | "in_progress" | "testing" | "done" (default: "pending")
related_files   list[str]   File paths this task relates to
project_root    string      Project root (default: ".")
```

> `"new"` is an alias for `"pending"`.

---

#### `neural_list_tasks`
List task nodes with optional filters.

```
status          string   Filter by status: "pending" | "in_progress" | "testing" | "done"
priority        string   Filter by priority: "low" | "medium" | "high"
include_archived bool    Include archived/done tasks (default: false)
project_root    string   Project root (default: ".")
```

---

#### `neural_update_task`
Update a task's status or priority.

```
node_id         string   Task node ID (required)
field           string   Field to update: "task_status" | "priority"
value           string   New value
project_root    string   Project root (default: ".")
```

---

### Insight Bank

The insight bank accumulates technical knowledge across sessions — implementation decisions, architecture patterns, performance tradeoffs. Use `/neural-insight` to synthesize everything into structured documentation.

#### `neural_add_insight`
Save a technical insight into the knowledge graph.

```
content         string      The insight text (required, min 10 chars)
topic           string      Topic area: "storage" | "hooks" | "embeddings" | "cli" | ... (required)
related_files   list[str]   File paths this insight relates to (optional)
project_root    string      Project root (default: ".")
```

Insights are **deduplicated by topic + content** — re-saving the same insight updates it rather than creating a duplicate.

---

#### `neural_list_insights`
Browse accumulated insights.

```
topic           string   Filter by topic (optional — omit for all)
project_root    string   Project root (default: ".")
```

---

#### `neural_generate_docs`
Synthesize all insights into technical documentation.

```
project_root    string   Project root (default: ".")
```

Gathers all insight nodes, groups by topic, follows `RELATES_TO` edges to include code references, renders structured markdown. Also writes to `.neural-memory/technical-docs.md`.

---

### Session Context

#### `neural_context`
Token-budgeted context snapshot (~500 tokens).

```
query_hint      string   Optional query for relevant node search
project_root    string   Project root (default: ".")
token_budget    int      Target token budget (default: 500)
```

Returns: index health, project overview, active bugs, active tasks, insight bank summary, and semantically relevant nodes.

---

#### `neural_save_context`
Save the current session state to `.neural-memory/session_context.md`.

```
project_root    string   Project root (default: ".")
```

Captures: active tasks with code connections, active bugs with code connections, insight bank summary, top important nodes, recent git commits. Called automatically by the Stop hook at session end.

---

### Dashboard & Visualization

#### `neural_serve`
Start the interactive dashboard HTTP server.

```
project_root    string   Project root (default: ".")
port            int      Port (default: 7891)
open_browser    bool     Auto-open in browser (default: true)
regenerate      bool     Regenerate HTML before serving (default: true)
```

Opens `http://localhost:7891` with three views: **Hierarchy** treemap, **Vectors** semantic scatter, **Graph** force-directed layout.

**Run outside Claude Code:**
```bash
neural-memory-viz                        # start at port 7891
neural-memory-viz --port 8080            # custom port
neural-memory-viz --no-browser           # headless
neural-memory-viz --project-root /path/to/project
```

---

#### `neural_stop_serve`
Stop the running dashboard server.

---

#### `neural_visualize_dashboard`
Generate the interactive D3 dashboard as a static HTML file.

```
project_root    string   Project root (default: ".")
output_path     string   Output path (default: ".neural-memory/dashboard.html")
```

Produces a single self-contained HTML file (~750KB) with:
- **Hierarchy tab** — treemap of module → class → function, sized by importance
- **Vectors tab** — PCA scatter of all embeddings, clustered by semantic similarity
- **Graph tab** — force-directed layout, draggable nodes, edge type coloring, zoom/pan
- **Sidebar** — filter by category, node type, importance, status, text search
- **Detail panel** — click any node for full info: summary, LSP data, diagnostics, file path

---

### Advanced

#### `neural_index_db`
Index a live database schema as graph nodes.

```
connection_string  string   Database connection string (required)
project_root       string   Project root (default: ".")
```

Creates TABLE, COLUMN, and VIEW nodes with REFERENCES edges, enabling queries like "what tables does this function read from?"

---

#### `neural_fetch_docs`
Fetch and index external package documentation.

```
package         string   Package name or URL (required)
project_root    string   Project root (default: ".")
```

Stores external docs as graph nodes so Claude can cross-reference your code with library documentation.

---

#### `neural_config`
View or update configuration.

```
project_root              string   Project root (default: ".")
action                    string   "view" | "set_mode" | "add_exclude" | "add_redaction_pattern" | "set_staleness_threshold"
value                     string   Value for the action
```

Available modes: `ast_only` (no API/LSP calls), `api` (API enrichment only), `both` (heuristic + API).

---

## Context Log Auto-Import

Neural Memory automatically reads your `.claude/` context files on every `/neural-index`:

**`.claude/context-log-gotchas.md`** → Bug nodes

```markdown
## 2024-01-15 — Short description of the bug

**File**: `neural_memory/models.py`
**Root cause**: What caused it.
**Fix**: How it was resolved.
```

**`.claude/context-log-tasks-01.md`** (and `tasks-02.md`, etc.) → Phase + Task nodes

```markdown
# Phase 1 — Core data model

## Fix 1 — Add BUG node type

**Status**: [x] DONE
**File**: `neural_memory/models.py` lines 10-40
```

The H1 becomes a PHASE node; each `## Fix N` becomes a TASK node with `PHASE_CONTAINS` and `RELATES_TO` edges to matched code nodes. Files are only re-imported when they change (mtime-gated).

---

## Task Lifecycle

Tasks move through states tracked in the graph:

```
new / pending  →  in_progress  →  testing  →  done
                                              ↓
                                          (auto-archived)
```

| Status | Meaning |
|--------|---------|
| `pending` | Not started (also aliased as `new`) |
| `in_progress` | Actively being worked on |
| `testing` | Implementation complete, under review/test |
| `done` | Complete — auto-archived at session end |

Update a task's status:
```json
Tool: neural_update_task
{ "node_id": "abc123", "field": "task_status", "value": "in_progress" }
```

Or use the `/neural-tasks` skill to list and filter tasks.

---

## Insight Bank

The insight bank accumulates technical knowledge that would otherwise be lost between sessions:

- **Non-obvious design decisions** — *why* something is done a specific way
- **Performance characteristics** — tradeoffs, bottlenecks, gotchas
- **Architecture patterns** — not visible from code alone
- **Implementation constraints** — things that affect future changes

### Building the insight bank

Insights accumulate automatically when the `neural-insight-collector` agent runs (triggered after significant implementations), or you can save them explicitly:

```json
Tool: neural_add_insight
{
  "content": "The bump_version.py script atomically updates 4 files in one run to ensure version consistency. Always run it before staging — never manually edit version strings.",
  "topic": "versioning",
  "related_files": ["scripts/bump_version.py"]
}
```

### Generating documentation

```
/neural-memory:neural-insight
```

Or use the `neural-doc-writer` agent directly. Output is grouped by topic with code references and written to `.neural-memory/technical-docs.md`.

---

## Hooks & Session Persistence

Two hooks run automatically after plugin installation:

### `UserPromptSubmit` — Context Injection
Fires before each message. Injects a compact context snapshot (~500 tokens) covering:
- Index staleness status
- Project overview
- Active bugs and tasks
- Insight bank summary

On first prompt of each session, also loads `.neural-memory/session_context.md` from the previous session.

### `Stop` — Session Save
Fires when the session ends. Automatically:
1. Saves a rich snapshot to `.neural-memory/session_context.md` (tasks + bugs + code connections + recent git log)
2. Archives completed tasks and fixed bugs
3. Runs incremental update if the index is stale and change set is small (≤3 commits, ≤10 files)

This creates **cross-session continuity** — Claude picks up exactly where the last session left off.

---

## LSP Enrichment

If Pyright or pylsp is installed, neural memory enriches nodes with `importance >= 0.3`:

```bash
npm install -g pyright       # recommended
pip install python-lsp-server  # alternative
```

Enriched nodes gain:
- `lsp_hover_doc` — resolved type signatures and docstrings
- `lsp_diagnostics` — type errors and warnings at definition site

Disable via config: `lsp_enabled: false` in `.neural-memory/config.json`.

---

## Semantic Search

Requires numpy:
```bash
pip install "neural-memory-mcp[vectors]"
```

Neural Memory builds **138-dimensional composite embeddings**:
- **100 dims** — TF-IDF corpus vectorized with Truncated SVD (content semantics)
- **38 dims** — structural graph features: node type one-hot (14), in/out degree (2), edge profile (10), metadata (12)

Search algorithm:
1. **Seed phase** — cosine similarity against all embeddings, take top-K seeds
2. **Branch expansion** — traverse graph edges from seeds, apply decay by hop count
3. **Rank phase** — combine cosine score × importance × (1 / hop_penalty)

Embeddings auto-recompute when model version changes (version-stamped in DB).

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      Claude Code                             │
│   Skills: /neural-index /neural-query /neural-insight ...   │
│   Agents: neural-explorer  neural-insight-collector  ...    │
└─────────────────────┬────────────────────────────────────────┘
                      │ MCP (stdio JSON-RPC)
┌─────────────────────▼────────────────────────────────────────┐
│                    MCP Server (26 tools)                     │
│                  neural_memory/server.py                     │
└──────┬──────────────┬──────────────┬──────────┬─────────────┘
       │              │              │          │
  ┌────▼────┐   ┌─────▼─────┐  ┌───▼────┐ ┌──▼──────────┐
  │ Indexer │   │  Search   │  │ Graph  │ │  Dashboard  │
  │ AST +   │   │ Embedding │  │Inspect │ │  D3.js      │
  │ context │   │ + branch  │  │+ callers│ │  3 views    │
  │  logs   │   └─────┬─────┘  │  + LSP │ └─────────────┘
  └────┬────┘         │        └───┬────┘
       │              │            │
  ┌────▼──────────────▼────────────▼────┐
  │           SQLite Storage            │
  │        .neural-memory/memory.db     │
  │  nodes | edges | embeddings |       │
  │  file_hashes | index_state          │
  └─────────────────────────────────────┘

Four Node Layers
─────────────────
Codebase layer        Bugs layer     Tasks layer     Insights layer
──────────────        ──────────     ───────────     ──────────────
module                bug            phase           insight
class                                task
function
method
project_overview
directory_overview
```

### Edge Types

| Type | Connects |
|------|----------|
| `calls` | function → function |
| `imports` | module → module |
| `inherits` | class → class |
| `contains` | parent → child |
| `relates_to` | bug / task / insight → code node |
| `fixed_by` | bug → fix function |
| `phase_contains` | phase → task |
| `task_contains` | task → subtask |
| `references` | table → table (FK) |
| `queries` | function → table (read) |
| `writes_to` | function → table (write) |

---

## Data Storage

All data lives in `.neural-memory/` inside your project:

```
.neural-memory/
  memory.db             # SQLite: nodes, edges, embeddings, state
  config.json           # Your settings
  session_context.md    # Auto-saved session state (cross-session continuity)
  technical-docs.md     # Generated from insight bank via /neural-insight
  dashboard.html        # Generated interactive visualization
  d3.min.js             # Cached D3 library (auto-downloaded)
  rtk_prompted          # One-time RTK install flag
```

`.neural-memory/` is added to `.gitignore` automatically.

---

## Configuration

Settings are stored in `.neural-memory/config.json` and editable via `/neural-config`.

| Setting | Default | Description |
|---------|---------|-------------|
| `index_mode` | `"both"` | `"ast_only"` / `"api"` / `"both"` |
| `include_patterns` | `["**/*.py"]` | Files to index |
| `exclude_patterns` | `["**/.venv/**", ...]` | Files to skip |
| `importance_threshold` | `0.2` | Min importance for API summarization |
| `staleness_threshold` | `5` | Commits behind before warning |
| `lsp_enabled` | `true` | Enable LSP type enrichment |
| `lsp_server` | `"auto"` | `"auto"` / `"pyright-langserver"` / `"pylsp"` / `"none"` |

---

## Privacy & Security

- **Local only** — the index never leaves your machine unless you use `api` or `both` mode
- **Automatic redaction** — secrets matching common patterns are replaced with `[REDACTED]` before any storage or API call
- **AST-aware** — variables named `secret`, `password`, `token`, `api_key`, `auth`, `credential` have their values redacted
- **Configurable** — add custom regex patterns via `/neural-config`
- **Mode control** — use `ast_only` for air-gapped or sensitive projects

---

## Troubleshooting

**`sqlite3` error on first index**
Delete `.neural-memory/memory.db` and re-run `/neural-index`. The schema will be recreated.

**Bug/task nodes not appearing in search**
Run `/neural-index` (not `/neural-update`) to force re-import of context logs.

**Embeddings not computing**
Install numpy: `pip install "neural-memory-mcp[vectors]"`. Without it, search falls back to name/summary text matching.

**LSP enrichment skipped**
Install `pyright` (`npm install -g pyright`) or `pylsp` (`pip install python-lsp-server`).

**`neural-memory: command not found`**
The pip scripts directory isn't on PATH. Use `python -m neural_memory.server` in your MCP config instead.

**Hooks not firing**
Run `neural-memory-setup doctor` to check hook registration. Re-run `neural-memory-setup install` to repair.

**Insights not appearing in `/neural-insight` docs**
Check with `neural_list_insights` — if empty, start saving insights with `neural_add_insight` or use the `neural-insight-collector` agent after your next implementation.

**Session context not loading on new session**
Check that `.neural-memory/session_context.md` exists. If missing, the Stop hook may not have fired — run `neural_save_context` manually.

```bash
neural-memory-setup doctor   # full diagnosis
```

---

## Contributing

```bash
git clone https://github.com/Yakoub-ai/neural-memory.git
cd neural-memory
pip install -e ".[all]"
pip install -e ".[test]"
pytest --tb=short -q
```

---

## License

MIT — see [LICENSE](LICENSE).
