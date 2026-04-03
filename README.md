# Neural Memory

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![MCP Plugin](https://img.shields.io/badge/Claude%20Code-Plugin-green.svg)](https://github.com/Yakoub-ai/neural-memory)

> A persistent knowledge graph plugin for Claude Code that maps your codebase into three layers — code, bugs, and tasks — so Claude always has deep context without burning tokens re-reading files.

---

## What It Does

Neural Memory builds and maintains a **three-layer directed knowledge graph** of your project:

| Layer | Nodes | Connected To |
|-------|-------|-------------|
| **Codebase** | modules, classes, functions, methods, project/dir overviews | each other via calls/imports/inheritance/containment |
| **Bugs** | bug entries from `.claude/context-log-gotchas.md` | code nodes by file path (`RELATES_TO`) |
| **Tasks** | phases and tasks from `.claude/context-log-tasks-*.md` | code nodes (`RELATES_TO`), each other (`PHASE_CONTAINS`) |

Every node gets:
- **Layered summaries** — heuristic one-liner from AST, optionally enriched via Claude API
- **128-dim composite embeddings** — 100-dim TF-IDF+SVD content vector + 28-dim structural graph features
- **LSP enrichment** — type signatures and diagnostics from Pyright/pylsp (if installed)
- **Importance scores** — graph-centrality weighted by node type

Search uses a **three-phase branch algorithm**: seed by cosine similarity → expand through graph edges with decay/pruning → re-rank by combined score. Sensitive values (API keys, tokens, passwords) are automatically redacted before storage.

---

## Quick Start

**Step 1 — Install via Claude Code plugin marketplace** (recommended — no manual setup):
```
/plugin marketplace add Yakoub-ai/neural-memory
/plugin install neural-memory@Yakoub-ai
```

The plugin automatically registers the MCP server via `uvx` — no pip install or config editing needed.

**Step 2 — Build the knowledge graph:**
```
/neural-memory:neural-index
```

**Done.** Claude now has persistent, token-efficient context about your entire codebase.

> **Manual setup?** See [Installation Options](#installation-options) below.

---

## Installation Options

### Option A — Claude Code Plugin (Recommended)

No pip install required. The MCP server runs via `uvx` directly from PyPI.

In Claude Code:
```
/plugin marketplace add Yakoub-ai/neural-memory
/plugin install neural-memory@Yakoub-ai
```

Skills are namespaced: `/neural-memory:neural-index`, `/neural-memory:neural-query`, etc.

### Option B — Manual MCP Setup

**1. Install the package:**
```bash
# Core only (no embeddings/visualization)
pip install neural-memory-mcp

# With semantic search (recommended)
pip install "neural-memory-mcp[vectors]"

# With everything (embeddings + dashboard)
pip install "neural-memory-mcp[all]"
```

**2. Add to your project's `.mcp.json`:**
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

Or add to `~/.claude.json` for global access across all projects.

**3. Add the agent hook** to your project's `CLAUDE.md` so Claude always checks freshness:
```markdown
## Agent Hook
On each invocation, check neural memory staleness by running the `neural_status` tool.
If the index is stale or uninitialized, inform the user and suggest the appropriate action.
```

### Option C — From source

```bash
git clone https://github.com/Yakoub-ai/neural-memory.git
cd neural-memory
pip install -e ".[all]"
```

---

## Commands Reference

### Slash Commands (in Claude Code)

When installed via plugin, skills are namespaced as `/neural-memory:<skill>`. When used standalone (`.claude/` directory), they run as `/neural-<skill>`.

| Skill | MCP Tool | Description |
|-------|----------|-------------|
| `neural-index` | `neural_index` | Full index of all Python files + context logs. Parses AST, imports bugs/tasks, computes embeddings, generates overviews. Run once at project start or after large changes. |
| `neural-update` | `neural_update` | Incremental update via git diff + file hash comparison. Only re-parses changed files. Fast — run after any code change. |
| `neural-query` | `neural_query` | Semantic search across all three layers. Finds functions, classes, bugs, and tasks by concept. Uses 128-dim embeddings + graph expansion. |
| `neural-inspect` | `neural_inspect` | Deep-dive into a node: full summary, callers, callees, parent/children, LSP type info, diagnostics. |
| `neural-status` | `neural_status` | Index health check — shows total nodes/edges, last index time, git staleness, and a freshness verdict. |
| `neural-config` | `neural_config` | View or modify settings: indexing mode, include/exclude patterns, redaction rules, staleness threshold. |
| `neural-visualize` | `neural_serve` | Generate the interactive dashboard and open it in your browser at `http://localhost:7891`. |
| `neural-stop` | `neural_stop_serve` | Stop the running dashboard HTTP server. |

### MCP Tools (callable by Claude directly)

All tools accept a `project_root` parameter (default `"."`).

---

#### `neural_index`
Full codebase index.

```
Parameters:
  project_root  string  Project root directory (default: ".")
  mode          string  "ast_only" | "api" | "both" (default: from config)
```

**What it does:**
1. Discovers all Python files matching `include_patterns`
2. Parses AST → module/class/function/method nodes + call/import/inheritance edges
3. Imports bug nodes from `.claude/context-log-gotchas.md`
4. Imports phase/task nodes from `.claude/context-log-tasks-*.md`
5. Computes graph importance scores
6. Optionally enriches high-importance nodes with LSP hover/diagnostics
7. Generates project + directory overview nodes
8. Computes 128-dim composite embeddings for all nodes
9. Optionally generates Claude API summaries for high-importance nodes

---

#### `neural_update`
Incremental update — only processes changed files.

```
Parameters:
  project_root  string  Project root directory (default: ".")
```

Uses git diff and file-hash comparison to detect changes. Re-parses only modified files, preserves all bug/task/overview nodes.

---

#### `neural_query`
Semantic search across code, bugs, and tasks.

```
Parameters:
  query         string  Natural language or keyword query
  project_root  string  Project root directory (default: ".")
  limit         int     Max results (default: 10)
  node_type     string  Filter by type: "function" | "class" | "bug" | "task" | ...
  category      string  Filter by layer: "codebase" | "bugs" | "tasks"
```

Returns ranked results with node type, file path, importance score, summary, and edge counts.

---

#### `neural_inspect`
Full deep-dive into a specific node.

```
Parameters:
  node_id       string  Node ID (from neural_query results)
  node_name     string  Node name to look up (alternative to node_id)
  project_root  string  Project root directory (default: ".")
  show_code     bool    Include raw source code (default: false)
  trace_calls   bool    Trace full call chain (default: false)
```

Returns: parent, children, callers (up to 50), callees, LSP hover doc, diagnostics, importance, summary.

---

#### `neural_status`
Index health and staleness check.

```
Parameters:
  project_root  string  Project root directory (default: ".")
```

Returns: total nodes/edges/files, last full index time, last incremental time, git commits behind, freshness verdict.

---

#### `neural_config`
View or update configuration.

```
Parameters:
  project_root      string  Project root directory (default: ".")
  action            string  "view" | "set_mode" | "add_exclude" | "add_redaction_pattern" | "set_staleness_threshold"
  value             string  Value for the action (e.g. "ast_only", "**/.venv/**", "(?i)my_secret", "10")
```

Available modes: `ast_only` (no API calls), `api` (API only), `both` (heuristic + API enrichment).

---

#### `neural_serve`
Start the interactive dashboard HTTP server.

```
Parameters:
  project_root  string  Project root directory (default: ".")
  port          int     Port to serve on (default: 7891)
  open_browser  bool    Auto-open in browser (default: true)
  regenerate    bool    Regenerate dashboard HTML before serving (default: true)
```

Opens `http://localhost:7891` with three views: Hierarchy treemap, Semantic radial tree, Force-directed graph.

---

#### `neural_stop_serve`
Stop the running dashboard server.

```
Parameters: none
```

---

#### `neural_visualize`
Generate static HTML visualizations (legacy).

```
Parameters:
  project_root  string  Project root directory (default: ".")
  mode          string  "hierarchy" | "vectors" | "both"
  color_by      string  "node_type" | "importance" | "category"
  dimensions    int     PCA dimensions for vector view (2 or 3)
```

Writes HTML files to `.neural-memory/`. Requires `numpy` (`pip install neural-memory[vectors]`).

---

#### `neural_visualize_dashboard`
Generate the interactive D3 knowledge-graph dashboard.

```
Parameters:
  project_root  string  Project root directory (default: ".")
  output_path   string  Output path (default: ".neural-memory/dashboard.html")
```

Produces a single self-contained HTML file (~750KB) with:
- **Hierarchy tab** — treemap of module → class → function, sized by importance
- **Vectors tab** — PCA scatter of all node embeddings, clustered by semantic similarity
- **Graph tab** — force-directed layout with draggable nodes, edge type coloring, zoom/pan
- **Sidebar filters** — category (codebase/bugs/tasks), node type, importance slider, status, text search, treemap depth
- **Detail panel** — click any node for full info: summary, LSP data, diagnostics, file path

Requires `numpy` for PCA. Falls back to circular layout without it.

---

#### `neural_add_bug`
Manually create a bug node (when not using context logs).

```
Parameters:
  description       string  What went wrong (required)
  severity          string  "low" | "medium" | "high" | "critical" (default: "medium")
  file_path         string  Source file this bug relates to
  line_start        int     Starting line number
  line_end          int     Ending line number
  root_cause        string  Root cause description
  fix_description   string  How it was / should be fixed
  project_root      string  Project root directory (default: ".")
```

---

#### `neural_add_task`
Manually create a task node.

```
Parameters:
  title           string  Task title (required)
  phase_name      string  Phase to attach this task to (creates phase if missing)
  priority        string  "low" | "medium" | "high" (default: "medium")
  task_status     string  "pending" | "in_progress" | "done" (default: "pending")
  related_files   list    File paths this task relates to
  project_root    string  Project root directory (default: ".")
```

---

## Context Log Auto-Import

Neural Memory automatically reads your `.claude/` context files on every `/neural-index`:

**`.claude/context-log-gotchas.md`** → Bug nodes

Each entry matching this format becomes a BUG node:
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

The H1 heading becomes a PHASE node; each `## Fix N — title` becomes a TASK node with a `PHASE_CONTAINS` edge and `RELATES_TO` edges to matched code nodes.

Re-parsing is mtime-gated — files are only re-imported when they change.

---

## LSP Enrichment

If Pyright or pylsp is installed, neural memory enriches nodes with `importance >= 0.3`:

```bash
# Install Pyright (recommended)
npm install -g pyright

# Or pylsp
pip install python-lsp-server
```

Enriched nodes gain:
- `lsp_hover_doc` — resolved type signatures and docstrings
- `lsp_diagnostics` — type errors and warnings at definition site

Disable LSP enrichment via `neural_config`:
```
action: "set_mode", value: "ast_only"   # disables all API + LSP calls
```
Or keep API summaries but disable LSP specifically by setting `lsp_enabled: false` in `.neural-memory/config.json`.

---

## Semantic Search

Requires numpy:
```bash
pip install "neural-memory-mcp[vectors]"
```

Neural Memory builds **128-dimensional composite embeddings**:
- **100 dims** — TF-IDF corpus vectorized with Truncated SVD (content semantics)
- **28 dims** — structural graph features: node type one-hot (14), in/out degree (2), edge profile (10), metadata (2)

Search algorithm:
1. **Seed phase** — cosine similarity against all embeddings, take top-K seeds
2. **Branch expansion** — traverse graph edges from seeds, apply decay by hop count
3. **Rank phase** — combine cosine score × importance × (1 / hop_penalty)

Embeddings auto-recompute when model version changes (version stamped in DB).

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     Claude Code                          │
│   /neural-index  /neural-query  /neural-inspect  ...    │
└────────────────────────┬─────────────────────────────────┘
                         │ MCP (stdio JSON-RPC)
┌────────────────────────▼─────────────────────────────────┐
│                  MCP Server                              │
│              neural_memory/server.py                     │
└──────┬─────────────┬──────────────┬───────────────┬──────┘
       │             │              │               │
  ┌────▼────┐  ┌─────▼─────┐ ┌────▼────┐  ┌──────▼──────┐
  │ Indexer │  │  Search   │ │  Graph  │  │  Dashboard  │
  │  (AST + │  │(Embedding │ │(Inspect │  │  (D3.js     │
  │ context │  │ + branch) │ │+ callers│  │  3 views)   │
  │  logs)  │  └─────┬─────┘ │  + LSP) │  └─────────────┘
  └────┬────┘        │       └────┬────┘
       │             │            │
  ┌────▼─────────────▼────────────▼────┐
  │            SQLite Storage          │
  │         .neural-memory/memory.db   │
  │  nodes | edges | embeddings |      │
  │  file_hashes | index_state         │
  └────────────────────────────────────┘
```

### Three Node Layers

```
Codebase layer          Bugs layer           Tasks layer
──────────────          ──────────           ───────────
module                  bug                  phase
class                                        task
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
| `relates_to` | bug/task → code node |
| `fixed_by` | bug → fix function |
| `phase_contains` | phase → task |
| `task_contains` | task → subtask |

---

## Data Storage

All data lives in `.neural-memory/` inside your project:

```
.neural-memory/
  memory.db          # SQLite: nodes, edges, embeddings, state
  config.json        # Your settings
  d3.min.js          # Cached D3 library (auto-downloaded)
  dashboard.html     # Generated interactive visualization
```

`.neural-memory/` should be in your `.gitignore` (added automatically).

---

## Privacy & Security

- **Local only** — the index never leaves your machine unless you use `api` or `both` mode
- **Automatic redaction** — secrets matching common patterns are replaced with `[REDACTED]` before any storage or API call
- **AST-aware** — variables named `secret`, `password`, `token`, `api_key`, `auth`, `credential` have their values redacted
- **Configurable** — add custom regex patterns, whitelist false positives via `/neural-config`
- **Mode control** — use `ast_only` for air-gapped or sensitive projects

---

## Configuration

Settings are stored in `.neural-memory/config.json` and editable via `/neural-config`.

| Setting | Default | Description |
|---------|---------|-------------|
| `index_mode` | `both` | `ast_only` / `api` / `both` |
| `include_patterns` | `["**/*.py"]` | Files to index |
| `exclude_patterns` | `["**/.venv/**", ...]` | Files to skip |
| `importance_threshold` | `0.2` | Min importance for API summarization |
| `staleness_threshold` | `5` | Commits behind before warning |
| `lsp_enabled` | `true` | Enable LSP enrichment |
| `lsp_server` | `"auto"` | `"auto"` / `"pyright-langserver"` / `"pylsp"` / `"none"` |

---

## Troubleshooting

**`sqlite3` error on first index**
Delete `.neural-memory/memory.db` and re-run `/neural-index`. The schema will be recreated.

**Bug/task nodes not appearing in search**
Run `/neural-index` (not `/neural-update`) to force re-import of context logs.

**Embeddings not computing**
Install numpy: `pip install "neural-memory-mcp[vectors]"`. Without it, search falls back to name/summary text matching.

**LSP enrichment skipped**
Install `pyright` (`npm install -g pyright`) or `pylsp` (`pip install python-lsp-server`). Or disable with `set_lsp: none`.

**`neural-memory: command not found`**
The pip scripts directory isn't on PATH. Use `python -m neural_memory.server` in your MCP config instead.

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
