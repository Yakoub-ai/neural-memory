# Neural Memory

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![MCP Plugin](https://img.shields.io/badge/Claude%20Code-Plugin-green.svg)](https://github.com/Yakoub-ai/neural-memory)

> A knowledge graph plugin for Claude Code that maps your codebase into layered, navigable neural nodes — understand any function without reading the entire file.

---

## Quick Start (3 steps)

**Step 1 — Install the Python package:**
```bash
pip install git+https://github.com/Yakoub-ai/neural-memory.git
```

**Step 2 — Add the marketplace and install the plugin in Claude Code:**
```
/plugin marketplace add Yakoub-ai/neural-memory
/plugin install neural-memory@Yakoub-ai
```

**Step 3 — Build your knowledge graph:**
```
/neural-index
```

That's it. Neural memory is now running.

---

## What It Does

Neural Memory parses your Python codebase into a directed knowledge graph — nodes are functions, classes, methods, and modules; edges are calls, imports, and inheritance. Every node gets layered summaries (one-liner + detailed), so you can understand any piece of code without reading the whole file. The index updates incrementally via git diff, and all sensitive values (API keys, passwords, tokens) are automatically redacted before anything is stored.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Claude Code                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ /neural- │  │ /neural- │  │ /neural- │ ...  │
│  │  index   │  │  query   │  │  inspect │      │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘      │
│       └──────────────┼──────────────┘            │
│                      ▼                           │
│            ┌──────────────────┐                  │
│            │  MCP Server      │                  │
│            │  (neural_memory) │                  │
│            └────────┬─────────┘                  │
└─────────────────────┼───────────────────────────┘
                      ▼
        ┌─────────────────────────────┐
        │       Core Engine           │
        │  ┌─────────┐ ┌──────────┐  │
        │  │ Parser  │ │Summarizer│  │
        │  │ (AST)   │ │(API+Heur)│  │
        │  └────┬────┘ └────┬─────┘  │
        │       ▼            ▼        │
        │  ┌─────────┐ ┌──────────┐  │
        │  │Redactor │ │  Graph   │  │
        │  │(Secrets)│ │(Traverse)│  │
        │  └────┬────┘ └────┬─────┘  │
        │       └──────┬─────┘        │
        │              ▼              │
        │     ┌────────────────┐      │
        │     │ SQLite + JSON  │      │
        │     │ (.neural-memory)│     │
        │     └────────────────┘      │
        └─────────────────────────────┘
```

---

## Commands Reference

| Command | Purpose |
|---------|---------|
| `/neural-index` | Build the full knowledge graph (first time or full rebuild) |
| `/neural-query` | Search for functions, classes, or concepts by name |
| `/neural-inspect` | Deep-dive into a node: callers, callees, call chains, source |
| `/neural-update` | Incremental sync with recent git changes |
| `/neural-status` | Check index health, staleness, and node/edge counts |
| `/neural-config` | View or modify settings (mode, exclusions, redaction, threshold) |

---

## Installation Options

### Option A: Claude Code Plugin (Recommended)

The plugin system handles MCP server registration and slash command setup automatically.

```bash
# 1. Install the Python package
pip install git+https://github.com/Yakoub-ai/neural-memory.git

# 2. In Claude Code, add the marketplace and install
/plugin marketplace add Yakoub-ai/neural-memory
/plugin install neural-memory@Yakoub-ai
```

### Option B: Manual Setup

<details>
<summary>Click to expand manual setup instructions</summary>

**1. Install the package:**
```bash
pip install git+https://github.com/Yakoub-ai/neural-memory.git
```

**2. Add the MCP server to your Claude Code settings:**

Global (`~/.claude/settings.json`) — recommended, works across all projects:
```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "neural-memory"
    }
  }
}
```

If `neural-memory` is not on your PATH, use the full Python form:
```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "python",
      "args": ["-m", "neural_memory.server"]
    }
  }
}
```

**3. Install slash commands:**
```bash
cd your-project
neural-memory-setup install-commands
```

This copies the 6 `/neural-*` command files into your project's `.claude/commands/`.

**4. (Optional) Add the agent hook to your project's CLAUDE.md:**

Append the following so Claude auto-checks staleness on every session:
```markdown
## Neural Memory Agent Hook

On each invocation, check neural memory staleness by running the `neural_status` tool.
If the index is stale or uninitialized, inform the user and suggest the appropriate action.
```

Or run `neural-memory-setup install` for an interactive guided setup.

</details>

---

## Data Residency & Privacy

Three indexing modes — choose based on your data sensitivity:

| Mode | Summaries | Code leaves machine? |
|------|-----------|---------------------|
| `ast_only` | Heuristic (from AST) | **No** |
| `api_only` | Claude API | Yes — high-importance nodes only |
| `both` | Heuristic + API enrichment | Yes — high-importance nodes only |

Switch modes anytime:
```
/neural-config   →   set_mode: ast_only
```

---

## Security

- **Automatic redaction**: API keys, JWT tokens, connection strings, private keys, passwords are masked before storage
- **AST-aware**: Variables named `secret`, `password`, `token`, `api_key`, etc. have values replaced with `[REDACTED]`
- **Custom patterns**: Add your own regex via `/neural-config` → `add_redaction_pattern`
- **Whitelist**: Mark false positives to prevent over-redaction
- **Local storage only**: The index lives in `.neural-memory/` inside your project and is excluded from git by default

---

## Troubleshooting

```bash
neural-memory-setup doctor
```

Checks Python version, sqlite3, mcp, pydantic, and PATH configuration — reports exactly what needs fixing.

**Common issues:**

- `neural-memory: command not found` — the pip install's bin directory isn't on your PATH. Either add it or use the `python -m neural_memory.server` fallback in your MCP config.
- `sqlite3` errors — very rare, but some minimal Python builds omit sqlite3. Reinstall Python from python.org.
- Index feels stale — run `/neural-update` after code changes, or `/neural-index` for a full rebuild.

---

## Roadmap

### v1.1
- [ ] Duplicate code detection (AST-normalized similarity)
- [ ] Multi-language support (TypeScript via tree-sitter)
- [ ] Export graph to visualization tools

### v1.2
- [ ] Semantic search (embeddings-based)
- [ ] Auto-documentation generation
- [ ] PR impact analysis

---

## Contributing

```bash
git clone https://github.com/Yakoub-ai/neural-memory.git
cd neural-memory
pip install -e .
```

PRs welcome. Open an issue first for significant changes.

---

## License

MIT — see [LICENSE](LICENSE).
