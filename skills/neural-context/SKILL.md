---
name: neural-context
description: Get a compact, token-budgeted context snapshot — project overview, active bugs, active tasks, and nodes relevant to your current query. Use this to orient quickly without burning context.
---

# Neural Memory — Compact Context

Get a token-budgeted (~500 token) snapshot of the current project's knowledge graph.

## What this does

Returns in one call:
- **Staleness status** — is the index current?
- **Project overview** — what this codebase does
- **Active bugs** — open, non-archived bugs
- **Active tasks** — pending/in-progress, non-archived tasks
- **Relevant nodes** — code nodes semantically matching your query (if provided)

## When to use

- At the start of any task to orient yourself
- Between major steps to check active bugs and tasks
- Before modifying a specific area (pass that area as `query_hint`)

## How to call

**Via MCP tool**:
```json
Tool: neural_context
{ "query_hint": "authentication middleware", "token_budget": 500 }
```

**Minimal call** (no query hint — overview + bugs + tasks only):
```json
Tool: neural_context
{}
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project_root` | str | `"."` | Project root directory |
| `query_hint` | str | `null` | Prompt or keyword to drive semantic pre-fetch of relevant nodes |
| `token_budget` | int | `500` | Approximate token budget (100–2000) |

## Lifecycle

Bugs marked `bug_status='fixed'` and tasks marked `task_status='done'` are **archived** automatically at session end. Archived items:
- Do **not** appear in `neural_context` output
- Do **not** appear in `neural_query` by default
- Are still findable with `neural_query` using `include_archived=true`
- Can be manually archived/unarchived with `neural_archive`

This keeps active context lean as your project grows.
