---
name: neural-context
description: Get a compact, token-budgeted context snapshot AND save a rich persistent session snapshot — project overview, active bugs/tasks with code connections, and recent git commits. Use this to orient quickly and persist context for the next session.
---

# Neural Memory — Compact Context + Session Save

Get a token-budgeted (~500 token) snapshot of the current project's knowledge graph, and save a richer persistent snapshot for cross-session continuity.

## What this does

Returns in one call:
- **Staleness status** — is the index current?
- **Project overview** — what this codebase does
- **Active bugs** — open, non-archived bugs
- **Active tasks** — pending/in-progress/testing, non-archived tasks
- **Relevant nodes** — code nodes semantically matching your query (if provided)

And saves to `.neural-memory/session_context.md`:
- Active tasks and bugs with their **code node connections** (file + line)
- Last 5 git commits
- Top 5 most important code nodes

## When to use

- At the start of any task to orient yourself
- At the end of a session to persist context for the next session
- Between major steps to check active bugs and tasks
- Before modifying a specific area (pass that area as `query_hint`)

## How to call

**Via MCP tool (snapshot + save)**:
```json
Tool: neural_save_context
{ "token_budget": 800 }
```

**Via MCP tool (snapshot only)**:
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

## Session Persistence

The saved snapshot at `.neural-memory/session_context.md` is automatically:
- **Written** at session end (Stop hook) — no manual action needed
- **Loaded** at session start (UserPromptSubmit hook) when no query hint is present
- **Refreshed** whenever you run `/neural-context` manually

This gives new sessions immediate deep context without re-discovery.

## Lifecycle

Bugs marked `bug_status='fixed'` and tasks marked `task_status='done'` are **archived** automatically at session end. Archived items:
- Do **not** appear in `neural_context` output
- Do **not** appear in `neural_query` by default
- Are still findable with `neural_query` using `include_archived=true`
- Can be manually archived/unarchived with `neural_archive`

This keeps active context lean as your project grows.
