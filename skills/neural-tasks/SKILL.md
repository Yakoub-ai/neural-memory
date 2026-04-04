---
name: neural-tasks
description: Manage the full task lifecycle in the neural knowledge graph — list, create, update status and priority. Tasks connect to code nodes and appear in every prompt via the context hook.
---

# Neural Memory — Task Management

Manage tasks across their full lifecycle: `new` → `in_progress` → `testing` → `done` → auto-archived.

Tasks are first-class citizens in the knowledge graph: they link to code nodes via `RELATES_TO` edges, appear in every prompt via the context hook, and are auto-archived when done.

## Task Statuses

| Status | Meaning |
|--------|---------|
| `new` | Alias for `pending` — task not yet started |
| `pending` | Task not yet started |
| `in_progress` | Actively being worked on |
| `testing` | Implementation done, under verification |
| `done` | Complete — will be auto-archived at session end |

## List Tasks

```json
Tool: neural_list_tasks
{}
```

Filter by status or priority:
```json
Tool: neural_list_tasks
{ "status": "in_progress", "priority": "high" }
```

Include archived tasks:
```json
Tool: neural_list_tasks
{ "include_archived": true }
```

## Create a Task

```json
Tool: neural_add_task
{
  "title": "Implement caching for importance scores",
  "phase_name": "Phase 3 — Performance",
  "priority": "high",
  "task_status": "in_progress",
  "related_files": ["neural_memory/graph.py", "neural_memory/storage.py"]
}
```

Use `task_status: "new"` or `"pending"` for not-yet-started tasks.

## Update a Task

Update by task name (substring match) or node ID:
```json
Tool: neural_update_task
{ "title_or_id": "caching", "task_status": "testing" }
```

Update priority:
```json
Tool: neural_update_task
{ "title_or_id": "caching", "priority": "high" }
```

Add related files:
```json
Tool: neural_update_task
{
  "title_or_id": "caching",
  "task_status": "done",
  "related_files": ["neural_memory/embeddings.py"]
}
```

## Parameters

### neural_list_tasks

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `status` | str | `null` | Filter: `pending`, `in_progress`, `testing`, `done` |
| `priority` | str | `null` | Filter: `low`, `medium`, `high` |
| `include_archived` | bool | `false` | Include done/archived tasks |
| `project_root` | str | `"."` | Project root |

### neural_update_task

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `title_or_id` | str | required | Task name substring or full node ID |
| `task_status` | str | `null` | New status: `pending`/`new`/`in_progress`/`testing`/`done` |
| `priority` | str | `null` | New priority: `low`/`medium`/`high` |
| `related_files` | list[str] | `[]` | Additional file paths to link |
| `project_root` | str | `"."` | Project root |

### neural_add_task

| Parameter | Required | Type | Description |
|-----------|----------|------|-------------|
| `title` | Yes | str | Task title (min 3 chars) |
| `phase_name` | No | str | Phase to attach this task to (auto-created if new) |
| `priority` | No | str | `"low"` / `"medium"` / `"high"` (default: `"medium"`) |
| `task_status` | No | str | `"new"` / `"pending"` / `"in_progress"` / `"testing"` / `"done"` (default: `"pending"`) |
| `related_files` | No | list[str] | File paths this task relates to |
| `project_root` | No | str | Project root (default: `"."`) |

## Autonomous Lifecycle

Tasks integrate with the full neural-memory automation stack:

- **Context hook**: Active tasks appear in every prompt automatically (no manual `/neural-context` needed)
- **Session end**: `done` tasks are auto-archived — no manual cleanup needed
- **Indexing**: Tasks are auto-imported from `.claude/context-log-tasks-*.md` on `/neural-index`
- **Session save**: Task→code connections are saved to `session_context.md` at session end

## Notes
- Phase nodes are created automatically from `phase_name` if they don't exist
- `new` is accepted as an alias for `pending` in all tools
- `neural_update_task` resolves by name substring — no need to know the exact node ID
- Use `/neural-query` to search tasks alongside code and bugs
