---
name: neural-add-task
description: Manually log a task or phase into the neural memory knowledge graph. Links the task to related code files so it appears in semantic search results.
---

# Neural Memory — Add Task

Manually create a task node in the knowledge graph. Use this to track in-progress work, planned features, or phase milestones directly in the graph without editing context log files.

## Usage

Call the `neural_add_task` tool:
```
neural_add_task(
    title="Implement caching for importance scores",
    phase_name="Phase 3 — Performance",
    priority="high",
    task_status="in_progress",
    related_files=["neural_memory/graph.py", "neural_memory/storage.py"]
)
```

## Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `title` | Yes | Task title |
| `phase_name` | No | Phase to attach this task to (creates the phase node if it doesn't exist) |
| `priority` | No | `"low"` / `"medium"` / `"high"` (default: `"medium"`) |
| `task_status` | No | `"pending"` / `"in_progress"` / `"done"` (default: `"pending"`) |
| `related_files` | No | List of file paths this task relates to |
| `project_root` | No | Project root (default: `"."`) |

## Notes
- The task node is linked to matching code nodes via `RELATES_TO` edges
- Phase nodes are created automatically from `phase_name` if they don't exist
- Tasks are also auto-imported from `.claude/context-log-tasks-*.md` on every `/neural-memory:neural-index`
- Use `/neural-memory:neural-query` to search tasks alongside code and bugs
