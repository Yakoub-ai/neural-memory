---
name: neural-add-task
description: Manually log a task or phase into the neural memory knowledge graph. Links the task to related code files so it appears in semantic search results.
---

# Neural Memory — Add Task

Manually create a task node in the knowledge graph. Use this to track in-progress work, planned features, or phase milestones directly in the graph without editing context log files.

## How to call

**Via MCP tool** (neural-memory configured as MCP server in Claude Code):
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

**Via Python** (working directly in the project):
```python
import asyncio
from neural_memory.server import neural_add_task, AddTaskInput

asyncio.run(neural_add_task(AddTaskInput(
    title="Implement caching for importance scores",
    phase_name="Phase 3 — Performance",
    priority="high",
    task_status="in_progress",
    related_files=["neural_memory/graph.py", "neural_memory/storage.py"],
)))
```

## Parameters

| Parameter | Required | Type | Description |
|-----------|----------|------|-------------|
| `title` | Yes | str | Task title (min 3 chars) |
| `phase_name` | No | str | Phase to attach this task to (creates the phase node if it doesn't exist) |
| `priority` | No | str | `"low"` / `"medium"` / `"high"` (default: `"medium"`) |
| `task_status` | No | str | `"pending"` / `"in_progress"` / `"done"` (default: `"pending"`) |
| `related_files` | No | list[str] | File paths this task relates to |
| `project_root` | No | str | Project root (default: `"."`) |

## Notes
- The task node is linked to matching code nodes via `RELATES_TO` edges
- Phase nodes are created automatically from `phase_name` if they don't exist
- Tasks are also auto-imported from `.claude/context-log-tasks-*.md` on every `/neural-index`
- Use `/neural-query` to search tasks alongside code and bugs
