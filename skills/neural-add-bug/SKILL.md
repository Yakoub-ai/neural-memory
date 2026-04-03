---
name: neural-add-bug
description: Manually log a bug into the neural memory knowledge graph when not using context log files. Links the bug node to the affected code by file path.
---

# Neural Memory — Add Bug

Manually create a bug node in the knowledge graph. Use this when you discover a bug during a session and want it indexed immediately, without waiting for a context log re-import.

## Usage

Call the `neural_add_bug` tool:
```
neural_add_bug(
    description="What went wrong",
    severity="high",
    file_path="neural_memory/parser.py",
    line_start=142,
    root_cause="Off-by-one in slice index",
    fix_description="Changed range(n) to range(n+1)"
)
```

## Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `description` | Yes | What went wrong |
| `severity` | No | `"low"` / `"medium"` / `"high"` / `"critical"` (default: `"medium"`) |
| `file_path` | No | Source file this bug relates to |
| `line_start` | No | Starting line number |
| `line_end` | No | Ending line number |
| `root_cause` | No | Root cause description |
| `fix_description` | No | How it was or should be fixed |
| `project_root` | No | Project root (default: `"."`) |

## Notes
- The bug node is linked to matching code nodes via `RELATES_TO` edges
- It will appear in `/neural-memory:neural-query` results alongside code nodes
- Bugs are also auto-imported from `.claude/context-log-gotchas.md` on every `/neural-memory:neural-index`
