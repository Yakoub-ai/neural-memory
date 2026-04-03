# Neural Memory — Incremental Update

Sync neural memory with recent code changes without a full re-index.

## What this does
1. Checks git diff since last indexed commit
2. Compares file hashes to detect modified files
3. Re-parses only changed/added files
4. Removes nodes for deleted files
5. Re-resolves cross-file edges
6. Recomputes importance scores

## How to call

**Via MCP tool** (neural-memory configured as MCP server in Claude Code):
```json
Tool: neural_update
{}
```

**Via Python** (working directly in the project):
```python
import asyncio
from neural_memory.server import neural_update, UpdateInput

asyncio.run(neural_update(UpdateInput()))
```

`UpdateInput` has no required fields. Pass `project_root="."` to be explicit.

## When to use
- After pulling new changes
- After a coding session with multiple file edits
- When `/neural-status` reports staleness

Much faster than a full index — only touches changed files.
