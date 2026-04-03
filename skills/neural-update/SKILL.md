---
name: neural-update
description: Sync neural memory with recent code changes incrementally without a full re-index.
---

# Neural Memory — Incremental Update

Sync neural memory with recent code changes without a full re-index.

## What this does
1. Checks git diff since last indexed commit
2. Compares file hashes to detect modified files
3. Re-parses only changed/added files
4. Removes nodes for deleted files
5. Re-resolves cross-file edges
6. Recomputes importance scores

## When to use
- After pulling new changes
- After a coding session with multiple file edits
- When `/neural-status` reports staleness

Much faster than a full index — only touches changed files.

Run the `neural_update` tool to execute.
