---
name: neural-stop
description: Stop the running neural memory dashboard server.
---

# Neural Memory — Stop Dashboard

Stop the running neural memory dashboard server.

## Usage

Call the `neural_stop_serve` tool:
```
neural_stop_serve()
```

Returns a confirmation with the URL that was stopped, or a message if no server was running.

## Notes
- The server is a lightweight local HTTP server (Python stdlib, no extra dependencies)
- It only serves files from `.neural-memory/` — no external access
- The server is automatically stopped when the Claude Code session ends
