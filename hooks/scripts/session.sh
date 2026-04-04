#!/bin/bash
# Neural Memory — Stop hook
# Archives completed tasks/bugs and auto-updates stale index on session end.
#
# Resolution order:
#   1. neural-session-hook on PATH (pip install or pipx)
#   2. uvx --from neural-memory-mcp neural-session-hook (marketplace/uvx install)
#
# Silently exits if unavailable — never breaks the user's workflow.

set -euo pipefail

if command -v neural-session-hook &>/dev/null 2>&1; then
    exec neural-session-hook
elif command -v uvx &>/dev/null 2>&1; then
    exec uvx --quiet --from neural-memory-mcp neural-session-hook
fi
# Neither available: silent exit
