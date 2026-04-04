#!/bin/bash
# Neural Memory — UserPromptSubmit hook
# Injects a compact context snapshot (~500 tokens) before each user prompt.
#
# Resolution order:
#   1. neural-context-hook on PATH (pip install or pipx)
#   2. uvx --from neural-memory-mcp neural-context-hook (marketplace/uvx install)
#
# Silently exits if unavailable — never breaks the user's workflow.

set -euo pipefail

if command -v neural-context-hook &>/dev/null 2>&1; then
    exec neural-context-hook
elif command -v uvx &>/dev/null 2>&1; then
    exec uvx --quiet --from neural-memory-mcp neural-context-hook
fi
# Neither available: silent exit (hook is optional, not critical)
