"""Enables `python -m neural_memory` as an alternative server launch method."""

from neural_memory.server import mcp
import sys

if __name__ == "__main__":
    try:
        mcp.run()
    except Exception as exc:
        print(f"neural-memory MCP server failed: {exc}", file=sys.stderr)
        sys.exit(1)
