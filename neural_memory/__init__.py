"""Neural Memory — A knowledge graph for codebases, built for Claude Code."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("neural-memory-mcp")
except PackageNotFoundError:
    __version__ = "0.6.6"  # fallback when running from source without install
