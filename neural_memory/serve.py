"""Lightweight local HTTP server for the neural memory dashboard.

Usage (programmatic):
    from neural_memory.serve import start_server, stop_server, is_running, get_url
    url = start_server(project_root=".", port=7891)
    stop_server()

Usage (CLI):
    neural-memory-viz [--port PORT] [--project-root DIR] [--no-browser]
"""

from __future__ import annotations

import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Optional

# ── Module-level server state ──────────────────────────────────────────────────

_server: Optional[HTTPServer] = None
_thread: Optional[threading.Thread] = None
_port: int = 0
_lock = threading.Lock()

DEFAULT_PORT = 7891


# ── Internal helpers ───────────────────────────────────────────────────────────

def _make_handler(directory: str):
    """Return a SimpleHTTPRequestHandler subclass anchored to `directory`."""
    class _Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)

        def log_message(self, fmt, *args):  # silence access log
            pass

        def log_error(self, fmt, *args):  # silence 404s etc.
            pass

    return _Handler


# ── Public API ─────────────────────────────────────────────────────────────────

def start_server(
    project_root: str = ".",
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
) -> str:
    """Start a local HTTP server serving `.neural-memory/` and return the URL.

    Idempotent: if a server is already running on the same port the existing
    URL is returned (browser is re-opened if ``open_browser`` is True).
    If a server is running on a *different* port it is stopped first.
    """
    global _server, _thread, _port

    with _lock:
        # Idempotent: already running on the right port
        if _server is not None and _port == port:
            url = f"http://localhost:{port}/dashboard.html"
            if open_browser:
                webbrowser.open(url)
            return url

        # Stop any existing server on a different port
        _stop_locked()

        memory_dir = str(Path(project_root).resolve() / ".neural-memory")
        Path(memory_dir).mkdir(parents=True, exist_ok=True)

        handler = _make_handler(memory_dir)
        srv = HTTPServer(("127.0.0.1", port), handler)

        t = threading.Thread(target=srv.serve_forever, daemon=True, name="neural-memory-viz")
        t.start()

        _server = srv
        _thread = t
        _port = port

    url = f"http://localhost:{port}/dashboard.html"
    if open_browser:
        webbrowser.open(url)
    return url


def stop_server() -> bool:
    """Stop the running dashboard server.

    Returns True if a server was running and has been stopped, False otherwise.
    """
    with _lock:
        return _stop_locked()


def _stop_locked() -> bool:
    """Internal: stop server; caller must hold _lock."""
    global _server, _thread, _port

    if _server is None:
        return False

    _server.shutdown()
    _server = None
    _thread = None
    _port = 0
    return True


def is_running() -> bool:
    """Return True if a dashboard server is currently running."""
    return _server is not None


def get_url() -> Optional[str]:
    """Return the dashboard URL if a server is running, otherwise None."""
    if _server is None:
        return None
    return f"http://localhost:{_port}/dashboard.html"


def get_port() -> Optional[int]:
    """Return the port the server is listening on, or None if not running."""
    return _port if _server is not None else None


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point: ``neural-memory-viz``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="neural-memory-viz",
        description="Serve the Neural Memory knowledge-graph dashboard in your browser.",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"Port to listen on (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--project-root", default=".",
        metavar="DIR",
        help="Project root that contains .neural-memory/ (default: current directory)",
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Don't open the browser automatically",
    )
    args = parser.parse_args()

    url = start_server(
        project_root=args.project_root,
        port=args.port,
        open_browser=not args.no_browser,
    )

    print(f"\nNeural Memory dashboard running at: {url}")
    print("Press Ctrl+C to stop.\n")

    try:
        # Keep the main thread alive; server thread is a daemon
        while True:
            threading.Event().wait(timeout=1.0)
    except KeyboardInterrupt:
        stop_server()
        print("\nServer stopped.")
