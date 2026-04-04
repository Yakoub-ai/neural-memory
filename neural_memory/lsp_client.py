"""Minimal Language Server Protocol client over subprocess stdio.

Spawns a Pyright or pylsp language server, sends JSON-RPC requests
(hover, diagnostics), and returns enrichment data for neural nodes.

All operations are best-effort: a timeout or crash returns None/empty
and never propagates an exception to the indexer.

Usage:
    with LSPClient(project_root) as lsp:
        type_info = lsp.hover("neural_memory/storage.py", line=86, col=8)
        diags     = lsp.diagnostics("neural_memory/storage.py")
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

# ── Availability ───────────────────────────────────────────────────────────────

def is_lsp_available(language_id: str = "python") -> bool:
    """Return True if any LSP server for the given language is on PATH."""
    from .languages import get_lsp_server
    return get_lsp_server(language_id) is not None


def _detect_server(language_id: str = "python") -> Optional[str]:
    """Return the first available LSP server binary for a language."""
    from .languages import get_lsp_server
    return get_lsp_server(language_id)


# ── JSON-RPC framing ───────────────────────────────────────────────────────────

def _encode(msg: dict) -> bytes:
    body = json.dumps(msg).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    return header + body


def _decode_one(data: bytes) -> Optional[dict]:
    """Try to parse one JSON-RPC message from a bytes buffer.

    Returns the parsed dict, or None if the buffer is incomplete/invalid.
    """
    try:
        header_end = data.find(b"\r\n\r\n")
        if header_end == -1:
            return None
        header = data[:header_end].decode("ascii")
        length = 0
        for line in header.split("\r\n"):
            if line.lower().startswith("content-length:"):
                length = int(line.split(":", 1)[1].strip())
        body = data[header_end + 4: header_end + 4 + length]
        if len(body) < length:
            return None
        return json.loads(body.decode("utf-8"))
    except Exception:
        return None


# ── LSP Client ─────────────────────────────────────────────────────────────────

class LSPClient:
    """Minimal LSP client for hover and diagnostics queries.

    Designed for use as a context manager inside the indexer.
    """

    _TIMEOUT = 5.0     # seconds to wait for a response
    _INIT_TIMEOUT = 8.0

    def __init__(self, project_root: str, server: str = "auto", language_id: str = "python"):
        self._root = Path(project_root).resolve()
        self._language_id = language_id
        self._server = server if server not in ("auto", "") else (_detect_server(language_id) or "")
        self._proc: Optional[subprocess.Popen] = None
        self._msg_id = 0
        self._diagnostics: dict[str, list[str]] = {}
        self._reader_thread: Optional[threading.Thread] = None
        self._pending: dict[int, Optional[dict]] = {}
        self._lock = threading.Lock()
        self._buf = b""

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Spawn the LSP server and send initialize. Returns True on success."""
        if not self._server:
            return False
        try:
            args = [self._server]
            # Pyright needs --stdio flag
            if "pyright" in self._server:
                args.append("--stdio")

            self._proc = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                cwd=str(self._root),
            )
        except (FileNotFoundError, OSError):
            return False

        # Start background reader thread
        self._reader_thread = threading.Thread(target=self._reader, daemon=True)
        self._reader_thread.start()

        # Send initialize
        init_id = self._next_id()
        self._send({
            "jsonrpc": "2.0",
            "id": init_id,
            "method": "initialize",
            "params": {
                "processId": None,
                "rootUri": self._root.as_uri(),
                "capabilities": {
                    "textDocument": {
                        "hover": {"contentFormat": ["plaintext"]},
                        "publishDiagnostics": {},
                    }
                },
                "initializationOptions": {},
            },
        })

        # Wait for initialize response
        resp = self._wait_for(init_id, timeout=self._INIT_TIMEOUT)
        if resp is None:
            self.stop()
            return False

        # Send initialized notification
        self._send({"jsonrpc": "2.0", "method": "initialized", "params": {}})
        return True

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._send({"jsonrpc": "2.0", "id": self._next_id(), "method": "shutdown", "params": None})
                self._send({"jsonrpc": "2.0", "method": "exit", "params": None})
                self._proc.wait(timeout=2)
            except Exception:
                pass
            finally:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self._proc = None

    def __enter__(self) -> "LSPClient":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()

    # ── Queries ────────────────────────────────────────────────────────────────

    def hover(self, file_path: str, line: int, col: int = 0) -> Optional[str]:
        """Request hover info for a position. Returns plain-text content or None."""
        if not self._proc or self._proc.poll() is not None:
            return None
        try:
            uri = Path(self._root / file_path).as_uri()
            # Open document first
            source = (self._root / file_path).read_text(encoding="utf-8", errors="replace")
            self._send({
                "jsonrpc": "2.0",
                "method": "textDocument/didOpen",
                "params": {
                    "textDocument": {
                        "uri": uri,
                        "languageId": self._language_id,
                        "version": 1,
                        "text": source,
                    }
                },
            })

            req_id = self._next_id()
            self._send({
                "jsonrpc": "2.0",
                "id": req_id,
                "method": "textDocument/hover",
                "params": {
                    "textDocument": {"uri": uri},
                    "position": {"line": max(0, line - 1), "character": col},
                },
            })

            resp = self._wait_for(req_id)
            if resp and "result" in resp and resp["result"]:
                result = resp["result"]
                contents = result.get("contents", "")
                if isinstance(contents, dict):
                    return contents.get("value", "")
                if isinstance(contents, list) and contents:
                    first = contents[0]
                    return first.get("value", "") if isinstance(first, dict) else str(first)
                return str(contents) if contents else None
        except Exception:
            pass
        return None

    def diagnostics(self, file_path: str) -> list[str]:
        """Return cached diagnostics for the given file path."""
        uri = Path(self._root / file_path).as_uri()
        return self._diagnostics.get(uri, [])

    # ── Internal ───────────────────────────────────────────────────────────────

    def _next_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    def _send(self, msg: dict) -> None:
        if self._proc and self._proc.stdin and self._proc.poll() is None:
            try:
                self._proc.stdin.write(_encode(msg))
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass

    def _reader(self) -> None:
        """Background thread: read stdout and dispatch responses."""
        while self._proc and self._proc.poll() is None:
            try:
                chunk = self._proc.stdout.read(4096)
                if not chunk:
                    break
                self._buf += chunk
                self._try_parse()
            except Exception:
                break

    def _try_parse(self) -> None:
        while True:
            msg = _decode_one(self._buf)
            if msg is None:
                break
            # Consume the parsed bytes
            header_end = self._buf.find(b"\r\n\r\n")
            if header_end != -1:
                header = self._buf[:header_end].decode("ascii", errors="replace")
                length = 0
                for line in header.split("\r\n"):
                    if line.lower().startswith("content-length:"):
                        try:
                            length = int(line.split(":", 1)[1].strip())
                        except ValueError:
                            pass
                self._buf = self._buf[header_end + 4 + length:]

            # Dispatch
            if "id" in msg:
                with self._lock:
                    self._pending[msg["id"]] = msg
            elif msg.get("method") == "textDocument/publishDiagnostics":
                params = msg.get("params", {})
                uri = params.get("uri", "")
                diags = params.get("diagnostics", [])
                self._diagnostics[uri] = [
                    f"[{d.get('severity', '?')}] {d.get('message', '')}"
                    for d in diags
                ]

    def _wait_for(self, msg_id: int, timeout: float = _TIMEOUT) -> Optional[dict]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if msg_id in self._pending:
                    return self._pending.pop(msg_id)
            time.sleep(0.05)
        return None
