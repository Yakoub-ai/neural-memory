"""Tests for neural_memory.lsp_client — mock-based, no real LSP server needed."""

import json
import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from neural_memory.lsp_client import (
    LSPClient,
    is_lsp_available,
    _encode,
    _decode_one,
)


# ── JSON-RPC framing ──────────────────────────────────────────────────────────

class TestEncodeDecode:
    def test_encode_produces_header(self):
        msg = {"jsonrpc": "2.0", "id": 1, "method": "test"}
        data = _encode(msg)
        assert data.startswith(b"Content-Length:")
        assert b"\r\n\r\n" in data

    def test_encode_content_length_correct(self):
        msg = {"jsonrpc": "2.0", "id": 1, "method": "test", "params": {}}
        data = _encode(msg)
        header_end = data.find(b"\r\n\r\n")
        header = data[:header_end].decode("ascii")
        length = int(header.split("Content-Length:")[1].strip())
        body = data[header_end + 4:]
        assert len(body) == length

    def test_round_trip(self):
        msg = {"jsonrpc": "2.0", "id": 42, "result": {"value": "hello"}}
        data = _encode(msg)
        decoded = _decode_one(data)
        assert decoded == msg

    def test_decode_incomplete(self):
        # Only partial data → returns None
        partial = b"Content-Length: 100\r\n\r\n{}"
        result = _decode_one(partial)
        assert result is None

    def test_decode_no_header(self):
        result = _decode_one(b'{"no": "header"}')
        assert result is None

    def test_decode_empty(self):
        result = _decode_one(b"")
        assert result is None


# ── is_lsp_available ──────────────────────────────────────────────────────────

class TestIsLspAvailable:
    def test_returns_bool(self):
        result = is_lsp_available()
        assert isinstance(result, bool)

    def test_false_when_no_servers_on_path(self):
        with patch("neural_memory.lsp_client.shutil.which", return_value=None):
            assert is_lsp_available() is False

    def test_true_when_server_found(self):
        with patch("neural_memory.lsp_client.shutil.which", return_value="/usr/bin/pylsp"):
            assert is_lsp_available() is True


# ── LSPClient with mocked subprocess ─────────────────────────────────────────

def _make_init_response(req_id: int) -> bytes:
    """Build a valid LSP initialize response."""
    msg = {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "capabilities": {
                "hoverProvider": True,
                "textDocumentSync": 1,
            }
        }
    }
    return _encode(msg)


def _make_hover_response(req_id: int, text: str) -> bytes:
    msg = {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "contents": {"kind": "plaintext", "value": text}
        }
    }
    return _encode(msg)


class TestLSPClientStart:
    def test_start_false_when_no_server(self):
        client = LSPClient("/tmp", server="")
        result = client.start()
        assert result is False
        assert client._proc is None

    def test_start_false_when_popen_fails(self):
        with patch("neural_memory.lsp_client.subprocess.Popen", side_effect=FileNotFoundError):
            client = LSPClient("/tmp", server="nonexistent-lsp")
            result = client.start()
            assert result is False

    def test_context_manager_calls_stop(self):
        client = LSPClient("/tmp", server="")
        with patch.object(client, "stop") as mock_stop:
            with client:
                pass
            mock_stop.assert_called_once()


class TestLSPClientWithMockedProcess:
    """Tests that mock subprocess I/O to simulate LSP responses."""

    def _build_client_with_mock_proc(self, responses: list[bytes]) -> LSPClient:
        """Return an LSPClient whose subprocess produces the given byte responses."""
        client = LSPClient("/tmp", server="mock-lsp")
        client._server = "mock-lsp"

        # Build a mock Popen process
        proc = MagicMock()
        proc.poll.return_value = None
        proc.stdin = MagicMock()

        # Simulate stdout that returns responses in sequence
        buf = b"".join(responses)
        proc.stdout.read.side_effect = [buf, b""]

        client._proc = proc
        # Start reader thread manually
        client._reader_thread = threading.Thread(target=client._reader, daemon=True)
        client._reader_thread.start()
        return client

    def test_hover_returns_text(self, tmp_path):
        root = str(tmp_path)
        client = LSPClient(root, server="mock-lsp")
        # pre-populate id=3 hover response
        hover_resp = _make_hover_response(3, "def greet(name: str) -> str")
        parsed_hover = _decode_one(hover_resp)
        with client._lock:
            client._pending[parsed_hover["id"]] = parsed_hover

        proc = MagicMock()
        proc.poll.return_value = None
        proc.stdin = MagicMock()
        client._proc = proc

        # hover will send didOpen (notification, no id) then hover request (id=3)
        client._msg_id = 2  # next call to _next_id() returns 3

        (tmp_path / "test.py").write_text("def greet(): pass")
        with patch.object(client, "_send"):
            result = client.hover("test.py", line=1)

        assert result == "def greet(name: str) -> str"

    def test_hover_returns_none_on_timeout(self, tmp_path):
        root = str(tmp_path)
        client = LSPClient(root, server="mock-lsp")
        proc = MagicMock()
        proc.poll.return_value = None
        proc.stdin = MagicMock()
        client._proc = proc
        client._msg_id = 10
        client._TIMEOUT = 0.05  # very short to keep test fast

        (tmp_path / "test.py").write_text("x = 1")
        with patch.object(client, "_send"):
            result = client.hover("test.py", line=1)
        assert result is None

    def test_diagnostics_empty_by_default(self, tmp_path):
        client = LSPClient(str(tmp_path), server="mock-lsp")
        result = client.diagnostics("any/file.py")
        assert result == []

    def test_diagnostics_after_publish(self, tmp_path):
        client = LSPClient(str(tmp_path), server="mock-lsp")
        uri = (tmp_path / "test.py").as_uri()
        client._diagnostics[uri] = ["[1] undefined name 'foo'"]
        result = client.diagnostics("test.py")
        assert result == ["[1] undefined name 'foo'"]

    def test_stop_safe_when_not_started(self, tmp_path):
        client = LSPClient(str(tmp_path), server="")
        # Should not raise
        client.stop()
        assert client._proc is None


class TestLSPDecodeMultiMessage:
    """Test that _try_parse correctly handles concatenated messages."""

    def test_parse_two_messages(self, tmp_path):
        client = LSPClient(str(tmp_path))
        msg1 = _encode({"jsonrpc": "2.0", "id": 1, "result": "a"})
        msg2 = _encode({"jsonrpc": "2.0", "id": 2, "result": "b"})
        client._buf = msg1 + msg2

        proc = MagicMock()
        proc.poll.return_value = None
        client._proc = proc
        client._try_parse()

        with client._lock:
            assert 1 in client._pending
            assert client._pending[1]["result"] == "a"
            assert 2 in client._pending
            assert client._pending[2]["result"] == "b"

    def test_parse_diagnostics_notification(self, tmp_path):
        client = LSPClient(str(tmp_path))
        uri = (tmp_path / "foo.py").as_uri()
        notif = _encode({
            "jsonrpc": "2.0",
            "method": "textDocument/publishDiagnostics",
            "params": {
                "uri": uri,
                "diagnostics": [{"severity": 1, "message": "unused import"}]
            }
        })
        client._buf = notif
        proc = MagicMock()
        proc.poll.return_value = None
        client._proc = proc
        client._try_parse()

        assert uri in client._diagnostics
        assert "unused import" in client._diagnostics[uri][0]
