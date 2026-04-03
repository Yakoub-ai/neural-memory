"""Tests for neural_memory.serve — local HTTP server."""

import time
import urllib.request

import pytest

from neural_memory.serve import (
    start_server,
    stop_server,
    is_running,
    get_url,
    get_port,
    DEFAULT_PORT,
)


@pytest.fixture(autouse=True)
def cleanup():
    """Ensure no server is left running after each test."""
    yield
    stop_server()


class TestServerLifecycle:
    def test_not_running_initially(self):
        assert not is_running()
        assert get_url() is None
        assert get_port() is None

    def test_start_returns_url(self, tmp_path):
        url = start_server(project_root=str(tmp_path), port=18791, open_browser=False)
        assert url == "http://localhost:18791/dashboard.html"

    def test_is_running_after_start(self, tmp_path):
        start_server(project_root=str(tmp_path), port=18791, open_browser=False)
        assert is_running()

    def test_get_url_after_start(self, tmp_path):
        start_server(project_root=str(tmp_path), port=18791, open_browser=False)
        assert get_url() == "http://localhost:18791/dashboard.html"

    def test_get_port_after_start(self, tmp_path):
        start_server(project_root=str(tmp_path), port=18791, open_browser=False)
        assert get_port() == 18791

    def test_stop_returns_true_when_running(self, tmp_path):
        start_server(project_root=str(tmp_path), port=18791, open_browser=False)
        assert stop_server() is True

    def test_stop_returns_false_when_not_running(self):
        assert stop_server() is False

    def test_not_running_after_stop(self, tmp_path):
        start_server(project_root=str(tmp_path), port=18791, open_browser=False)
        stop_server()
        assert not is_running()
        assert get_url() is None

    def test_idempotent_start_same_port(self, tmp_path):
        url1 = start_server(project_root=str(tmp_path), port=18791, open_browser=False)
        url2 = start_server(project_root=str(tmp_path), port=18791, open_browser=False)
        assert url1 == url2
        assert is_running()

    def test_serves_files(self, tmp_path):
        # Write a test file into .neural-memory/
        memory_dir = tmp_path / ".neural-memory"
        memory_dir.mkdir()
        (memory_dir / "hello.txt").write_text("hello world", encoding="utf-8")

        start_server(project_root=str(tmp_path), port=18791, open_browser=False)
        time.sleep(0.1)  # brief settle

        resp = urllib.request.urlopen("http://localhost:18791/hello.txt", timeout=3)
        assert resp.read().decode() == "hello world"

    def test_creates_memory_dir_if_missing(self, tmp_path):
        memory_dir = tmp_path / ".neural-memory"
        assert not memory_dir.exists()
        start_server(project_root=str(tmp_path), port=18791, open_browser=False)
        assert memory_dir.exists()
