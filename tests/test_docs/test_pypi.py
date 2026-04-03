"""Tests for PyPI documentation fetcher."""
import json
import urllib.error
from unittest.mock import patch, MagicMock

import pytest

from neural_memory.docs.pypi import PyPIFetcher


def _make_response(data: dict):
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(data).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


PYPI_PAYLOAD = {
    "info": {
        "name": "requests",
        "version": "2.31.0",
        "summary": "Python HTTP for Humans.",
        "description": "Requests is a simple HTTP library.",
        "home_page": "https://requests.readthedocs.io",
        "project_urls": {
            "Homepage": "https://requests.readthedocs.io",
            "Documentation": "https://docs.python-requests.org",
        },
    }
}


class TestPyPIFetcher:
    def setup_method(self):
        self.fetcher = PyPIFetcher()

    def test_registry_name(self):
        assert self.fetcher.registry_name == "pypi"

    def test_supports_simple_package(self):
        assert self.fetcher.supports("requests") is True

    def test_supports_rejects_scoped(self):
        assert self.fetcher.supports("@types/react") is False

    def test_supports_rejects_slash(self):
        assert self.fetcher.supports("github.com/some/pkg") is False

    def test_fetch_success(self):
        with patch("urllib.request.urlopen", return_value=_make_response(PYPI_PAYLOAD)):
            doc = self.fetcher.fetch("requests")

        assert doc is not None
        assert doc.package_name == "requests"
        assert doc.registry == "pypi"
        assert doc.version == "2.31.0"
        assert doc.summary == "Python HTTP for Humans."
        assert "Requests is a simple" in doc.description
        assert doc.homepage_url == "https://requests.readthedocs.io"
        assert doc.doc_url == "https://docs.python-requests.org"
        assert doc.fetched_at != ""

    def test_fetch_404_returns_none(self):
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.HTTPError(
                url="", code=404, msg="Not Found", hdrs=None, fp=None
            ),
        ):
            doc = self.fetcher.fetch("nonexistent-pkg-xyz")

        assert doc is None

    def test_fetch_network_error_returns_none(self):
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("network unreachable"),
        ):
            doc = self.fetcher.fetch("requests")

        assert doc is None

    def test_fetch_empty_info_does_not_crash(self):
        payload = {"info": {}}
        with patch("urllib.request.urlopen", return_value=_make_response(payload)):
            doc = self.fetcher.fetch("some-pkg")

        assert doc is not None
        assert doc.version == ""
        assert doc.summary == ""
        assert doc.description == ""

    def test_fetch_missing_project_urls_does_not_crash(self):
        payload = {
            "info": {
                "version": "1.0",
                "summary": "A package",
                "home_page": "https://example.com",
                "project_urls": None,
            }
        }
        with patch("urllib.request.urlopen", return_value=_make_response(payload)):
            doc = self.fetcher.fetch("some-pkg")

        assert doc is not None
        assert doc.homepage_url == "https://example.com"
        assert doc.doc_url == ""

    def test_description_truncated_at_5000(self):
        long_desc = "x" * 10000
        payload = {"info": {"description": long_desc, "version": "1.0"}}
        with patch("urllib.request.urlopen", return_value=_make_response(payload)):
            doc = self.fetcher.fetch("some-pkg")

        assert doc is not None
        assert len(doc.description) == 5000
