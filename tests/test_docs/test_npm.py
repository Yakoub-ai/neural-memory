"""Tests for npm registry documentation fetcher."""
import json
import urllib.error
from unittest.mock import patch, MagicMock, call

import pytest

from neural_memory.docs.npm import NpmFetcher


def _make_response(data: dict):
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(data).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


NPM_PAYLOAD = {
    "name": "lodash",
    "description": "Lodash modular utilities.",
    "homepage": "https://lodash.com",
    "readme": "# Lodash\nA modern JavaScript utility library.",
    "dist-tags": {"latest": "4.17.21"},
}

SCOPED_PAYLOAD = {
    "name": "@types/react",
    "description": "TypeScript definitions for React.",
    "homepage": "",
    "readme": "",
    "dist-tags": {"latest": "18.2.0"},
}


class TestNpmFetcher:
    def setup_method(self):
        self.fetcher = NpmFetcher()

    def test_registry_name(self):
        assert self.fetcher.registry_name == "npm"

    def test_supports_simple_package(self):
        assert self.fetcher.supports("lodash") is True

    def test_supports_scoped_package(self):
        assert self.fetcher.supports("@types/react") is True

    def test_supports_rejects_go_path(self):
        assert self.fetcher.supports("github.com/user/repo") is False

    def test_fetch_simple_package_success(self):
        with patch("urllib.request.urlopen", return_value=_make_response(NPM_PAYLOAD)):
            doc = self.fetcher.fetch("lodash")

        assert doc is not None
        assert doc.package_name == "lodash"
        assert doc.registry == "npm"
        assert doc.version == "4.17.21"
        assert "Lodash" in doc.summary
        assert doc.homepage_url == "https://lodash.com"
        assert doc.fetched_at != ""

    def test_fetch_scoped_package_encodes_slash(self):
        """Scoped package @types/react must encode / as %2F in the URL."""
        captured_urls = []

        def fake_urlopen(url, timeout=None):
            captured_urls.append(url)
            return _make_response(SCOPED_PAYLOAD)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            doc = self.fetcher.fetch("@types/react")

        assert len(captured_urls) == 1
        assert "%2F" in captured_urls[0]
        assert "@types/react" not in captured_urls[0]  # raw slash must not appear
        assert doc is not None
        assert doc.version == "18.2.0"

    def test_fetch_404_returns_none(self):
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.HTTPError(
                url="", code=404, msg="Not Found", hdrs=None, fp=None
            ),
        ):
            doc = self.fetcher.fetch("nonexistent-npm-pkg-xyz")

        assert doc is None

    def test_fetch_network_error_returns_none(self):
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("timeout"),
        ):
            doc = self.fetcher.fetch("lodash")

        assert doc is None

    def test_fetch_combined_description_and_readme(self):
        payload = {
            "description": "Short desc.",
            "readme": "Long readme content here.",
            "dist-tags": {"latest": "1.0.0"},
        }
        with patch("urllib.request.urlopen", return_value=_make_response(payload)):
            doc = self.fetcher.fetch("some-pkg")

        assert doc is not None
        assert "Short desc." in doc.description
        assert "Long readme content here." in doc.description
