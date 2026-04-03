"""Tests for the docs fetcher registry."""
import pytest
from unittest.mock import MagicMock
from typing import Optional

from neural_memory.docs.registry import (
    register_fetcher,
    get_fetcher,
    fetch_docs,
    get_all_fetchers,
    _FETCHERS,
)
from neural_memory.docs.base import PackageDoc


def _make_doc(package_name: str, registry: str) -> PackageDoc:
    return PackageDoc(
        package_name=package_name,
        registry=registry,
        version="1.0.0",
        summary="Test summary",
        fetched_at="2024-01-01T00:00:00+00:00",
    )


class FakeFetcher:
    """A minimal fake fetcher for testing the registry."""

    def __init__(self, name: str, supported_prefix: str, doc: Optional[PackageDoc] = None):
        self._name = name
        self._prefix = supported_prefix
        self._doc = doc

    @property
    def registry_name(self) -> str:
        return self._name

    def supports(self, package_name: str) -> bool:
        return package_name.startswith(self._prefix)

    def fetch(self, package_name: str) -> Optional[PackageDoc]:
        return self._doc


class TestRegistryRoundtrip:
    def setup_method(self):
        # Save and clear the global registry state for isolation
        self._original = dict(_FETCHERS)
        _FETCHERS.clear()

    def teardown_method(self):
        _FETCHERS.clear()
        _FETCHERS.update(self._original)

    def test_register_and_get_fetcher(self):
        fetcher = FakeFetcher("test_reg", "test-")
        register_fetcher(fetcher)
        assert get_fetcher("test_reg") is fetcher

    def test_get_unknown_fetcher_returns_none(self):
        assert get_fetcher("nonexistent") is None

    def test_fetch_docs_explicit_registry(self):
        doc = _make_doc("mypkg", "test_reg")
        fetcher = FakeFetcher("test_reg", "my", doc=doc)
        register_fetcher(fetcher)

        result = fetch_docs("mypkg", registry="test_reg")
        assert result is doc

    def test_fetch_docs_explicit_registry_unknown_returns_none(self):
        result = fetch_docs("mypkg", registry="does_not_exist")
        assert result is None

    def test_fetch_docs_auto_detect_uses_supports(self):
        doc_a = _make_doc("alpha-pkg", "reg_a")
        doc_b = _make_doc("beta-pkg", "reg_b")

        fetcher_a = FakeFetcher("reg_a", "alpha-", doc=doc_a)
        fetcher_b = FakeFetcher("reg_b", "beta-", doc=doc_b)

        register_fetcher(fetcher_a)
        register_fetcher(fetcher_b)

        # alpha- prefix → reg_a
        result = fetch_docs("alpha-mything")
        assert result is doc_a

        # beta- prefix → reg_b
        result = fetch_docs("beta-thing")
        assert result is doc_b

    def test_fetch_docs_no_match_returns_none(self):
        fetcher = FakeFetcher("reg_x", "xonly-", doc=_make_doc("xonly-pkg", "reg_x"))
        register_fetcher(fetcher)

        result = fetch_docs("completely-unmatched-pkg")
        assert result is None

    def test_fetch_docs_fetcher_returns_none_skips_to_next(self):
        """If fetcher.supports() returns True but fetch() returns None, try next."""
        failing = FakeFetcher("reg_first", "pkg-", doc=None)
        succeeding_doc = _make_doc("pkg-foo", "reg_second")
        succeeding = FakeFetcher("reg_second", "pkg-", doc=succeeding_doc)

        register_fetcher(failing)
        register_fetcher(succeeding)

        result = fetch_docs("pkg-foo")
        assert result is succeeding_doc

    def test_get_all_fetchers(self):
        f1 = FakeFetcher("r1", "a")
        f2 = FakeFetcher("r2", "b")
        register_fetcher(f1)
        register_fetcher(f2)

        all_f = get_all_fetchers()
        assert f1 in all_f
        assert f2 in all_f
