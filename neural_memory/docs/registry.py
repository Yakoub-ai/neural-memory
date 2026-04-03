"""Fetcher registry — maps registry names to fetcher instances."""
from __future__ import annotations
from typing import Optional
from .base import PackageDoc, DocFetcher

_FETCHERS: dict[str, DocFetcher] = {}


def register_fetcher(fetcher: DocFetcher) -> None:
    _FETCHERS[fetcher.registry_name] = fetcher


def get_fetcher(registry_name: str) -> Optional[DocFetcher]:
    return _FETCHERS.get(registry_name)


def fetch_docs(package_name: str, registry: Optional[str] = None) -> Optional[PackageDoc]:
    """Fetch docs for a package. If registry given, use that fetcher. Otherwise try all."""
    if registry:
        fetcher = _FETCHERS.get(registry)
        return fetcher.fetch(package_name) if fetcher else None
    for fetcher in _FETCHERS.values():
        if fetcher.supports(package_name):
            result = fetcher.fetch(package_name)
            if result:
                return result
    return None


def get_all_fetchers() -> list[DocFetcher]:
    return list(_FETCHERS.values())
