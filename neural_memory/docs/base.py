"""Base types for documentation fetching."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable


@dataclass
class PackageDoc:
    package_name: str
    registry: str
    version: str = ""
    summary: str = ""
    description: str = ""
    homepage_url: str = ""
    doc_url: str = ""
    fetched_at: str = ""

    def to_storage_dict(self) -> dict:
        return {
            "version": self.version,
            "summary": self.summary,
            "description": self.description,
            "homepage_url": self.homepage_url,
            "doc_url": self.doc_url,
        }


@runtime_checkable
class DocFetcher(Protocol):
    @property
    def registry_name(self) -> str: ...
    def fetch(self, package_name: str) -> Optional[PackageDoc]: ...
    def supports(self, package_name: str) -> bool: ...
