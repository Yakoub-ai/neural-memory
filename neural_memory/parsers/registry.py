"""Parser registry — dispatches file paths to the correct LanguageParser."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import LanguageParser

_PARSERS: dict[str, LanguageParser] = {}  # extension (with dot) -> parser


def register_parser(parser: LanguageParser) -> None:
    """Register a parser for its declared file extensions."""
    for ext in parser.file_extensions:
        _PARSERS[ext.lower()] = parser


def get_parser(file_path: str) -> Optional[LanguageParser]:
    """Return the parser for this file, or None if unsupported."""
    ext = Path(file_path).suffix.lower()
    return _PARSERS.get(ext)


def get_all_supported_extensions() -> set[str]:
    """Return all extensions that have a registered parser."""
    return set(_PARSERS.keys())
