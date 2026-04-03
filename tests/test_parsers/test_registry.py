"""Tests for the parser registry."""

import pytest
from neural_memory.parsers.registry import get_parser, get_all_supported_extensions, register_parser


def test_python_parser_registered():
    p = get_parser("foo.py")
    assert p is not None
    assert p.language_id == "python"


def test_unknown_extension_returns_none():
    assert get_parser("foo.xyz123") is None
    assert get_parser("foo.pdf") is None


def test_all_supported_extensions_includes_python():
    exts = get_all_supported_extensions()
    assert ".py" in exts


def test_typescript_registered():
    p = get_parser("component.ts")
    assert p is not None
    assert p.language_id == "typescript"


def test_tsx_registered():
    p = get_parser("App.tsx")
    assert p is not None
    assert p.language_id == "typescript"


def test_javascript_registered():
    p = get_parser("app.js")
    assert p is not None
    assert p.language_id == "javascript"


def test_go_registered():
    p = get_parser("main.go")
    assert p is not None
    assert p.language_id == "go"


def test_rust_registered():
    p = get_parser("lib.rs")
    assert p is not None
    assert p.language_id == "rust"


def test_extension_case_insensitive():
    assert get_parser("FOO.PY") is not None
    assert get_parser("FILE.TS") is not None
