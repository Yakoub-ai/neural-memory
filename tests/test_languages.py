"""Tests for neural_memory.languages — language detection and pattern helpers."""

import pytest
from pathlib import Path

from neural_memory.languages import (
    LANGUAGES,
    LanguageSpec,
    detect_language,
    auto_detect_languages,
    get_all_include_patterns,
    get_all_exclude_patterns,
    supported_extensions,
    get_lsp_server,
)


# ── LanguageSpec dataclass ─────────────────────────────────────────────────────

def test_language_spec_single_extension_include_pattern():
    """Single extension produces a simple glob pattern."""
    spec = LANGUAGES["python"]
    assert spec.include_patterns == ["**/*.py"]


def test_language_spec_multiple_extensions_include_pattern():
    """Multiple extensions produce a brace-expansion glob pattern."""
    spec = LANGUAGES["typescript"]
    # expects something like **/*.{ts,tsx}
    assert len(spec.include_patterns) == 1
    pattern = spec.include_patterns[0]
    assert pattern.startswith("**/*.")
    assert "ts" in pattern
    assert "tsx" in pattern


def test_language_spec_javascript_include_pattern():
    """JavaScript has 4 extensions; pattern uses brace expansion."""
    spec = LANGUAGES["javascript"]
    assert len(spec.include_patterns) == 1
    pattern = spec.include_patterns[0]
    assert "js" in pattern
    assert "jsx" in pattern
    assert "mjs" in pattern


def test_language_spec_fields_populated():
    """Core fields are present and non-empty for every registered language."""
    for lang_id, spec in LANGUAGES.items():
        assert spec.id == lang_id
        assert spec.name
        assert len(spec.extensions) >= 1
        assert spec.tree_sitter_package
        assert spec.lsp_language_id
        assert spec.code_fence
        assert spec.grammar_fn  # default is "language", may be overridden


# ── detect_language ────────────────────────────────────────────────────────────

def test_detect_language_python():
    spec = detect_language("mymodule.py")
    assert spec is not None
    assert spec.id == "python"


def test_detect_language_typescript():
    spec = detect_language("app.ts")
    assert spec is not None
    assert spec.id == "typescript"


def test_detect_language_tsx():
    spec = detect_language("component.tsx")
    assert spec is not None
    assert spec.id == "typescript"


def test_detect_language_javascript():
    spec = detect_language("index.js")
    assert spec is not None
    assert spec.id == "javascript"


def test_detect_language_jsx():
    spec = detect_language("Button.jsx")
    assert spec is not None
    assert spec.id == "javascript"


def test_detect_language_go():
    spec = detect_language("main.go")
    assert spec is not None
    assert spec.id == "go"


def test_detect_language_rust():
    spec = detect_language("lib.rs")
    assert spec is not None
    assert spec.id == "rust"


def test_detect_language_ruby():
    spec = detect_language("schema.rb")
    assert spec is not None
    assert spec.id == "ruby"


def test_detect_language_php():
    spec = detect_language("index.php")
    assert spec is not None
    assert spec.id == "php"


def test_detect_language_sql():
    spec = detect_language("migration.sql")
    assert spec is not None
    assert spec.id == "sql"


def test_detect_language_unknown_extension_returns_none():
    assert detect_language("readme.txt") is None
    assert detect_language("image.png") is None
    assert detect_language("archive.zip") is None


def test_detect_language_no_extension_returns_none():
    assert detect_language("Makefile") is None
    assert detect_language("Dockerfile") is None


def test_detect_language_uppercase_extension_normalised():
    """detect_language lowercases the suffix before lookup.

    Actual behaviour: .PY → .py → returns python spec.
    """
    spec = detect_language("SCRIPT.PY")
    assert spec is not None
    assert spec.id == "python"


def test_detect_language_full_path():
    """Works with a full absolute-style path, not just a bare filename."""
    spec = detect_language("/home/user/project/src/main.rs")
    assert spec is not None
    assert spec.id == "rust"


def test_detect_language_returns_language_spec_instance():
    spec = detect_language("app.py")
    assert isinstance(spec, LanguageSpec)


# ── auto_detect_languages ──────────────────────────────────────────────────────

def test_auto_detect_python_via_pyproject(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[build-system]\n")
    results = auto_detect_languages(tmp_path)
    ids = {s.id for s in results}
    assert "python" in ids


def test_auto_detect_python_via_setup_py(tmp_path):
    (tmp_path / "setup.py").write_text("from setuptools import setup\n")
    results = auto_detect_languages(tmp_path)
    ids = {s.id for s in results}
    assert "python" in ids


def test_auto_detect_javascript_via_package_json(tmp_path):
    (tmp_path / "package.json").write_text('{"name": "my-app"}\n')
    results = auto_detect_languages(tmp_path)
    ids = {s.id for s in results}
    assert "javascript" in ids


def test_auto_detect_typescript_via_tsconfig(tmp_path):
    (tmp_path / "tsconfig.json").write_text('{"compilerOptions": {}}\n')
    results = auto_detect_languages(tmp_path)
    ids = {s.id for s in results}
    assert "typescript" in ids


def test_auto_detect_rust_via_cargo_toml(tmp_path):
    (tmp_path / "Cargo.toml").write_text('[package]\nname = "my-crate"\n')
    results = auto_detect_languages(tmp_path)
    ids = {s.id for s in results}
    assert "rust" in ids


def test_auto_detect_go_via_go_mod(tmp_path):
    (tmp_path / "go.mod").write_text("module example.com/myapp\n\ngo 1.21\n")
    results = auto_detect_languages(tmp_path)
    ids = {s.id for s in results}
    assert "go" in ids


def test_auto_detect_ruby_via_gemfile(tmp_path):
    (tmp_path / "Gemfile").write_text("source 'https://rubygems.org'\n")
    results = auto_detect_languages(tmp_path)
    ids = {s.id for s in results}
    assert "ruby" in ids


def test_auto_detect_php_via_composer_json(tmp_path):
    (tmp_path / "composer.json").write_text('{"require": {}}\n')
    results = auto_detect_languages(tmp_path)
    ids = {s.id for s in results}
    assert "php" in ids


def test_auto_detect_sql_via_file_extension(tmp_path):
    """SQL has no project markers, so it is detected via file scan."""
    (tmp_path / "migration.sql").write_text("CREATE TABLE foo (id INT);\n")
    results = auto_detect_languages(tmp_path)
    ids = {s.id for s in results}
    assert "sql" in ids


def test_auto_detect_empty_directory(tmp_path):
    """An empty directory returns an empty list (no markers, no source files)."""
    results = auto_detect_languages(tmp_path)
    assert isinstance(results, list)
    assert len(results) == 0


def test_auto_detect_returns_list_of_language_specs(tmp_path):
    (tmp_path / "pyproject.toml").touch()
    results = auto_detect_languages(tmp_path)
    assert all(isinstance(s, LanguageSpec) for s in results)


def test_auto_detect_deduplicates(tmp_path):
    """Both a marker file and a source file are present — no duplicates."""
    (tmp_path / "pyproject.toml").touch()
    (tmp_path / "main.py").write_text("print('hello')\n")
    results = auto_detect_languages(tmp_path)
    ids = [s.id for s in results]
    assert ids.count("python") == 1


def test_auto_detect_source_files_in_subdirectory(tmp_path):
    """Source files nested up to 3 levels are still detected."""
    subdir = tmp_path / "src" / "lib"
    subdir.mkdir(parents=True)
    (subdir / "module.go").write_text("package main\n")
    results = auto_detect_languages(tmp_path)
    ids = {s.id for s in results}
    assert "go" in ids


# ── get_all_include_patterns ───────────────────────────────────────────────────

def test_get_all_include_patterns_nonempty():
    patterns = get_all_include_patterns(list(LANGUAGES.keys()))
    assert len(patterns) > 0


def test_get_all_include_patterns_contains_python():
    patterns = get_all_include_patterns(["python"])
    assert "**/*.py" in patterns


def test_get_all_include_patterns_contains_typescript_glob():
    patterns = get_all_include_patterns(["typescript"])
    assert any("ts" in p for p in patterns)


def test_get_all_include_patterns_unknown_language_ignored():
    """Unknown language IDs produce no error and no patterns."""
    patterns = get_all_include_patterns(["nonexistent_lang"])
    assert patterns == []


def test_get_all_include_patterns_subset():
    """Requesting a subset returns only patterns for those languages."""
    patterns = get_all_include_patterns(["python", "rust"])
    assert "**/*.py" in patterns
    assert "**/*.rs" in patterns
    # TypeScript patterns should not be included
    assert not any("tsx" in p for p in patterns)


# ── get_all_exclude_patterns ───────────────────────────────────────────────────

def test_get_all_exclude_patterns_nonempty():
    patterns = get_all_exclude_patterns(list(LANGUAGES.keys()))
    assert len(patterns) > 0


def test_get_all_exclude_patterns_python_includes_pycache():
    patterns = get_all_exclude_patterns(["python"])
    assert any("__pycache__" in p for p in patterns)


def test_get_all_exclude_patterns_javascript_includes_node_modules():
    patterns = get_all_exclude_patterns(["javascript"])
    assert any("node_modules" in p for p in patterns)


def test_get_all_exclude_patterns_deduplicates():
    """Both javascript and typescript share node_modules — deduplicated."""
    patterns = get_all_exclude_patterns(["javascript", "typescript"])
    node_modules_count = sum(1 for p in patterns if "node_modules" in p)
    assert node_modules_count == 1


def test_get_all_exclude_patterns_unknown_language_ignored():
    patterns = get_all_exclude_patterns(["bogus_lang"])
    assert patterns == []


# ── supported_extensions ───────────────────────────────────────────────────────

def test_supported_extensions_returns_set():
    exts = supported_extensions()
    assert isinstance(exts, set)


def test_supported_extensions_contains_common():
    exts = supported_extensions()
    assert ".py" in exts
    assert ".ts" in exts
    assert ".rs" in exts
    assert ".go" in exts
    assert ".js" in exts
    assert ".sql" in exts


def test_supported_extensions_all_lowercase():
    """All returned extensions are lowercase (matching detect_language's normalisation)."""
    exts = supported_extensions()
    for ext in exts:
        assert ext == ext.lower(), f"Extension {ext!r} is not lowercase"


# ── get_lsp_server ─────────────────────────────────────────────────────────────

def test_get_lsp_server_unknown_language_returns_none():
    result = get_lsp_server("not_a_real_language")
    assert result is None


def test_get_lsp_server_returns_string_or_none_for_known_language():
    """Known languages return either a server name string or None (if not installed)."""
    result = get_lsp_server("python")
    assert result is None or isinstance(result, str)


def test_get_lsp_server_result_is_in_lsp_servers_list():
    """If a server is found, it must be one from the language's lsp_servers list."""
    result = get_lsp_server("python")
    if result is not None:
        assert result in LANGUAGES["python"].lsp_servers


def test_get_lsp_server_all_known_languages_do_not_raise():
    """Calling get_lsp_server for every registered language never raises."""
    for lang_id in LANGUAGES:
        result = get_lsp_server(lang_id)
        assert result is None or isinstance(result, str)
