"""Language registry for neural memory multi-language support.

Central source of truth mapping file extensions → language metadata.
All other components (parser, LSP, config, server) import from here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class LanguageSpec:
    """Metadata for a supported programming language."""
    id: str                              # "python", "typescript", "rust", etc.
    name: str                            # "Python", "TypeScript", "Rust", etc.
    extensions: tuple[str, ...]          # (".py",), (".ts", ".tsx"), etc.
    tree_sitter_package: str             # import name: "tree_sitter_python"
    lsp_servers: list[str]               # in preference order: ["pyright-langserver", "pylsp"]
    lsp_language_id: str                 # LSP protocol languageId string
    code_fence: str                      # markdown code fence identifier
    project_markers: tuple[str, ...]     # filenames that signal this language is in use
    extra_exclude: tuple[str, ...]       # language-specific additional exclude patterns
    grammar_fn: str = "language"         # function name within the package: mod.language()
    include_patterns: list[str] = field(default_factory=list)  # computed from extensions

    def __post_init__(self) -> None:
        if not self.include_patterns:
            if len(self.extensions) == 1:
                self.include_patterns = [f"**/*{self.extensions[0]}"]
            else:
                # e.g. "**/*.{ts,tsx}"
                exts = ",".join(e.lstrip(".") for e in self.extensions)
                self.include_patterns = [f"**/*.{{{exts}}}"]


# ── Language definitions ───────────────────────────────────────────────────────

LANGUAGES: dict[str, LanguageSpec] = {
    "python": LanguageSpec(
        id="python",
        name="Python",
        extensions=(".py",),
        tree_sitter_package="tree_sitter_python",
        lsp_servers=["pyright-langserver", "pylsp", "pyright"],
        lsp_language_id="python",
        code_fence="python",
        project_markers=("pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "Pipfile", "poetry.lock"),
        extra_exclude=("**/__pycache__/**", "**/*.egg-info/**", "**/.venv/**", "**/venv/**"),
    ),
    "javascript": LanguageSpec(
        id="javascript",
        name="JavaScript",
        extensions=(".js", ".jsx", ".mjs", ".cjs"),
        tree_sitter_package="tree_sitter_javascript",
        lsp_servers=["typescript-language-server"],
        lsp_language_id="javascript",
        code_fence="javascript",
        project_markers=("package.json",),
        extra_exclude=("**/node_modules/**", "**/dist/**", "**/build/**", "**/.next/**", "**/coverage/**"),
    ),
    "typescript": LanguageSpec(
        id="typescript",
        name="TypeScript",
        extensions=(".ts", ".tsx"),
        tree_sitter_package="tree_sitter_typescript",
        grammar_fn="language_typescript",
        lsp_servers=["typescript-language-server"],
        lsp_language_id="typescript",
        code_fence="typescript",
        project_markers=("tsconfig.json", "tsconfig.base.json"),
        extra_exclude=("**/node_modules/**", "**/dist/**", "**/build/**", "**/.next/**", "**/coverage/**"),
    ),
    "rust": LanguageSpec(
        id="rust",
        name="Rust",
        extensions=(".rs",),
        tree_sitter_package="tree_sitter_rust",
        lsp_servers=["rust-analyzer"],
        lsp_language_id="rust",
        code_fence="rust",
        project_markers=("Cargo.toml",),
        extra_exclude=("**/target/**", "**/.cargo/**"),
    ),
    "go": LanguageSpec(
        id="go",
        name="Go",
        extensions=(".go",),
        tree_sitter_package="tree_sitter_go",
        lsp_servers=["gopls"],
        lsp_language_id="go",
        code_fence="go",
        project_markers=("go.mod",),
        extra_exclude=("**/vendor/**", "**/pkg/**"),
    ),
    "sql": LanguageSpec(
        id="sql",
        name="SQL",
        extensions=(".sql",),
        tree_sitter_package="tree_sitter_sql",
        lsp_servers=["sql-language-server"],
        lsp_language_id="sql",
        code_fence="sql",
        project_markers=(),
        extra_exclude=(),
    ),
    "ruby": LanguageSpec(
        id="ruby",
        name="Ruby",
        extensions=(".rb", ".rake"),
        tree_sitter_package="tree_sitter_ruby",
        lsp_servers=["solargraph"],
        lsp_language_id="ruby",
        code_fence="ruby",
        project_markers=("Gemfile", "Gemfile.lock"),
        extra_exclude=("**/vendor/bundle/**",),
    ),
    "php": LanguageSpec(
        id="php",
        name="PHP",
        extensions=(".php",),
        tree_sitter_package="tree_sitter_php",
        grammar_fn="language_php",
        lsp_servers=["phpactor", "intelephense"],
        lsp_language_id="php",
        code_fence="php",
        project_markers=("composer.json", "composer.lock"),
        extra_exclude=("**/vendor/**",),
    ),
}

# Extension → LanguageSpec lookup for O(1) detection
_EXT_INDEX: dict[str, LanguageSpec] = {
    ext: spec
    for spec in LANGUAGES.values()
    for ext in spec.extensions
}


# ── Public helpers ─────────────────────────────────────────────────────────────

def detect_language(file_path: str) -> Optional[LanguageSpec]:
    """Detect a file's language from its extension. Returns None if unsupported."""
    suffix = Path(file_path).suffix.lower()
    return _EXT_INDEX.get(suffix)


def auto_detect_languages(project_root: str | Path) -> list[LanguageSpec]:
    """Scan project root for language markers and file extensions.

    Returns a deduplicated list of detected LanguageSpecs, ordered by
    confidence (marker presence > file count).
    """
    root = Path(project_root)
    detected: dict[str, LanguageSpec] = {}

    # Pass 1: check for project marker files (high confidence)
    for spec in LANGUAGES.values():
        for marker in spec.project_markers:
            if (root / marker).exists():
                detected[spec.id] = spec
                break

    # Pass 2: scan for source files (lower confidence — catches mixed repos
    # and languages without dedicated marker files like SQL)
    try:
        # Walk up to 3 directory levels to avoid full traversal on large repos
        for dirpath, dirnames, filenames in os.walk(root):
            # Skip common noise directories
            dirnames[:] = [
                d for d in dirnames
                if d not in {".git", "node_modules", "__pycache__", ".venv", "venv",
                             "target", "vendor", ".next", "dist", "build", "coverage"}
            ]
            depth = len(Path(dirpath).relative_to(root).parts)
            if depth > 4:
                dirnames.clear()
                continue
            for fname in filenames:
                suffix = Path(fname).suffix.lower()
                spec = _EXT_INDEX.get(suffix)
                if spec and spec.id not in detected:
                    detected[spec.id] = spec
    except PermissionError:
        pass

    return list(detected.values())


def get_all_include_patterns(language_ids: list[str]) -> list[str]:
    """Return glob include patterns for the given language IDs."""
    patterns: list[str] = []
    for lid in language_ids:
        spec = LANGUAGES.get(lid)
        if spec:
            patterns.extend(spec.include_patterns)
    return patterns


def get_all_exclude_patterns(language_ids: list[str]) -> list[str]:
    """Return extra exclude patterns for the given language IDs."""
    patterns: list[str] = []
    seen: set[str] = set()
    for lid in language_ids:
        spec = LANGUAGES.get(lid)
        if spec:
            for p in spec.extra_exclude:
                if p not in seen:
                    patterns.append(p)
                    seen.add(p)
    return patterns


def supported_extensions() -> set[str]:
    """Return the set of all file extensions supported across all languages."""
    return set(_EXT_INDEX.keys())


def get_lsp_server(language_id: str) -> Optional[str]:
    """Return the first available LSP server binary for a language, or None."""
    import shutil
    spec = LANGUAGES.get(language_id)
    if not spec:
        return None
    for server in spec.lsp_servers:
        if shutil.which(server):
            return server
    return None
