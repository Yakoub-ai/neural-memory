"""Configuration management for neural memory."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from .models import IndexMode

NEURAL_MEMORY_DIR = ".neural-memory"
CONFIG_FILE = "config.json"
DB_FILE = "memory.db"

# The old Python-only include pattern (pre-multi-language). Used for auto-migration.
_LEGACY_PYTHON_ONLY = ["**/*.py"]


@dataclass
class RedactionConfig:
    """Redaction rules for sensitive content."""
    enabled: bool = True
    # Regex patterns for secrets detection
    builtin_patterns: bool = True
    # AST-aware variable name patterns to redact values of
    sensitive_var_patterns: list[str] = field(default_factory=lambda: [
        r"(?i)(secret|password|passwd|pwd|token|api_key|apikey|auth|credential|private_key|access_key)"
    ])
    # User-defined custom patterns
    custom_patterns: list[str] = field(default_factory=list)
    # Whitelist for false positive variable names
    whitelist_vars: list[str] = field(default_factory=list)


@dataclass
class NeuralConfig:
    """Top-level configuration."""
    # Project root (auto-detected)
    project_root: str = "."
    # Indexing mode
    index_mode: IndexMode = IndexMode.BOTH
    # Language selection — "auto" detects from project markers and file extensions.
    # Or specify explicitly: ["python", "typescript", "rust"]
    languages: list[str] = field(default_factory=lambda: ["auto"])
    # File patterns — ["auto"] resolves from the languages list at index time.
    # Can also specify manually: ["**/*.py", "**/*.ts"]
    include_patterns: list[str] = field(default_factory=lambda: ["auto"])
    exclude_patterns: list[str] = field(default_factory=lambda: [
        # Language-agnostic noise
        "**/.git/**", "**/migrations/**",
        # Neural memory storage directory
        "**/.neural-memory/**",
        # Python
        "**/__pycache__/**", "**/*.egg-info/**", "**/.venv/**", "**/venv/**",
        # JavaScript/TypeScript
        "**/node_modules/**", "**/dist/**", "**/build/**", "**/.next/**", "**/coverage/**",
        # Rust
        "**/target/**", "**/.cargo/**",
        # Go
        "**/vendor/**", "**/pkg/**",
        # General build artifacts
        "**/bin/**", "**/obj/**", "**/Pods/**",
    ])
    # Redaction
    redaction: RedactionConfig = field(default_factory=RedactionConfig)
    # Summary settings
    max_summary_short_chars: int = 200
    max_summary_detailed_chars: int = 1000
    # Staleness threshold (commits behind before suggesting update)
    staleness_threshold: int = 5
    # Importance threshold — nodes below this are typed as OTHER
    importance_threshold: float = 0.2
    # LSP enrichment settings
    lsp_enabled: bool = True
    # Per-language LSP server overrides. Empty dict = auto-detect.
    # e.g. {"python": "pyright-langserver", "rust": "rust-analyzer"}
    lsp_servers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.index_mode, str):
            self.index_mode = IndexMode(self.index_mode)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["index_mode"] = self.index_mode.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> NeuralConfig:
        import dataclasses
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        filtered["index_mode"] = IndexMode(filtered.get("index_mode", "both"))
        if "redaction" in filtered and isinstance(filtered["redaction"], dict):
            filtered["redaction"] = RedactionConfig(**filtered["redaction"])
        # Backward compat: migrate old lsp_server string to lsp_servers dict
        if "lsp_server" in data and "lsp_servers" not in data:
            old = data["lsp_server"]
            if old and old not in ("auto", "none"):
                filtered["lsp_servers"] = {"python": old}
        # Backward compat: old include_patterns defaulting to ["**/*.py"]
        if "include_patterns" in filtered and filtered["include_patterns"] == ["**/*.py"]:
            filtered["include_patterns"] = ["auto"]
        return cls(**filtered)


def get_memory_dir(project_root: str = ".") -> Path:
    """Get or create the .neural-memory directory."""
    p = Path(project_root) / NEURAL_MEMORY_DIR
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_include_patterns(config: "NeuralConfig") -> list[str]:
    """Expand 'auto' in include_patterns to detected language patterns.

    If the patterns list contains 'auto', auto-detects languages from the
    project root and substitutes the appropriate glob patterns.
    """
    if "auto" not in config.include_patterns:
        return config.include_patterns

    from .languages import auto_detect_languages, get_all_include_patterns, get_all_exclude_patterns

    languages = config.languages
    if "auto" in languages:
        detected = auto_detect_languages(config.project_root)
        language_ids = [spec.id for spec in detected]
        if not language_ids:
            language_ids = ["python"]  # fallback
    else:
        language_ids = languages

    patterns = get_all_include_patterns(language_ids)
    if not patterns:
        patterns = ["**/*.py"]

    return patterns


def load_config(project_root: str = ".") -> NeuralConfig:
    """Load config from disk or return defaults.

    If an existing config only has the legacy Python-only include pattern,
    it is automatically expanded to the full multi-language set.
    """
    config_path = get_memory_dir(project_root) / CONFIG_FILE
    if config_path.exists():
        with open(config_path, "r") as f:
            data = json.load(f)
        data["project_root"] = project_root
        cfg = NeuralConfig.from_dict(data)
        # Auto-migrate legacy Python-only configs
        if cfg.include_patterns == _LEGACY_PYTHON_ONLY:
            cfg.include_patterns = NeuralConfig.__dataclass_fields__[
                "include_patterns"
            ].default_factory()
            save_config(cfg)
        return cfg
    cfg = NeuralConfig(project_root=project_root)
    save_config(cfg)
    return cfg


def save_config(config: NeuralConfig) -> None:
    """Persist config to disk."""
    config_path = get_memory_dir(config.project_root) / CONFIG_FILE
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
