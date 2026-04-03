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
    # File patterns
    include_patterns: list[str] = field(default_factory=lambda: ["**/*.py"])
    exclude_patterns: list[str] = field(default_factory=lambda: [
        "**/__pycache__/**", "**/node_modules/**", "**/.venv/**",
        "**/venv/**", "**/.git/**", "**/dist/**", "**/build/**",
        "**/*.egg-info/**", "**/migrations/**",
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
    lsp_enabled: bool = True        # attempt LSP enrichment during indexing
    lsp_server: str = "auto"        # "auto" | "pyright-langserver" | "pylsp" | "none"

    def __post_init__(self):
        if isinstance(self.index_mode, str):
            self.index_mode = IndexMode(self.index_mode)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["index_mode"] = self.index_mode.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> NeuralConfig:
        data["index_mode"] = IndexMode(data.get("index_mode", "both"))
        if "redaction" in data and isinstance(data["redaction"], dict):
            data["redaction"] = RedactionConfig(**data["redaction"])
        return cls(**data)


def get_memory_dir(project_root: str = ".") -> Path:
    """Get or create the .neural-memory directory."""
    p = Path(project_root) / NEURAL_MEMORY_DIR
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_config(project_root: str = ".") -> NeuralConfig:
    """Load config from disk or return defaults."""
    config_path = get_memory_dir(project_root) / CONFIG_FILE
    if config_path.exists():
        with open(config_path, "r") as f:
            data = json.load(f)
        data["project_root"] = project_root
        return NeuralConfig.from_dict(data)
    cfg = NeuralConfig(project_root=project_root)
    save_config(cfg)
    return cfg


def save_config(config: NeuralConfig) -> None:
    """Persist config to disk."""
    config_path = get_memory_dir(config.project_root) / CONFIG_FILE
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
