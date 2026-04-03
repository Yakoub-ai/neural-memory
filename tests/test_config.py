"""Tests for neural_memory.config — configuration management."""

import json
import pytest
from pathlib import Path

from neural_memory.config import (
    NeuralConfig,
    RedactionConfig,
    load_config,
    save_config,
    get_memory_dir,
    NEURAL_MEMORY_DIR,
    CONFIG_FILE,
)
from neural_memory.models import IndexMode


def test_get_memory_dir_creates_directory(tmp_path):
    mem_dir = get_memory_dir(str(tmp_path))
    assert mem_dir.exists()
    assert mem_dir.is_dir()
    assert mem_dir.name == NEURAL_MEMORY_DIR


def test_get_memory_dir_idempotent(tmp_path):
    get_memory_dir(str(tmp_path))
    get_memory_dir(str(tmp_path))  # No error on second call
    assert (tmp_path / NEURAL_MEMORY_DIR).exists()


def test_default_config_values(tmp_path):
    cfg = NeuralConfig(project_root=str(tmp_path))
    assert cfg.index_mode == IndexMode.BOTH
    assert "**/*.py" in cfg.include_patterns
    assert cfg.redaction.enabled is True
    assert cfg.redaction.builtin_patterns is True
    assert cfg.staleness_threshold == 5


def test_default_exclude_patterns(tmp_path):
    cfg = NeuralConfig(project_root=str(tmp_path))
    excludes = " ".join(cfg.exclude_patterns)
    assert "__pycache__" in excludes
    assert "node_modules" in excludes
    assert ".venv" in excludes


def test_load_config_creates_file_on_first_call(tmp_path):
    cfg = load_config(str(tmp_path))
    config_file = tmp_path / NEURAL_MEMORY_DIR / CONFIG_FILE
    assert config_file.exists()
    assert cfg.project_root == str(tmp_path)


def test_save_and_load_config_roundtrip(tmp_path):
    cfg = NeuralConfig(
        project_root=str(tmp_path),
        index_mode=IndexMode.AST_ONLY,
        staleness_threshold=10,
    )
    save_config(cfg)
    loaded = load_config(str(tmp_path))
    assert loaded.index_mode == IndexMode.AST_ONLY
    assert loaded.staleness_threshold == 10


def test_neural_config_from_dict_string_index_mode(tmp_path):
    data = {
        "project_root": str(tmp_path),
        "index_mode": "ast_only",
        "include_patterns": ["**/*.py"],
        "exclude_patterns": [],
        "redaction": {
            "enabled": True,
            "builtin_patterns": True,
            "sensitive_var_patterns": [],
            "custom_patterns": [],
            "whitelist_vars": [],
        },
        "max_summary_short_chars": 200,
        "max_summary_detailed_chars": 1000,
        "staleness_threshold": 5,
        "importance_threshold": 0.2,
    }
    cfg = NeuralConfig.from_dict(data)
    assert cfg.index_mode == IndexMode.AST_ONLY


def test_config_to_dict_index_mode_is_string(tmp_path):
    cfg = NeuralConfig(project_root=str(tmp_path), index_mode=IndexMode.AST_ONLY)
    d = cfg.to_dict()
    assert isinstance(d["index_mode"], str)
    assert d["index_mode"] == "ast_only"


def test_redaction_config_defaults():
    rc = RedactionConfig()
    assert rc.enabled is True
    assert rc.builtin_patterns is True
    assert rc.custom_patterns == []
    assert rc.whitelist_vars == []
    assert len(rc.sensitive_var_patterns) >= 1
