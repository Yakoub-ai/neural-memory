"""Tests for plugin manifest validity, path integrity, and version consistency.

These tests catch issues that would prevent successful installation or update
of the neural-memory plugin — things that only surface at install time if
not caught here first.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
PLUGIN_JSON = ROOT / ".claude-plugin" / "plugin.json"
MARKETPLACE_JSON = ROOT / ".claude-plugin" / "marketplace.json"
PYPROJECT = ROOT / "pyproject.toml"
INIT_PY = ROOT / "neural_memory" / "__init__.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_plugin_json() -> dict:
    return json.loads(PLUGIN_JSON.read_text(encoding="utf-8"))


def _load_marketplace_json() -> dict:
    return json.loads(MARKETPLACE_JSON.read_text(encoding="utf-8"))


def _pyproject_version() -> str:
    text = PYPROJECT.read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert m, "version not found in pyproject.toml"
    return m.group(1)


def _init_version() -> str:
    text = INIT_PY.read_text(encoding="utf-8")
    m = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    assert m, "__version__ not found in neural_memory/__init__.py"
    return m.group(1)


# ---------------------------------------------------------------------------
# Manifest structure
# ---------------------------------------------------------------------------

class TestPluginJsonStructure:
    def test_is_valid_json(self):
        """plugin.json must be parseable JSON."""
        data = _load_plugin_json()
        assert isinstance(data, dict)

    def test_required_name_field(self):
        data = _load_plugin_json()
        assert "name" in data
        assert isinstance(data["name"], str)
        assert data["name"] == "neural-memory"

    def test_version_is_semver_string(self):
        data = _load_plugin_json()
        assert "version" in data
        parts = data["version"].split(".")
        assert len(parts) == 3, "version must be MAJOR.MINOR.PATCH"
        assert all(p.isdigit() for p in parts)

    def test_agents_field_must_not_be_plain_string(self):
        """Regression: 'agents' as a string path fails validator with 'Invalid input'.

        The plugin validator expects agents to be absent (auto-discovered) or
        an array of paths, not a bare string like './agents'.
        """
        data = _load_plugin_json()
        if "agents" in data:
            assert not isinstance(data["agents"], str), (
                "agents field must be an array or absent -- "
                "string paths fail manifest validation with 'agents: Invalid input'"
            )

    def test_hooks_path_if_present_is_string(self):
        data = _load_plugin_json()
        if "hooks" in data:
            assert isinstance(data["hooks"], str)

    def test_mcp_servers_path_if_present_is_string(self):
        data = _load_plugin_json()
        if "mcpServers" in data:
            assert isinstance(data["mcpServers"], str)


# ---------------------------------------------------------------------------
# Path integrity
# ---------------------------------------------------------------------------

class TestPluginJsonPaths:
    def test_hooks_path_exists(self):
        data = _load_plugin_json()
        if "hooks" not in data:
            pytest.skip("no hooks field in plugin.json")
        hooks_path = ROOT / data["hooks"]
        assert hooks_path.exists(), f"hooks path missing: {hooks_path}"

    def test_mcp_servers_file_exists(self):
        data = _load_plugin_json()
        if "mcpServers" not in data:
            pytest.skip("no mcpServers field in plugin.json")
        mcp_path = ROOT / data["mcpServers"]
        assert mcp_path.exists(), f"mcpServers path missing: {mcp_path}"

    def test_agents_dir_exists_at_root(self):
        """The agents/ directory should exist at the plugin root for auto-discovery."""
        agents_dir = ROOT / "agents"
        assert agents_dir.exists() and agents_dir.is_dir(), (
            "agents/ directory missing -- bundled agents will not be discoverable"
        )

    def test_agents_dir_contains_md_files(self):
        agents_dir = ROOT / "agents"
        md_files = list(agents_dir.glob("*.md"))
        assert len(md_files) > 0, "agents/ directory is empty -- no agent definitions found"

    def test_hooks_json_is_valid(self):
        data = _load_plugin_json()
        if "hooks" not in data:
            pytest.skip("no hooks field in plugin.json")
        hooks_path = ROOT / data["hooks"]
        hooks_data = json.loads(hooks_path.read_text(encoding="utf-8"))
        assert isinstance(hooks_data, dict)


# ---------------------------------------------------------------------------
# Version consistency across all version files
# ---------------------------------------------------------------------------

class TestVersionConsistency:
    """All version strings must match. Version drift causes confusing install failures."""

    def test_plugin_json_matches_pyproject(self):
        plugin_ver = _load_plugin_json()["version"]
        pyproject_ver = _pyproject_version()
        assert plugin_ver == pyproject_ver, (
            f"plugin.json ({plugin_ver}) != pyproject.toml ({pyproject_ver})"
        )

    def test_init_py_matches_pyproject(self):
        init_ver = _init_version()
        pyproject_ver = _pyproject_version()
        assert init_ver == pyproject_ver, (
            f"__init__.py ({init_ver}) != pyproject.toml ({pyproject_ver})"
        )

    def test_marketplace_json_matches_pyproject(self):
        marketplace = _load_marketplace_json()
        plugin_entry = next(
            (p for p in marketplace.get("plugins", []) if p["name"] == "neural-memory"),
            None,
        )
        assert plugin_entry is not None, "neural-memory not found in marketplace.json plugins list"
        market_ver = plugin_entry["version"]
        pyproject_ver = _pyproject_version()
        assert market_ver == pyproject_ver, (
            f"marketplace.json ({market_ver}) != pyproject.toml ({pyproject_ver})"
        )


# ---------------------------------------------------------------------------
# bump_version.py smoke test
# ---------------------------------------------------------------------------

def _import_bump_version():
    """Import bump_version as a module via importlib (no exec/shell)."""
    import importlib.util
    script_path = ROOT / "scripts" / "bump_version.py"
    assert script_path.exists(), f"bump_version.py not found at {script_path}"
    spec = importlib.util.spec_from_file_location("bump_version", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestBumpVersionScript:
    def test_bump_version_script_exits_cleanly(self):
        """bump_version.py must accept the current version and exit 0 (idempotent re-stamp)."""
        current = _pyproject_version()
        script_path = ROOT / "scripts" / "bump_version.py"
        assert script_path.is_file(), "bump_version.py missing"
        result = subprocess.run(
            [sys.executable, str(script_path), current],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        assert result.returncode == 0, (
            f"bump_version.py exited {result.returncode}:\n{result.stderr}"
        )

    def test_bump_version_rejects_invalid_version(self):
        """bump_version.py must exit non-zero for malformed version strings."""
        script_path = ROOT / "scripts" / "bump_version.py"
        result = subprocess.run(
            [sys.executable, str(script_path), "not-a-version"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        assert result.returncode != 0, "bump_version.py should reject 'not-a-version'"

    def test_bump_version_updates_all_files(self, tmp_path):
        """All four version files must be updated in sync by bump_version.py."""
        import shutil

        # Mirror the relevant files into tmp_path
        plugin_dir = tmp_path / ".claude-plugin"
        plugin_dir.mkdir()
        shutil.copy(PLUGIN_JSON, plugin_dir / "plugin.json")
        shutil.copy(MARKETPLACE_JSON, plugin_dir / "marketplace.json")
        nm_dir = tmp_path / "neural_memory"
        nm_dir.mkdir()
        shutil.copy(INIT_PY, nm_dir / "__init__.py")
        shutil.copy(PYPROJECT, tmp_path / "pyproject.toml")

        # Import the module and redirect ROOT to tmp_path via monkeypatch
        mod = _import_bump_version()
        original_root = mod.ROOT
        try:
            mod.ROOT = tmp_path
            mod.bump_pyproject("99.0.0")
            mod.bump_init("99.0.0")
            mod.bump_plugin_json("99.0.0")
            mod.bump_marketplace_json("99.0.0")
        finally:
            mod.ROOT = original_root  # always restore

        pj = json.loads((plugin_dir / "plugin.json").read_text())
        assert pj["version"] == "99.0.0", "plugin.json version not updated"

        mj = json.loads((plugin_dir / "marketplace.json").read_text())
        entry = next(p for p in mj["plugins"] if p["name"] == "neural-memory")
        assert entry["version"] == "99.0.0", "marketplace.json version not updated"

        pyp = (tmp_path / "pyproject.toml").read_text()
        assert re.search(r'version\s*=\s*"99\.0\.0"', pyp), "pyproject.toml version not updated"

        init = (nm_dir / "__init__.py").read_text()
        assert re.search(r'__version__\s*=\s*"99\.0\.0"', init), "__init__.py version not updated"
