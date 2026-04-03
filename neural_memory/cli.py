"""neural-memory-setup — installation helper for Neural Memory."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
import sqlite3
import sys
from pathlib import Path


HOOK_TEXT = """
## Neural Memory Agent Hook

On each invocation, check neural memory staleness by running the `neural_status` tool.
If the index is stale or uninitialized, inform the user and suggest the appropriate action.
"""

MCP_ENTRY = {
    "neural-memory": {
        "command": "neural-memory"
    }
}

MCP_ENTRY_FALLBACK = {
    "neural-memory": {
        "command": sys.executable,
        "args": ["-m", "neural_memory.server"]
    }
}


def _check_deps() -> list[str]:
    """Return list of problem strings (empty = all good)."""
    problems = []

    # Python version
    if sys.version_info < (3, 10):
        problems.append(f"Python 3.10+ required (found {platform.python_version()})")

    # sqlite3 (stdlib, but some minimal installs strip it)
    try:
        sqlite3.connect(":memory:").close()
    except Exception:
        problems.append("sqlite3 unavailable — reinstall Python with sqlite3 support")

    # mcp
    try:
        importlib.metadata.version("mcp")
    except importlib.metadata.PackageNotFoundError:
        problems.append("mcp package missing — run: pip install neural-memory")

    # pydantic
    try:
        importlib.metadata.version("pydantic")
    except importlib.metadata.PackageNotFoundError:
        problems.append("pydantic package missing — run: pip install neural-memory")

    return problems


def _neural_memory_on_path() -> bool:
    return shutil.which("neural-memory") is not None


def _settings_path(scope: str, project_root: Path) -> Path:
    if scope == "global":
        home = Path.home()
        return home / ".claude" / "settings.json"
    else:
        return project_root / ".claude" / "settings.json"


def _add_mcp_config(settings_path: Path, use_fallback: bool) -> None:
    entry = MCP_ENTRY_FALLBACK if use_fallback else MCP_ENTRY
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    data: dict = {}
    if settings_path.exists():
        try:
            data = json.loads(settings_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass  # overwrite corrupt file

    mcp_servers = data.setdefault("mcpServers", {})
    if "neural-memory" in mcp_servers:
        print("  (MCP server config already present — skipping)")
        return

    mcp_servers["neural-memory"] = entry["neural-memory"]
    settings_path.write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8"
    )


def _install_skills(target: Path) -> int:
    dest = target / ".claude" / "commands"
    dest.mkdir(parents=True, exist_ok=True)
    skills_dir = Path(__file__).parent.parent / "skills"
    copied = 0
    for skill_dir in skills_dir.iterdir():
        skill_md = skill_dir / "SKILL.md"
        if skill_md.exists():
            shutil.copy2(skill_md, dest / f"{skill_dir.name}.md")
            copied += 1
    return copied


def _append_hook(claude_md: Path) -> bool:
    marker = "Neural Memory Agent Hook"
    if claude_md.exists():
        if marker in claude_md.read_text(encoding="utf-8"):
            return False  # already present
        claude_md.write_text(
            claude_md.read_text(encoding="utf-8") + HOOK_TEXT,
            encoding="utf-8"
        )
    else:
        claude_md.write_text(HOOK_TEXT.lstrip(), encoding="utf-8")
    return True


def cmd_doctor() -> None:
    """Check environment health."""
    print("Neural Memory - Doctor\n")

    ok = True

    # Python
    v = platform.python_version()
    if sys.version_info >= (3, 10):
        print(f"  [OK] Python {v}")
    else:
        print(f"  [!!] Python {v} (need 3.10+)")
        ok = False

    # sqlite3
    try:
        sqlite3.connect(":memory:").close()
        print("  [OK] sqlite3")
    except Exception:
        print("  [!!] sqlite3 -- reinstall Python with sqlite3 support")
        ok = False

    # mcp
    try:
        ver = importlib.metadata.version("mcp")
        print(f"  [OK] mcp {ver}")
    except importlib.metadata.PackageNotFoundError:
        print("  [!!] mcp not installed -- run: pip install neural-memory")
        ok = False

    # pydantic
    try:
        ver = importlib.metadata.version("pydantic")
        print(f"  [OK] pydantic {ver}")
    except importlib.metadata.PackageNotFoundError:
        print("  [!!] pydantic not installed -- run: pip install neural-memory")
        ok = False

    # neural-memory command
    if _neural_memory_on_path():
        print("  [OK] neural-memory command on PATH")
    else:
        print("  [ !] neural-memory not on PATH (will use python -m fallback)")

    if ok:
        print("\nAll checks passed.")
    else:
        print("\nFix the issues above, then re-run: neural-memory-setup doctor")
        sys.exit(1)


def cmd_install() -> None:
    """Interactive full setup."""
    print("Neural Memory - Setup\n")

    # 1. Check deps
    print("Checking dependencies...")
    problems = _check_deps()
    if not _neural_memory_on_path():
        on_path = False
        print("  ! neural-memory not on PATH — will use python -m fallback in MCP config")
    else:
        on_path = True

    if problems:
        for p in problems:
            print(f"  [!!] {p}")
        print("\nFix the issues above before continuing.")
        sys.exit(1)
    else:
        print("  All dependencies OK\n")

    # 2. MCP scope
    print("Where should the MCP server be configured?")
    print("  [1] Global (~/.claude/settings.json) — works across all projects")
    print("  [2] Per-project (.claude/settings.json) — this project only")
    choice = input("  > ").strip()
    scope = "global" if choice != "2" else "project"
    project_root = Path.cwd()
    settings_path = _settings_path(scope, project_root)

    _add_mcp_config(settings_path, use_fallback=not on_path)
    print(f"  [OK] MCP server added to {settings_path}\n")

    # 3. Slash commands
    print("Install slash commands to current project?")
    print("  (copies 8 neural-memory commands to .claude/commands/)")
    if input("  [y/N] > ").strip().lower() == "y":
        n = _install_skills(project_root)
        print(f"  [OK] Installed {n} commands to .claude/commands/\n")
    else:
        print("  Skipped\n")

    # 4. CLAUDE.md hook
    claude_md = project_root / "CLAUDE.md"
    print("Add neural-memory agent hook to CLAUDE.md?")
    print("  (tells Claude to auto-check index staleness each session)")
    if input("  [y/N] > ").strip().lower() == "y":
        added = _append_hook(claude_md)
        if added:
            print(f"  [OK] Agent hook added to {claude_md}\n")
        else:
            print("  (hook already present)\n")
    else:
        print("  Skipped\n")

    print("Setup complete! Run /neural-index in Claude Code to build your knowledge graph.")


def cmd_install_commands() -> None:
    """Copy slash commands non-interactively."""
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    n = _install_skills(target)
    print(f"Installed {n} neural-memory commands to {target / '.claude' / 'commands'}")


def main() -> None:
    args = sys.argv[1:]
    cmd = args[0] if args else "install"

    if cmd == "install":
        cmd_install()
    elif cmd == "install-commands":
        sys.argv = sys.argv[1:]  # shift
        cmd_install_commands()
    elif cmd == "doctor":
        cmd_doctor()
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: neural-memory-setup [install|install-commands|doctor]")
        sys.exit(1)
