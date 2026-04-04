#!/usr/bin/env python3
"""Bump version across all version files in sync.

Usage:
    python scripts/bump_version.py 0.6.0

Updates:
    - pyproject.toml
    - neural_memory/__init__.py
    - .claude-plugin/plugin.json
    - .claude-plugin/marketplace.json
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def bump_pyproject(version: str) -> str:
    path = ROOT / "pyproject.toml"
    text = path.read_text(encoding="utf-8")
    new_text, count = re.subn(
        r'^(version\s*=\s*")[^"]+(")',
        rf"\g<1>{version}\g<2>",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if not count:
        raise ValueError("version line not found in pyproject.toml")
    path.write_text(new_text, encoding="utf-8")
    return str(path.relative_to(ROOT))


def bump_init(version: str) -> str:
    path = ROOT / "neural_memory" / "__init__.py"
    text = path.read_text(encoding="utf-8")
    new_text, count = re.subn(
        r'(__version__\s*=\s*")[^"]+(")',
        rf"\g<1>{version}\g<2>",
        text,
        count=1,
    )
    if not count:
        raise ValueError("__version__ line not found in neural_memory/__init__.py")
    path.write_text(new_text, encoding="utf-8")
    return str(path.relative_to(ROOT))


def bump_plugin_json(version: str) -> str:
    path = ROOT / ".claude-plugin" / "plugin.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["version"] = version
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return str(path.relative_to(ROOT))


def bump_marketplace_json(version: str) -> str:
    path = ROOT / ".claude-plugin" / "marketplace.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for plugin in data.get("plugins", []):
        plugin["version"] = version
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return str(path.relative_to(ROOT))


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <version>", file=sys.stderr)
        print("Example: python scripts/bump_version.py 0.6.0", file=sys.stderr)
        sys.exit(1)

    version = sys.argv[1].lstrip("v")

    # Basic semver validation
    parts = version.split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        print(f"Error: version must be X.Y.Z (got '{version}')", file=sys.stderr)
        sys.exit(1)

    updated = []
    try:
        updated.append(bump_pyproject(version))
        updated.append(bump_init(version))
        updated.append(bump_plugin_json(version))
        updated.append(bump_marketplace_json(version))
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Bumped to v{version}:")
    for f in updated:
        print(f"  {f}")


if __name__ == "__main__":
    main()
