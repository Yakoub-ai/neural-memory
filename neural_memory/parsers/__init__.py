"""Multi-language parser system for neural memory.

Parsers are auto-registered at import time. A missing grammar package causes
that language to be silently skipped — Python always works.
"""

from __future__ import annotations

from .registry import register_parser
from .python_parser import PythonParser

# Python is always available
register_parser(PythonParser())

# Tree-sitter languages — skip if grammar not installed
try:
    from .treesitter_parser import TreeSitterParser
    from .languages.typescript import get_typescript_config, get_javascript_config

    register_parser(TreeSitterParser(get_typescript_config()))
    # .tsx is handled by the TypeScript tsx grammar
    try:
        from .languages.typescript import get_tsx_config
        # Re-register .tsx with the tsx-specific grammar
        _tsx = TreeSitterParser(get_tsx_config())
        from .registry import _PARSERS
        _PARSERS[".tsx"] = _tsx
    except Exception:
        pass

    register_parser(TreeSitterParser(get_javascript_config()))
except Exception:
    pass

try:
    from .treesitter_parser import TreeSitterParser
    from .languages.go import get_go_config
    register_parser(TreeSitterParser(get_go_config()))
except Exception:
    pass

try:
    from .treesitter_parser import TreeSitterParser
    from .languages.rust import get_rust_config
    register_parser(TreeSitterParser(get_rust_config()))
except Exception:
    pass

try:
    from .languages.sql import SQLParser
    register_parser(SQLParser())
except Exception:
    pass

__all__ = ["register_parser", "get_parser", "get_all_supported_extensions"]

from .registry import get_parser, get_all_supported_extensions
