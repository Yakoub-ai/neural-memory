"""Universal multi-language parser using tree-sitter.

Provides the same interface as parser.py's parse_file / resolve_edges but
handles all supported languages via tree-sitter grammars and S-expression
query files in the queries/ directory.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from pathlib import Path
from typing import Optional

from .languages import LanguageSpec, detect_language
from .models import NeuralNode, NeuralEdge, NodeType, EdgeType

logger = logging.getLogger(__name__)

_QUERIES_DIR = Path(__file__).parent / "queries"
_UNRESOLVED = "__unresolved__"

# Branch constructs per language for complexity estimation
_COMPLEXITY_NODE_TYPES: dict[str, set[str]] = {
    "python": {"if_statement", "elif_clause", "for_statement", "while_statement",
               "except_clause", "with_statement", "boolean_operator", "conditional_expression"},
    "javascript": {"if_statement", "else_clause", "for_statement", "for_in_statement",
                   "while_statement", "do_statement", "switch_case", "catch_clause",
                   "ternary_expression", "logical_expression"},
    "typescript": {"if_statement", "else_clause", "for_statement", "for_in_statement",
                   "while_statement", "do_statement", "switch_case", "catch_clause",
                   "ternary_expression", "logical_expression"},
    "rust": {"if_expression", "else_clause", "for_expression", "while_expression",
             "loop_expression", "match_arm", "if_let_expression", "while_let_expression"},
    "go": {"if_statement", "else_clause", "for_statement", "select_statement",
           "case_clause", "type_switch_statement"},
    "ruby": {"if", "elsif", "unless", "while", "until", "for", "when", "rescue"},
    "php": {"if_statement", "else_clause", "elseif_clause", "for_statement",
            "foreach_statement", "while_statement", "do_statement", "switch_statement",
            "case_statement", "catch_clause"},
    "sql": set(),  # SQL complexity not applicable
}

# Doc comment patterns per language
_DOC_COMMENT_RE: dict[str, re.Pattern] = {
    "python": re.compile(r'^\s*(?:"""([\s\S]*?)"""|\'\'\'([\s\S]*?)\'\'\')', re.MULTILINE),
    "javascript": re.compile(r'/\*\*([\s\S]*?)\*/'),
    "typescript": re.compile(r'/\*\*([\s\S]*?)\*/'),
    "rust": re.compile(r'^(?:\s*///[^\n]*\n)+', re.MULTILINE),
    "go": re.compile(r'^(?:\s*//[^\n]*\n)+', re.MULTILINE),
    "ruby": re.compile(r'^(?:\s*#[^\n]*\n)+', re.MULTILINE),
    "php": re.compile(r'/\*\*([\s\S]*?)\*/'),
    "sql": re.compile(r'^--[^\n]*', re.MULTILINE),
}


def _node_id(file_path: str, name: str, node_type: NodeType) -> str:
    raw = f"{file_path}::{node_type.value}::{name}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _module_id(file_path: str) -> str:
    return _node_id(file_path, "__module__", NodeType.MODULE)


def _node_text(node, source_bytes: bytes) -> str:
    """Extract UTF-8 text for a tree-sitter node."""
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _clean_doc_comment(raw: str, language_id: str) -> str:
    """Strip comment markers from a raw doc comment string."""
    if not raw:
        return ""
    # Strip /** */ markers
    raw = re.sub(r"^\s*/\*+\s*", "", raw)
    raw = re.sub(r"\s*\*+/\s*$", "", raw)
    # Strip leading * on each line (JSDoc style)
    raw = re.sub(r"^\s*\*\s?", "", raw, flags=re.MULTILINE)
    # Strip /// and // and # markers
    raw = re.sub(r"^\s*/{1,3}\s?", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"^\s*#\s?", "", raw, flags=re.MULTILINE)
    # Strip -- SQL comments
    raw = re.sub(r"^\s*--\s?", "", raw, flags=re.MULTILINE)
    return raw.strip()


class TreeSitterParser:
    """Universal multi-language AST parser backed by tree-sitter.

    One instance is typically shared across a full index run. Each language
    gets its own cached tree_sitter.Parser and pre-compiled Query objects.
    """

    def __init__(self) -> None:
        self._ts_parsers: dict[str, object] = {}   # language_id → Parser
        self._ts_langs: dict[str, object] = {}     # language_id → Language
        self._queries: dict[str, str] = {}         # language_id → raw .scm text
        self._compiled_queries: dict[str, object] = {}  # language_id → compiled Query

    # ── Public interface ────────────────────────────────────────────────────────

    def parse_file(
        self,
        file_path: str,
        source: Optional[str] = None,
    ) -> tuple[list[NeuralNode], list[NeuralEdge]]:
        """Parse a source file into NeuralNodes and NeuralEdges.

        Args:
            file_path: Relative path used as the node's file_path.
            source: File source text. If None, the file is read from disk.

        Returns:
            (nodes, edges) — all edges initially unresolved for cross-file linking.
        """
        lang = detect_language(file_path)
        if lang is None:
            return [], []

        if source is None:
            try:
                source = Path(file_path).read_text(errors="replace")
            except OSError:
                return [], []

        try:
            ts_lang = self._get_language(lang)
            if ts_lang is None:
                return [], []
            parser = self._get_parser(lang.id, ts_lang)
            source_bytes = source.encode("utf-8")
            tree = parser.parse(source_bytes)
            return self._extract(tree, lang, file_path, source, source_bytes)
        except Exception as exc:
            logger.warning("ts_parser: failed to parse %s (%s): %s", file_path, lang.id, exc)
            return [], []

    # ── Grammar loading ─────────────────────────────────────────────────────────

    def _get_language(self, lang: LanguageSpec) -> Optional[object]:
        """Load and cache a tree-sitter Language object for the given spec."""
        if lang.id in self._ts_langs:
            return self._ts_langs[lang.id]
        try:
            import importlib, tree_sitter
            mod = importlib.import_module(lang.tree_sitter_package)
            grammar_fn = getattr(mod, lang.grammar_fn, None)
            if grammar_fn is None:
                raise AttributeError(
                    f"module '{lang.tree_sitter_package}' has no attribute '{lang.grammar_fn}'"
                )
            ts_lang = tree_sitter.Language(grammar_fn())
            self._ts_langs[lang.id] = ts_lang
            return ts_lang
        except Exception as exc:
            logger.warning("ts_parser: cannot load grammar for %s: %s", lang.id, exc)
            self._ts_langs[lang.id] = None
            return None

    def _get_parser(self, language_id: str, ts_lang: object) -> object:
        """Get or create a cached tree_sitter.Parser for the given language."""
        if language_id not in self._ts_parsers:
            import tree_sitter
            p = tree_sitter.Parser(ts_lang)
            self._ts_parsers[language_id] = p
        return self._ts_parsers[language_id]

    def _get_query_text(self, language_id: str) -> Optional[str]:
        """Load and cache the .scm query file for a language."""
        if language_id not in self._queries:
            scm_path = _QUERIES_DIR / f"{language_id}.scm"
            if scm_path.exists():
                self._queries[language_id] = scm_path.read_text()
            else:
                logger.warning("ts_parser: no query file for %s at %s", language_id, scm_path)
                self._queries[language_id] = ""
        return self._queries[language_id] or None

    # ── Extraction ──────────────────────────────────────────────────────────────

    def _extract(
        self,
        tree: object,
        lang: LanguageSpec,
        file_path: str,
        source: str,
        source_bytes: bytes,
    ) -> tuple[list[NeuralNode], list[NeuralEdge]]:
        """Run all queries and build nodes/edges from captures."""
        source_lines = source.splitlines()
        nodes: list[NeuralNode] = []
        edges: list[NeuralEdge] = []

        # ── MODULE node (always created) ──────────────────────────────────────
        file_hash = hashlib.sha256(source_bytes).hexdigest()[:16]
        module_node = NeuralNode(
            id=_module_id(file_path),
            name=Path(file_path).stem,
            node_type=NodeType.MODULE,
            file_path=file_path,
            language=lang.id,
            line_start=1,
            line_end=len(source_lines),
            summary_short=f"{lang.name} module: {Path(file_path).name}",
            is_public=True,
            content_hash=file_hash,
            raw_code=source[:8000],  # cap to avoid huge blobs
        )
        # Extract module docstring
        module_node.docstring = self._extract_module_docstring(tree.root_node, source_bytes, lang.id)
        if module_node.docstring:
            module_node.summary_short = module_node.docstring[:120]
        nodes.append(module_node)

        query_text = self._get_query_text(lang.id)
        if not query_text:
            return nodes, edges

        # Load query from the tree-sitter Language (compile once, reuse per language)
        ts_lang_obj = self._ts_langs.get(lang.id)
        if ts_lang_obj is None:
            return nodes, edges

        if lang.id not in self._compiled_queries:
            try:
                import tree_sitter as _ts
                self._compiled_queries[lang.id] = _ts.Query(ts_lang_obj, query_text)
            except Exception as exc:
                logger.warning("ts_parser: bad query for %s: %s", lang.id, exc)
                self._compiled_queries[lang.id] = None
        query = self._compiled_queries[lang.id]
        if query is None:
            return nodes, edges

        # Run all captures at once — tree-sitter >=0.25 returns dict[str, list[Node]]
        try:
            import tree_sitter as _ts
            captures_dict: dict = _ts.QueryCursor(query).captures(tree.root_node)
        except Exception as exc:
            logger.warning("ts_parser: query execution failed for %s: %s", file_path, exc)
            return nodes, edges

        # Flatten dict[capture_name, list[Node]] → list of (capture_name, node) pairs,
        # sorted by byte offset so ordering is consistent with source order.
        captures = sorted(
            ((name, node) for name, nodes_list in captures_dict.items() for node in nodes_list),
            key=lambda pair: pair[1].start_byte,
        )

        # Group captures by category prefix (function, class, method, etc.)
        dispatch = {
            "function": self._handle_function,
            "class": self._handle_class,
            "method": self._handle_method,
            "interface": self._handle_interface,
            "trait": self._handle_trait,
            "struct": self._handle_struct,
            "enum": self._handle_enum,
            "type_alias": self._handle_type_alias,
            "procedure": self._handle_procedure,
            "table": self._handle_table,
            "view": self._handle_view,
            "import": self._handle_import,
            "call": self._handle_call,
            "constant": self._handle_constant,
        }

        seen_defs: set[str] = set()  # deduplicate by (category, start_byte)

        for capture_name, ts_node in captures:
            # capture_name format: "function.def", "class.name", "import.stmt", etc.
            parts = capture_name.split(".", 1)
            category = parts[0]
            role = parts[1] if len(parts) > 1 else ""

            # Only process top-level definition anchors to avoid duplicate nodes
            if role in ("def", "pub_def", "exported", "decorated", "stmt",
                        "arrow", "arrow_decl", "singleton_def", "impl",
                        "decl", "group_stmt", "include", "require",
                        "from_stmt", "require", "recv_expr", "expr",
                        "member_expr", "scoped_expr", "selector_expr",
                        "field_expr", "qualified_def"):
                key = (category, ts_node.start_byte)
                if key in seen_defs:
                    continue
                seen_defs.add(key)

                handler = dispatch.get(category)
                if handler:
                    result = handler(ts_node, lang, file_path, source_bytes, source_lines,
                                     module_node, nodes, edges)
                    if result:
                        new_nodes, new_edges = result
                        nodes.extend(new_nodes)
                        edges.extend(new_edges)
            elif category == "fk" and role == "def":
                self._handle_fk(ts_node, lang, file_path, source_bytes, module_node, edges)
            elif category == "ref" and role in ("from", "join"):
                self._handle_table_ref(ts_node, lang, file_path, source_bytes, module_node, edges)

        return nodes, edges

    # ── Node handlers ───────────────────────────────────────────────────────────

    def _make_function_node(
        self,
        ts_node, lang: LanguageSpec, file_path: str,
        source_bytes: bytes, source_lines: list[str],
        name: str, node_type: NodeType,
        parent_name: str = "",
    ) -> NeuralNode:
        full_name = f"{parent_name}.{name}" if parent_name else name
        node_id = _node_id(file_path, full_name, node_type)
        line_start = ts_node.start_point[0] + 1
        line_end = ts_node.end_point[0] + 1
        raw = _node_text(ts_node, source_bytes)

        sig = self._extract_signature(ts_node, source_bytes, lang.id)
        doc = self._extract_docstring(ts_node, source_bytes, lang.id)
        complexity = self._estimate_complexity(ts_node, lang.id)
        decorators = self._extract_decorators(ts_node, source_bytes, lang.id)
        is_public = not name.startswith("_") and not name.startswith("__")

        summary = doc.split("\n")[0][:120] if doc else f"{lang.name} {node_type.value}: {full_name}"

        return NeuralNode(
            id=node_id,
            name=full_name,
            node_type=node_type,
            file_path=file_path,
            language=lang.id,
            line_start=line_start,
            line_end=line_end,
            signature=sig or full_name,
            docstring=doc,
            summary_short=summary,
            complexity=complexity,
            decorators=decorators,
            is_public=is_public,
            content_hash=hashlib.sha256(raw.encode()).hexdigest()[:16],
            raw_code=raw,
        )

    # Node types whose body contains methods — functions inside these are handled by _handle_method
    _METHOD_CONTAINER_TYPES = frozenset({
        "impl_item",           # Rust impl block
        "class_body",          # JS/TS class body
        "class_declaration",   # PHP/TS class
        "class_definition",    # Python class
        "class",               # Ruby class
    })

    def _handle_function(
        self, ts_node, lang, file_path, source_bytes, source_lines,
        module_node, existing_nodes, existing_edges,
    ):
        name = self._find_capture_text("function.name", ts_node, source_bytes, lang)
        if not name:
            return None
        # Skip functions nested inside class/impl bodies — handled by _handle_method
        parent = ts_node.parent
        while parent is not None:
            if parent.type in self._METHOD_CONTAINER_TYPES:
                return None
            parent = parent.parent
        node = self._make_function_node(
            ts_node, lang, file_path, source_bytes, source_lines,
            name, NodeType.FUNCTION,
        )
        edges = [
            NeuralEdge(source_id=module_node.id, target_id=node.id, edge_type=EdgeType.CONTAINS),
        ]
        edges.extend(self._extract_call_edges(ts_node, node.id, source_bytes, lang))
        return [node], edges

    def _handle_class(
        self, ts_node, lang, file_path, source_bytes, source_lines,
        module_node, existing_nodes, existing_edges,
    ):
        name = self._find_capture_text("class.name", ts_node, source_bytes, lang)
        if not name:
            return None
        node_id = _node_id(file_path, name, NodeType.CLASS)
        line_start = ts_node.start_point[0] + 1
        line_end = ts_node.end_point[0] + 1
        raw = _node_text(ts_node, source_bytes)
        doc = self._extract_docstring(ts_node, source_bytes, lang.id)
        decorators = self._extract_decorators(ts_node, source_bytes, lang.id)

        node = NeuralNode(
            id=node_id,
            name=name,
            node_type=NodeType.CLASS,
            file_path=file_path,
            language=lang.id,
            line_start=line_start,
            line_end=line_end,
            docstring=doc,
            summary_short=(doc.split("\n")[0][:120] if doc else f"{lang.name} class: {name}"),
            decorators=decorators,
            is_public=not name.startswith("_"),
            content_hash=hashlib.sha256(raw.encode()).hexdigest()[:16],
            raw_code=raw,
        )
        edges = [
            NeuralEdge(source_id=module_node.id, target_id=node.id, edge_type=EdgeType.CONTAINS),
        ]
        # Inheritance
        bases = self._find_all_capture_texts("class.extends", ts_node, source_bytes, lang)
        bases += self._find_all_capture_texts("class.bases", ts_node, source_bytes, lang)
        bases += self._find_all_capture_texts("class.superclass", ts_node, source_bytes, lang)
        for base in bases:
            base = base.strip("() ")
            if base:
                edges.append(NeuralEdge(
                    source_id=node.id,
                    target_id=f"{_UNRESOLVED}{base}",
                    edge_type=EdgeType.INHERITS,
                    context=f"extends {base}",
                ))
        # Implements
        for iface in self._find_all_capture_texts("class.implements", ts_node, source_bytes, lang):
            edges.append(NeuralEdge(
                source_id=node.id,
                target_id=f"{_UNRESOLVED}{iface}",
                edge_type=EdgeType.IMPLEMENTS,
                context=f"implements {iface}",
            ))
        return [node], edges

    def _handle_method(
        self, ts_node, lang, file_path, source_bytes, source_lines,
        module_node, existing_nodes, existing_edges,
    ):
        name = self._find_capture_text("method.name", ts_node, source_bytes, lang)
        if not name:
            return None
        # Walk parent chain to find the enclosing class/impl/module name.
        # method.class_name is captured on the outer class node, not on the fn node itself.
        _CLASS_NODE_TYPES = {
            "class_definition", "class_declaration", "class",   # py/ts/js/php/ruby
            "impl_item",                                         # rust
            "module",                                            # ruby modules
        }
        class_name = ""
        node_ptr = ts_node.parent
        while node_ptr is not None:
            if node_ptr.type in _CLASS_NODE_TYPES:
                name_child = node_ptr.child_by_field_name("name")
                if name_child:
                    class_name = _node_text(name_child, source_bytes).strip()
                elif node_ptr.type == "impl_item":
                    # Rust impl: type: field holds the implementing type
                    type_child = node_ptr.child_by_field_name("type")
                    if type_child:
                        class_name = _node_text(type_child, source_bytes).strip()
                break
            node_ptr = node_ptr.parent

        # Find the parent class node to CONTAINS from
        parent_id = (
            _node_id(file_path, class_name, NodeType.CLASS)
            if class_name
            else module_node.id
        )
        node = self._make_function_node(
            ts_node, lang, file_path, source_bytes, source_lines,
            name, NodeType.METHOD, parent_name=class_name or "",
        )
        edges = [
            NeuralEdge(source_id=parent_id, target_id=node.id, edge_type=EdgeType.CONTAINS),
        ]
        edges.extend(self._extract_call_edges(ts_node, node.id, source_bytes, lang))
        return [node], edges

    def _handle_interface(
        self, ts_node, lang, file_path, source_bytes, source_lines,
        module_node, existing_nodes, existing_edges,
    ):
        name = self._find_capture_text("interface.name", ts_node, source_bytes, lang)
        if not name:
            return None
        node_id = _node_id(file_path, name, NodeType.INTERFACE)
        raw = _node_text(ts_node, source_bytes)
        node = NeuralNode(
            id=node_id,
            name=name,
            node_type=NodeType.INTERFACE,
            file_path=file_path,
            language=lang.id,
            line_start=ts_node.start_point[0] + 1,
            line_end=ts_node.end_point[0] + 1,
            summary_short=f"{lang.name} interface: {name}",
            is_public=True,
            content_hash=hashlib.sha256(raw.encode()).hexdigest()[:16],
            raw_code=raw,
        )
        edges = [
            NeuralEdge(source_id=module_node.id, target_id=node.id, edge_type=EdgeType.CONTAINS),
        ]
        for ext in self._find_all_capture_texts("interface.extends", ts_node, source_bytes, lang):
            edges.append(NeuralEdge(
                source_id=node.id,
                target_id=f"{_UNRESOLVED}{ext}",
                edge_type=EdgeType.INHERITS,
                context=f"extends {ext}",
            ))
        return [node], edges

    def _handle_trait(
        self, ts_node, lang, file_path, source_bytes, source_lines,
        module_node, existing_nodes, existing_edges,
    ):
        name = self._find_capture_text("trait.name", ts_node, source_bytes, lang)
        if not name:
            return None
        node_id = _node_id(file_path, name, NodeType.TRAIT)
        raw = _node_text(ts_node, source_bytes)
        node = NeuralNode(
            id=node_id, name=name, node_type=NodeType.TRAIT,
            file_path=file_path, language=lang.id,
            line_start=ts_node.start_point[0] + 1,
            line_end=ts_node.end_point[0] + 1,
            summary_short=f"Rust trait: {name}",
            is_public=not name.startswith("_"),
            content_hash=hashlib.sha256(raw.encode()).hexdigest()[:16],
            raw_code=raw,
        )
        edges = [NeuralEdge(source_id=module_node.id, target_id=node.id, edge_type=EdgeType.CONTAINS)]
        return [node], edges

    def _handle_struct(
        self, ts_node, lang, file_path, source_bytes, source_lines,
        module_node, existing_nodes, existing_edges,
    ):
        name = self._find_capture_text("struct.name", ts_node, source_bytes, lang)
        if not name:
            return None
        node_id = _node_id(file_path, name, NodeType.STRUCT)
        raw = _node_text(ts_node, source_bytes)
        node = NeuralNode(
            id=node_id, name=name, node_type=NodeType.STRUCT,
            file_path=file_path, language=lang.id,
            line_start=ts_node.start_point[0] + 1,
            line_end=ts_node.end_point[0] + 1,
            summary_short=f"{lang.name} struct: {name}",
            is_public=not name.startswith("_"),
            content_hash=hashlib.sha256(raw.encode()).hexdigest()[:16],
            raw_code=raw,
        )
        edges = [NeuralEdge(source_id=module_node.id, target_id=node.id, edge_type=EdgeType.CONTAINS)]
        return [node], edges

    def _handle_enum(
        self, ts_node, lang, file_path, source_bytes, source_lines,
        module_node, existing_nodes, existing_edges,
    ):
        name = self._find_capture_text("enum.name", ts_node, source_bytes, lang)
        if not name:
            return None
        node_id = _node_id(file_path, name, NodeType.ENUM)
        raw = _node_text(ts_node, source_bytes)
        node = NeuralNode(
            id=node_id, name=name, node_type=NodeType.ENUM,
            file_path=file_path, language=lang.id,
            line_start=ts_node.start_point[0] + 1,
            line_end=ts_node.end_point[0] + 1,
            summary_short=f"{lang.name} enum: {name}",
            is_public=not name.startswith("_"),
            content_hash=hashlib.sha256(raw.encode()).hexdigest()[:16],
            raw_code=raw,
        )
        edges = [NeuralEdge(source_id=module_node.id, target_id=node.id, edge_type=EdgeType.CONTAINS)]
        return [node], edges

    def _handle_type_alias(
        self, ts_node, lang, file_path, source_bytes, source_lines,
        module_node, existing_nodes, existing_edges,
    ):
        name = self._find_capture_text("type_alias.name", ts_node, source_bytes, lang)
        if not name:
            return None
        node_id = _node_id(file_path, name, NodeType.TYPE_ALIAS)
        raw = _node_text(ts_node, source_bytes)
        node = NeuralNode(
            id=node_id, name=name, node_type=NodeType.TYPE_ALIAS,
            file_path=file_path, language=lang.id,
            line_start=ts_node.start_point[0] + 1,
            line_end=ts_node.end_point[0] + 1,
            summary_short=f"{lang.name} type alias: {name}",
            is_public=not name.startswith("_"),
            content_hash=hashlib.sha256(raw.encode()).hexdigest()[:16],
            raw_code=raw,
        )
        edges = [NeuralEdge(source_id=module_node.id, target_id=node.id, edge_type=EdgeType.CONTAINS)]
        return [node], edges

    def _handle_constant(
        self, ts_node, lang, file_path, source_bytes, source_lines,
        module_node, existing_nodes, existing_edges,
    ):
        name = self._find_capture_text("constant.name", ts_node, source_bytes, lang)
        # Filter: ALL_CAPS (Python/Rust/PHP style) or PascalCase-exported (Go/Ruby style)
        if not name or not (name.isupper() or (name[0].isupper() and "_" not in name)):
            return None
        node_id = _node_id(file_path, name, NodeType.CONSTANT)
        raw = _node_text(ts_node, source_bytes)
        node = NeuralNode(
            id=node_id, name=name, node_type=NodeType.CONSTANT,
            file_path=file_path, language=lang.id,
            line_start=ts_node.start_point[0] + 1,
            line_end=ts_node.end_point[0] + 1,
            summary_short=f"{lang.name} constant: {name}",
            is_public=not name.startswith("_"),
            content_hash=hashlib.sha256(raw.encode()).hexdigest()[:16],
            raw_code=raw,
        )
        edges = [NeuralEdge(source_id=module_node.id, target_id=node.id, edge_type=EdgeType.CONTAINS)]
        return [node], edges

    def _handle_table(
        self, ts_node, lang, file_path, source_bytes, source_lines,
        module_node, existing_nodes, existing_edges,
    ):
        name = self._find_capture_text("table.name", ts_node, source_bytes, lang)
        if not name:
            return None
        node_id = _node_id(file_path, name, NodeType.TABLE)
        raw = _node_text(ts_node, source_bytes)
        sig = self._extract_sql_columns(ts_node, source_bytes)
        node = NeuralNode(
            id=node_id, name=name, node_type=NodeType.TABLE,
            file_path=file_path, language=lang.id,
            line_start=ts_node.start_point[0] + 1,
            line_end=ts_node.end_point[0] + 1,
            signature=sig,
            summary_short=f"SQL table: {name}",
            is_public=True,
            content_hash=hashlib.sha256(raw.encode()).hexdigest()[:16],
            raw_code=raw,
        )
        edges = [NeuralEdge(source_id=module_node.id, target_id=node.id, edge_type=EdgeType.CONTAINS)]
        return [node], edges

    def _handle_view(
        self, ts_node, lang, file_path, source_bytes, source_lines,
        module_node, existing_nodes, existing_edges,
    ):
        name = self._find_capture_text("view.name", ts_node, source_bytes, lang)
        if not name:
            return None
        node_id = _node_id(file_path, name, NodeType.VIEW)
        raw = _node_text(ts_node, source_bytes)
        node = NeuralNode(
            id=node_id, name=name, node_type=NodeType.VIEW,
            file_path=file_path, language=lang.id,
            line_start=ts_node.start_point[0] + 1,
            line_end=ts_node.end_point[0] + 1,
            summary_short=f"SQL view: {name}",
            is_public=True,
            content_hash=hashlib.sha256(raw.encode()).hexdigest()[:16],
            raw_code=raw,
        )
        edges = [NeuralEdge(source_id=module_node.id, target_id=node.id, edge_type=EdgeType.CONTAINS)]
        return [node], edges

    def _handle_procedure(
        self, ts_node, lang, file_path, source_bytes, source_lines,
        module_node, existing_nodes, existing_edges,
    ):
        name = self._find_capture_text("procedure.name", ts_node, source_bytes, lang)
        if not name:
            return None
        node_id = _node_id(file_path, name, NodeType.STORED_PROCEDURE)
        raw = _node_text(ts_node, source_bytes)
        node = NeuralNode(
            id=node_id, name=name, node_type=NodeType.STORED_PROCEDURE,
            file_path=file_path, language=lang.id,
            line_start=ts_node.start_point[0] + 1,
            line_end=ts_node.end_point[0] + 1,
            summary_short=f"SQL procedure: {name}",
            is_public=True,
            content_hash=hashlib.sha256(raw.encode()).hexdigest()[:16],
            raw_code=raw,
        )
        edges = [NeuralEdge(source_id=module_node.id, target_id=node.id, edge_type=EdgeType.CONTAINS)]
        return [node], edges

    def _handle_import(
        self, ts_node, lang, file_path, source_bytes, source_lines,
        module_node, existing_nodes, existing_edges,
    ):
        path_text = self._find_capture_text("import.source", ts_node, source_bytes, lang)
        module_text = self._find_capture_text("import.module", ts_node, source_bytes, lang)
        from_module = self._find_capture_text("import.from_module", ts_node, source_bytes, lang)
        target = (path_text or module_text or from_module or "").strip("\"'`")
        if target:
            existing_edges.append(NeuralEdge(
                source_id=module_node.id,
                target_id=f"{_UNRESOLVED}{target}",
                edge_type=EdgeType.IMPORTS,
                context=f"imports {target}",
            ))
        return None

    def _handle_call(
        self, ts_node, lang, file_path, source_bytes, source_lines,
        module_node, existing_nodes, existing_edges,
    ):
        name = self._find_capture_text("call.name", ts_node, source_bytes, lang)
        if name and name not in {"print", "len", "str", "int", "bool", "list",
                                  "dict", "set", "tuple", "range", "type", "super",
                                  "console", "require", "exports"}:
            existing_edges.append(NeuralEdge(
                source_id=module_node.id,
                target_id=f"{_UNRESOLVED}{name}",
                edge_type=EdgeType.CALLS,
                context=f"calls {name}",
            ))
        return None

    def _handle_fk(self, ts_node, lang, file_path, source_bytes, module_node, edges):
        ref_table = self._find_capture_text("fk.references_table", ts_node, source_bytes, lang)
        if ref_table:
            edges.append(NeuralEdge(
                source_id=module_node.id,
                target_id=f"{_UNRESOLVED}{ref_table}",
                edge_type=EdgeType.RELATES_TO,
                context=f"FK references {ref_table}",
            ))

    def _handle_table_ref(self, ts_node, lang, file_path, source_bytes, module_node, edges):
        table_name = self._find_capture_text("ref.table_name", ts_node, source_bytes, lang)
        if table_name:
            edges.append(NeuralEdge(
                source_id=module_node.id,
                target_id=f"{_UNRESOLVED}{table_name}",
                edge_type=EdgeType.USES,
                context=f"references table {table_name}",
            ))

    # ── Call edge extraction ────────────────────────────────────────────────────

    def _extract_call_edges(
        self, ts_node, source_node_id: str, source_bytes: bytes, lang: LanguageSpec,
    ) -> list[NeuralEdge]:
        """Find call expressions inside a function/method body."""
        edges = []
        seen: set[str] = set()

        CALL_NODE_TYPES = {
            "python": "call",
            "javascript": "call_expression",
            "typescript": "call_expression",
            "rust": "call_expression",
            "go": "call_expression",
            "ruby": "call",
            "php": "function_call_expression",
            "sql": None,
        }
        call_type = CALL_NODE_TYPES.get(lang.id)
        if not call_type:
            return edges

        def walk(node):
            if node.type == call_type:
                # Get the function name child
                for child in node.children:
                    if child.type in ("identifier", "name", "field_identifier",
                                      "property_identifier", "field_expression",
                                      "member_expression", "attribute"):
                        text = _node_text(child, source_bytes).split(".")[-1]
                        if text and text not in seen and len(text) > 1:
                            seen.add(text)
                            edges.append(NeuralEdge(
                                source_id=source_node_id,
                                target_id=f"{_UNRESOLVED}{text}",
                                edge_type=EdgeType.CALLS,
                                context=f"calls {text}",
                            ))
                        break
            for child in node.children:
                walk(child)

        walk(ts_node)
        return edges

    # ── Signature extraction ────────────────────────────────────────────────────

    def _extract_signature(self, ts_node, source_bytes: bytes, language_id: str) -> str:
        """Extract a human-readable signature from a function/method node."""
        param_types = {
            "python": "parameters",
            "javascript": "formal_parameters",
            "typescript": "formal_parameters",
            "rust": "parameters",
            "go": "parameter_list",
            "ruby": "method_parameters",
            "php": "formal_parameters",
        }
        param_type = param_types.get(language_id, "parameters")
        return_types = {
            "python": "type",
            "typescript": "type_annotation",
            "rust": ("type_identifier", "generic_type", "reference_type",
                     "scoped_type_identifier", "primitive_type"),
            "go": ("type_identifier", "pointer_type", "qualified_type"),
        }

        name = ""
        params = ""
        ret = ""
        is_async = False

        for child in ts_node.children:
            if child.type == "async":
                is_async = True
            elif child.type == "identifier" or child.type == "name" or child.type == "field_identifier":
                name = _node_text(child, source_bytes)
            elif child.type == param_type:
                params = _node_text(child, source_bytes)
            elif child.type in ("type_annotation", "return_type"):
                ret = " -> " + _node_text(child, source_bytes).lstrip(":").strip()
            elif language_id == "rust" and child.type in (
                "type_identifier", "generic_type", "reference_type",
                "scoped_type_identifier", "primitive_type",
            ):
                # Rust return type comes after ->
                pass

        prefix = "async def " if (is_async and language_id == "python") else ""
        sig = f"{prefix}{name}{params}{ret}".strip()
        return sig[:256] if sig else ""

    # ── Docstring extraction ────────────────────────────────────────────────────

    def _extract_module_docstring(self, root_node, source_bytes: bytes, language_id: str) -> str:
        """Extract module-level docstring from the file's root node."""
        for child in root_node.children:
            if child.type in ("expression_statement", "comment"):
                text = _node_text(child, source_bytes)
                if text.startswith(('"""', "'''", "/**", "//!", "///", "#")):
                    return _clean_doc_comment(text, language_id)
                break
            if child.type not in ("shebang", "newline", "blank_line"):
                break
        return ""

    def _extract_docstring(self, ts_node, source_bytes: bytes, language_id: str) -> str:
        """Extract doc comment for a function/class node."""
        # Python: first child of body is a string expression
        if language_id == "python":
            for child in ts_node.children:
                if child.type == "block":
                    for body_child in child.children:
                        if body_child.type == "expression_statement":
                            for expr in body_child.children:
                                if expr.type == "string":
                                    return _clean_doc_comment(
                                        _node_text(expr, source_bytes), language_id
                                    )
                        break

        # JS/TS/PHP: look for /** */ comment before the node
        if language_id in ("javascript", "typescript", "php"):
            prev = ts_node.prev_sibling
            while prev and prev.type in ("comment", "decorator"):
                text = _node_text(prev, source_bytes)
                if text.startswith("/**"):
                    return _clean_doc_comment(text, language_id)
                prev = prev.prev_sibling

        # Rust: /// line comments before the item
        if language_id == "rust":
            lines: list[str] = []
            prev = ts_node.prev_sibling
            while prev and prev.type in ("line_comment", "block_comment", "attribute_item"):
                text = _node_text(prev, source_bytes)
                if text.startswith("///"):
                    lines.insert(0, text)
                prev = prev.prev_sibling
            if lines:
                return _clean_doc_comment("\n".join(lines), language_id)

        # Go: // line comments before the declaration
        if language_id == "go":
            lines = []
            prev = ts_node.prev_sibling
            while prev and prev.type == "comment":
                lines.insert(0, _node_text(prev, source_bytes))
                prev = prev.prev_sibling
            if lines:
                return _clean_doc_comment("\n".join(lines), language_id)

        return ""

    # ── Complexity estimation ───────────────────────────────────────────────────

    def _estimate_complexity(self, ts_node, language_id: str) -> int:
        """Estimate cyclomatic complexity by counting branch nodes."""
        branch_types = _COMPLEXITY_NODE_TYPES.get(language_id, set())
        if not branch_types:
            return 0

        count = 1  # base complexity

        def walk(node):
            nonlocal count
            if node.type in branch_types:
                count += 1
            for child in node.children:
                walk(child)

        walk(ts_node)
        return min(count, 100)  # cap at 100

    # ── Decorator extraction ────────────────────────────────────────────────────

    def _extract_decorators(self, ts_node, source_bytes: bytes, language_id: str) -> list[str]:
        """Extract decorator/attribute names for a node."""
        decorators = []
        if language_id == "python":
            prev = ts_node.prev_sibling
            while prev and prev.type in ("decorator",):
                decorators.insert(0, _node_text(prev, source_bytes).lstrip("@").strip())
                prev = prev.prev_sibling
        elif language_id in ("typescript", "javascript"):
            prev = ts_node.prev_sibling
            while prev and prev.type == "decorator":
                decorators.insert(0, _node_text(prev, source_bytes).lstrip("@").strip())
                prev = prev.prev_sibling
        elif language_id == "rust":
            prev = ts_node.prev_sibling
            while prev and prev.type == "attribute_item":
                decorators.insert(0, _node_text(prev, source_bytes).strip())
                prev = prev.prev_sibling
        return decorators

    # ── SQL helpers ─────────────────────────────────────────────────────────────

    def _extract_sql_columns(self, ts_node, source_bytes: bytes) -> str:
        """Build a column summary signature for a CREATE TABLE node."""
        cols: list[str] = []
        for child in ts_node.children:
            if "column" in child.type or child.type == "column_definitions":
                for col in child.children:
                    if "column" in col.type:
                        cols.append(_node_text(col, source_bytes).split()[0])
        return "(" + ", ".join(cols[:8]) + ("..." if len(cols) > 8 else "") + ")" if cols else ""

    # ── Capture text helpers ────────────────────────────────────────────────────

    def _find_capture_text(
        self, capture_name: str, ts_node, source_bytes: bytes, lang: LanguageSpec,
    ) -> str:
        """Run a minimal inline query to find the text of the first named capture within ts_node."""
        # We search the node's text for the child by type that matches the capture purpose
        # This is a simpler approach: parse the node text directly
        parts = capture_name.split(".")
        if len(parts) < 2:
            return ""

        role = parts[1]  # "name", "params", "extends", etc.

        NAME_CHILD_TYPES = {
            "name": ("identifier", "type_identifier", "name", "constant",
                     "property_identifier", "field_identifier"),
            "class_name": ("identifier", "type_identifier", "name", "constant"),
            "impl_type": ("type_identifier",),
            "receiver_type": ("type_identifier",),
        }
        child_types = NAME_CHILD_TYPES.get(role, ("identifier", "type_identifier", "name", "constant"))

        # Named-field lookup (fast path for well-structured grammars)
        field_map = {
            "name": ts_node.child_by_field_name("name"),
            "params": ts_node.child_by_field_name("parameters"),
            "return_type": ts_node.child_by_field_name("return_type"),
            "extends": ts_node.child_by_field_name("superclass"),
            "from_module": ts_node.child_by_field_name("module_name"),
            "source": ts_node.child_by_field_name("source"),
            "path": ts_node.child_by_field_name("path"),
            # Python import_statement uses field "name" for the module; others use "module"
            "module": ts_node.child_by_field_name("module") or ts_node.child_by_field_name("name"),
            # Python class_definition uses field "superclasses" for base classes
            "bases": ts_node.child_by_field_name("superclasses"),
        }
        child_node = field_map.get(role)
        if child_node:
            return _node_text(child_node, source_bytes).strip("\"' \t\n`")

        # Recursive BFS up to 3 levels for name/identifier nodes (handles nested grammar structures)
        if role in ("name", "class_name", "impl_type", "receiver_type"):
            def _bfs(node, depth: int) -> str:
                for child in node.children:
                    if child.type in child_types:
                        return _node_text(child, source_bytes).strip()
                if depth > 0:
                    for child in node.children:
                        result = _bfs(child, depth - 1)
                        if result:
                            return result
                return ""
            found = _bfs(ts_node, 2)
            if found:
                return found

        return ""

    def _find_all_capture_texts(
        self, capture_name: str, ts_node, source_bytes: bytes, lang: LanguageSpec,
    ) -> list[str]:
        """Return all texts matching a capture name within a node (for multiple bases, etc.)"""
        result = self._find_capture_text(capture_name, ts_node, source_bytes, lang)
        return [result] if result else []
