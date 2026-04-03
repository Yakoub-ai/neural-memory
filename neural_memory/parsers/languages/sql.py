"""SQL and Prisma schema parser for neural memory.

Handles:
- Plain SQL DDL (CREATE TABLE statements)
- Prisma schema files (model blocks)

No tree-sitter dependency — uses regex parsing only.
"""

from __future__ import annotations

import re
from typing import Optional

from ...models import NeuralNode, NeuralEdge, NodeType, EdgeType


# SQL column type keywords (case-insensitive)
_SQL_TYPES = (
    "VARCHAR", "CHAR", "TEXT", "CLOB",
    "INTEGER", "INT", "BIGINT", "SMALLINT", "TINYINT",
    "FLOAT", "DOUBLE", "REAL", "DECIMAL", "NUMERIC",
    "BOOLEAN", "BOOL",
    "DATE", "TIME", "DATETIME", "TIMESTAMP",
    "BLOB", "BINARY", "VARBINARY",
    "JSON", "JSONB", "UUID",
    "SERIAL", "BIGSERIAL",
)

_SQL_TYPE_PATTERN = re.compile(
    r"^\s*(\w+)\s+(" + "|".join(_SQL_TYPES) + r")(?:[\s(,]|$)",
    re.IGNORECASE,
)

_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`\"']?(\w+)[`\"']?\s*\(",
    re.IGNORECASE,
)

_REFERENCES_RE = re.compile(
    r"REFERENCES\s+[`\"']?(\w+)[`\"']?\s*\(\s*[`\"']?(\w+)[`\"']?\s*\)",
    re.IGNORECASE,
)

_PRIMARY_KEY_INLINE_RE = re.compile(r"\bPRIMARY\s+KEY\b", re.IGNORECASE)

_PRISMA_MODEL_RE = re.compile(r"^model\s+(\w+)\s*\{", re.MULTILINE)
_PRISMA_FIELD_RE = re.compile(r"^\s+(\w+)\s+(\w+\??)\s*(.*)")
_PRISMA_RELATION_RE = re.compile(r"@relation\s*\(.*?fields:\s*\[(\w+)\].*?references:\s*\[(\w+)\]", re.DOTALL)


# ---------------------------------------------------------------------------
# SQL parsing helpers
# ---------------------------------------------------------------------------

def _extract_table_block(source: str, match_end: int) -> str:
    """Extract the content between the opening ( and matching ) of a CREATE TABLE."""
    depth = 0
    start = -1
    for i in range(match_end - 1, len(source)):
        ch = source[i]
        if ch == "(":
            if depth == 0:
                start = i + 1
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return source[start:i]
    return ""


def _parse_sql_columns(block: str, table_name: str) -> tuple[list[NeuralNode], list[NeuralEdge], list[tuple[str, str, str]]]:
    """Parse column definitions from a CREATE TABLE block.

    Returns:
        nodes: COLUMN NeuralNodes
        edges: empty list (FK edges resolved later)
        fk_hints: list of (source_col_id, ref_table, ref_col) for later wiring
    """
    nodes: list[NeuralNode] = []
    fk_hints: list[tuple[str, str, str]] = []

    lines = block.splitlines()
    for line in lines:
        stripped = line.strip().rstrip(",")
        if not stripped or stripped.upper().startswith("PRIMARY KEY") or stripped.upper().startswith("UNIQUE") or stripped.upper().startswith("INDEX") or stripped.upper().startswith("KEY") or stripped.upper().startswith("CONSTRAINT"):
            continue

        col_m = _SQL_TYPE_PATTERN.match(line)
        if not col_m:
            continue

        col_name = col_m.group(1)
        col_type = col_m.group(2).upper()

        is_primary = bool(_PRIMARY_KEY_INLINE_RE.search(stripped))
        is_nullable = "NOT NULL" not in stripped.upper()

        col_id = f"col::{table_name}::{col_name}"
        col_node = NeuralNode(
            id=col_id,
            name=col_name,
            node_type=NodeType.COLUMN,
            file_path="",
            line_start=0,
            line_end=0,
            category="database",
            signature=col_type,
            summary_short=f"Column {col_name} ({col_type})"
            + (" PK" if is_primary else "")
            + ("" if is_nullable else " NOT NULL"),
        )
        nodes.append(col_node)

        # Check for REFERENCES on same line
        ref_m = _REFERENCES_RE.search(stripped)
        if ref_m:
            fk_hints.append((col_id, ref_m.group(1), ref_m.group(2)))

    return nodes, [], fk_hints


def _parse_sql(source: str, file_path: str) -> tuple[list[NeuralNode], list[NeuralEdge]]:
    """Parse a SQL DDL file for CREATE TABLE statements."""
    nodes: list[NeuralNode] = []
    edges: list[NeuralEdge] = []
    fk_hints: list[tuple[str, str, str]] = []  # (source_col_id, ref_table, ref_col)

    # Track table node IDs for edge wiring
    table_nodes: dict[str, NeuralNode] = {}

    # Create a single DATABASE node for this file
    db_id = f"db::sql::{file_path}"
    db_node = NeuralNode(
        id=db_id,
        name=f"SQL Schema ({file_path})",
        node_type=NodeType.DATABASE,
        file_path=file_path,
        line_start=0,
        line_end=0,
        category="database",
        summary_short=f"SQL schema file: {file_path}",
    )
    nodes.append(db_node)

    for m in _CREATE_TABLE_RE.finditer(source):
        table_name = m.group(1)
        table_id = f"table::{table_name}"

        # Compute approximate line numbers
        line_start = source[:m.start()].count("\n") + 1

        table_node = NeuralNode(
            id=table_id,
            name=table_name,
            node_type=NodeType.TABLE,
            file_path=file_path,
            line_start=line_start,
            line_end=line_start,
            category="database",
            summary_short=f"Database table: {table_name}",
        )
        nodes.append(table_node)
        table_nodes[table_name] = table_node

        # DATABASE → TABLE
        edges.append(NeuralEdge(
            source_id=db_id,
            target_id=table_id,
            edge_type=EdgeType.CONTAINS,
        ))

        # Parse columns
        block = _extract_table_block(source, m.end())
        col_nodes, _, col_fk_hints = _parse_sql_columns(block, table_name)
        fk_hints.extend(col_fk_hints)

        for col_node in col_nodes:
            col_node.file_path = file_path
            nodes.append(col_node)
            # TABLE → COLUMN
            edges.append(NeuralEdge(
                source_id=table_id,
                target_id=col_node.id,
                edge_type=EdgeType.CONTAINS,
            ))

    # Wire FK REFERENCES edges (second pass, after all tables known)
    for source_col_id, ref_table, ref_col in fk_hints:
        target_id = f"col::{ref_table}::{ref_col}"
        edges.append(NeuralEdge(
            source_id=source_col_id,
            target_id=target_id,
            edge_type=EdgeType.REFERENCES,
            context=f"REFERENCES {ref_table}({ref_col})",
        ))

    # If no tables found, return empty (skip orphan db node)
    if not table_nodes:
        return [], []

    return nodes, edges


# ---------------------------------------------------------------------------
# Prisma parsing
# ---------------------------------------------------------------------------

def _parse_prisma(source: str, file_path: str) -> tuple[list[NeuralNode], list[NeuralEdge]]:
    """Parse a Prisma schema file for model blocks."""
    nodes: list[NeuralNode] = []
    edges: list[NeuralEdge] = []

    db_id = f"db::prisma::{file_path}"
    db_node = NeuralNode(
        id=db_id,
        name=f"Prisma Schema ({file_path})",
        node_type=NodeType.DATABASE,
        file_path=file_path,
        line_start=0,
        line_end=0,
        category="database",
        summary_short=f"Prisma schema file: {file_path}",
    )

    model_nodes: list[NeuralNode] = []

    # Split source into model blocks
    lines = source.splitlines()
    i = 0
    while i < len(lines):
        m = re.match(r"^model\s+(\w+)\s*\{", lines[i])
        if not m:
            i += 1
            continue

        model_name = m.group(1)
        table_name = _snake_case(model_name)
        line_start = i + 1

        # Collect body until closing }
        body_lines = []
        j = i + 1
        while j < len(lines):
            if lines[j].strip() == "}":
                break
            body_lines.append(lines[j])
            j += 1

        line_end = j + 1
        table_id = f"table::{table_name}"

        table_node = NeuralNode(
            id=table_id,
            name=table_name,
            node_type=NodeType.TABLE,
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            category="database",
            summary_short=f"Prisma model: {model_name} → table {table_name}",
        )
        model_nodes.append(table_node)

        # Parse fields
        for field_line in body_lines:
            fm = _PRISMA_FIELD_RE.match(field_line)
            if not fm:
                continue
            field_name = fm.group(1)
            field_type = fm.group(2)
            field_attrs = fm.group(3)

            # Skip relation fields (they reference other models, not columns)
            if "@relation" in field_attrs:
                # Still create the edge after all tables known
                rel_m = _PRISMA_RELATION_RE.search(field_attrs)
                if rel_m:
                    src_col_id = f"col::{table_name}::{rel_m.group(1)}"
                    # Target resolved after all models parsed
                    # Store as a deferred hint via context on a placeholder edge
                    pass
                continue

            # Skip @@-level directives
            if field_name.startswith("@@") or field_name.startswith("//"):
                continue

            is_primary = "@id" in field_attrs
            is_nullable = "?" in field_type
            col_type = field_type.rstrip("?")

            col_id = f"col::{table_name}::{field_name}"
            col_node = NeuralNode(
                id=col_id,
                name=field_name,
                node_type=NodeType.COLUMN,
                file_path=file_path,
                line_start=line_start,
                line_end=line_end,
                category="database",
                signature=col_type,
                summary_short=f"Column {field_name} ({col_type})"
                + (" PK" if is_primary else "")
                + (" nullable" if is_nullable else ""),
            )
            nodes.append(col_node)

            edges.append(NeuralEdge(
                source_id=table_id,
                target_id=col_id,
                edge_type=EdgeType.CONTAINS,
            ))

        i = j + 1

    if not model_nodes:
        return [], []

    nodes.insert(0, db_node)
    for tbl in model_nodes:
        nodes.insert(1, tbl)
        edges.append(NeuralEdge(
            source_id=db_id,
            target_id=tbl.id,
            edge_type=EdgeType.CONTAINS,
        ))

    return nodes, edges


def _snake_case(name: str) -> str:
    """Convert PascalCase to snake_case for Prisma table names."""
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s)
    return s.lower()


# ---------------------------------------------------------------------------
# SQLParser — LanguageParser protocol implementation
# ---------------------------------------------------------------------------

class SQLParser:
    """Parses SQL DDL and Prisma schema files into neural memory nodes/edges."""

    @property
    def language_id(self) -> str:
        return "sql"

    @property
    def file_extensions(self) -> list[str]:
        return [".sql", ".prisma"]

    def parse_file(
        self,
        file_path: str,
        source: Optional[str] = None,
    ) -> tuple[list[NeuralNode], list[NeuralEdge]]:
        """Parse a SQL or Prisma file."""
        if source is None:
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    source = f.read()
            except OSError:
                return [], []

        if not source.strip():
            return [], []

        if file_path.endswith(".prisma"):
            return _parse_prisma(source, file_path)
        return _parse_sql(source, file_path)

    def resolve_edges(
        self,
        all_nodes: dict[str, NeuralNode],
        edges: list[NeuralEdge],
    ) -> list[NeuralEdge]:
        """Edges are already resolved during parse_file; return as-is."""
        return edges
