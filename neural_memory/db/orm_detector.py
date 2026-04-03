"""ORM model detector — converts CLASS nodes to database schema nodes/edges.

Supports:
- Python SQLAlchemy (declarative Base)
- Python Django (models.Model)
- TypeScript TypeORM (@Entity decorator)
- Go GORM (gorm struct tags)
- Rust Diesel (table_name attribute / #[derive(Queryable)])
"""

from __future__ import annotations

import re
from typing import Optional

from ..models import NeuralNode, NeuralEdge, NodeType, EdgeType
from . import ColumnDef, TableSchema


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _snake_case(name: str) -> str:
    """Convert CamelCase to snake_case (Django/GORM default table naming)."""
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s)
    return s.lower()


def _snake_plural(name: str) -> str:
    """snake_case + naive plural (GORM default)."""
    s = _snake_case(name)
    if s.endswith("s"):
        return s
    return s + "s"


# ---------------------------------------------------------------------------
# Per-ORM detection helpers
# ---------------------------------------------------------------------------

def _detect_sqlalchemy(node: NeuralNode) -> Optional[TableSchema]:
    """Detect SQLAlchemy declarative model."""
    raw = node.raw_code or ""
    summary = node.summary_short or ""
    combined = raw + "\n" + summary

    has_tablename = "__tablename__" in combined
    has_base = bool(re.search(r"\b(Base|DeclarativeBase)\b", combined))
    if not (has_tablename and has_base):
        return None

    # Extract table name
    m = re.search(r'__tablename__\s*=\s*["\'](\w+)["\']', combined)
    if not m:
        return None
    table_name = m.group(1)

    schema = TableSchema(
        table_name=table_name,
        source_node_id=node.id,
        language="python",
        file_path=node.file_path,
    )

    # Extract columns: lines like "col_name = Column(Type, ...)"
    for line in combined.splitlines():
        col_match = re.match(r"\s*(\w+)\s*=\s*Column\s*\(([^)]*)\)", line)
        if not col_match:
            continue
        col_name = col_match.group(1)
        col_args = col_match.group(2)

        # Type is the first positional arg (e.g. String, Integer, ForeignKey)
        type_m = re.match(r"\s*([A-Za-z_][\w.]*)", col_args)
        col_type = type_m.group(1) if type_m else ""

        is_primary = "primary_key=True" in col_args or "primary_key = True" in col_args
        is_nullable = "nullable=False" not in col_args

        # ForeignKey("other_table.col")
        fk_m = re.search(r'ForeignKey\s*\(\s*["\'](\w+)\.(\w+)["\']', col_args)
        foreign_key = f"{fk_m.group(1)}.{fk_m.group(2)}" if fk_m else ""

        schema.columns.append(ColumnDef(
            name=col_name,
            col_type=col_type,
            is_primary=is_primary,
            is_nullable=is_nullable,
            foreign_key=foreign_key,
        ))

    return schema


def _detect_django(node: NeuralNode) -> Optional[TableSchema]:
    """Detect Django model (inherits models.Model)."""
    raw = node.raw_code or ""
    summary = node.summary_short or ""
    combined = raw + "\n" + summary

    if "models.Model" not in combined:
        return None

    table_name = _snake_case(node.name)

    schema = TableSchema(
        table_name=table_name,
        source_node_id=node.id,
        language="python",
        file_path=node.file_path,
    )

    # Extract fields: patterns like CharField, IntegerField, ForeignKey, etc.
    field_pattern = re.compile(
        r"\s*(\w+)\s*=\s*models\.(\w+Field|ForeignKey|ManyToManyField|OneToOneField)\s*\(([^)]*)\)"
    )
    for line in combined.splitlines():
        m = field_pattern.match(line)
        if not m:
            continue
        col_name = m.group(1)
        col_type = m.group(2)
        col_args = m.group(3)

        is_primary = col_name == "id" or "primary_key=True" in col_args
        is_nullable = "null=True" in col_args or "blank=True" in col_args

        # ForeignKey to another model — track as FK
        foreign_key = ""
        if col_type in ("ForeignKey", "OneToOneField"):
            fk_m = re.match(r"\s*['\"]?(\w+)['\"]?", col_args)
            if fk_m and fk_m.group(1) not in ("self", ""):
                foreign_key = f"{_snake_case(fk_m.group(1))}.id"

        schema.columns.append(ColumnDef(
            name=col_name,
            col_type=col_type,
            is_primary=is_primary,
            is_nullable=is_nullable,
            foreign_key=foreign_key,
        ))

    return schema


def _detect_typeorm(node: NeuralNode) -> Optional[TableSchema]:
    """Detect TypeORM entity (TypeScript @Entity decorator)."""
    raw = node.raw_code or ""
    summary = node.summary_short or ""
    combined = raw + "\n" + summary

    if "@Entity" not in combined:
        return None

    # Extract table name from @Entity("tablename") or default to snake_case of class name
    m = re.search(r'@Entity\s*\(\s*["\'](\w+)["\']', combined)
    table_name = m.group(1) if m else _snake_case(node.name)

    schema = TableSchema(
        table_name=table_name,
        source_node_id=node.id,
        language="typescript",
        file_path=node.file_path,
    )

    # Extract @Column() decorated properties
    lines = combined.splitlines()
    for i, line in enumerate(lines):
        if "@Column" not in line and "@PrimaryColumn" not in line and "@PrimaryGeneratedColumn" not in line:
            continue
        is_primary = "@PrimaryColumn" in line or "@PrimaryGeneratedColumn" in line
        # Next non-decorator, non-empty line is the property
        for j in range(i + 1, min(i + 3, len(lines))):
            prop_m = re.match(r"\s*(\w+)\s*[?!]?\s*:\s*([\w<>\[\]|]+)", lines[j])
            if prop_m:
                schema.columns.append(ColumnDef(
                    name=prop_m.group(1),
                    col_type=prop_m.group(2),
                    is_primary=is_primary,
                    is_nullable="?" in lines[j],
                ))
                break

    return schema


def _detect_gorm(node: NeuralNode) -> Optional[TableSchema]:
    """Detect Go GORM model (struct with gorm tags)."""
    raw = node.raw_code or ""
    summary = node.summary_short or ""
    combined = raw + "\n" + summary

    if 'gorm:"' not in combined:
        return None

    table_name = _snake_plural(node.name)

    schema = TableSchema(
        table_name=table_name,
        source_node_id=node.id,
        language="go",
        file_path=node.file_path,
    )

    # Parse struct fields with gorm tags
    field_pattern = re.compile(
        r"\s*(\w+)\s+(\w+).*?gorm:\"([^\"]*)\""
    )
    for line in combined.splitlines():
        m = field_pattern.match(line)
        if not m:
            continue
        field_name = m.group(1)
        go_type = m.group(2)
        tag = m.group(3)

        col_name_m = re.search(r"column:(\w+)", tag)
        col_name = col_name_m.group(1) if col_name_m else _snake_case(field_name)

        is_primary = "primaryKey" in tag or "primary_key" in tag
        is_nullable = "not null" not in tag

        schema.columns.append(ColumnDef(
            name=col_name,
            col_type=go_type,
            is_primary=is_primary,
            is_nullable=is_nullable,
        ))

    return schema


def _detect_diesel(node: NeuralNode) -> Optional[TableSchema]:
    """Detect Rust Diesel model."""
    raw = node.raw_code or ""
    summary = node.summary_short or ""
    combined = raw + "\n" + summary

    has_queryable = "#[derive(Queryable" in combined
    has_table_name = "table_name" in combined or "#[diesel(" in combined
    if not (has_queryable or has_table_name):
        return None

    # Extract table name from attribute
    m = re.search(r'#\[diesel\(table_name\s*=\s*(\w+)\)\]', combined)
    table_name = m.group(1) if m else _snake_case(node.name)

    schema = TableSchema(
        table_name=table_name,
        source_node_id=node.id,
        language="rust",
        file_path=node.file_path,
    )

    # Parse struct fields (simple: field_name: Type)
    field_pattern = re.compile(r"\s*pub\s+(\w+)\s*:\s*([\w<>]+)")
    for line in combined.splitlines():
        m = field_pattern.match(line)
        if not m:
            continue
        schema.columns.append(ColumnDef(
            name=m.group(1),
            col_type=m.group(2),
        ))

    return schema


# ---------------------------------------------------------------------------
# Node/edge builders
# ---------------------------------------------------------------------------

def _build_nodes_edges(
    schema: TableSchema,
    db_node_id: str,
) -> tuple[list[NeuralNode], list[NeuralEdge]]:
    """Convert a TableSchema into NeuralNodes + NeuralEdges (no FK REFERENCES yet)."""
    nodes: list[NeuralNode] = []
    edges: list[NeuralEdge] = []

    table_id = f"table::{schema.table_name}"
    table_node = NeuralNode(
        id=table_id,
        name=schema.table_name,
        node_type=NodeType.TABLE,
        file_path=schema.file_path,
        line_start=0,
        line_end=0,
        category="database",
        language=schema.language,
        summary_short=f"Database table: {schema.table_name}",
    )
    nodes.append(table_node)

    # DATABASE → TABLE
    edges.append(NeuralEdge(
        source_id=db_node_id,
        target_id=table_id,
        edge_type=EdgeType.CONTAINS,
    ))

    # Source class → TABLE (DEFINES relationship)
    if schema.source_node_id:
        edges.append(NeuralEdge(
            source_id=schema.source_node_id,
            target_id=table_id,
            edge_type=EdgeType.DEFINES,
        ))

    # COLUMN nodes
    for col in schema.columns:
        col_id = f"col::{schema.table_name}::{col.name}"
        col_node = NeuralNode(
            id=col_id,
            name=col.name,
            node_type=NodeType.COLUMN,
            file_path=schema.file_path,
            line_start=0,
            line_end=0,
            category="database",
            language=schema.language,
            signature=col.col_type,
            summary_short=f"Column {col.name} ({col.col_type})"
            + (" PK" if col.is_primary else "")
            + ("" if col.is_nullable else " NOT NULL"),
        )
        nodes.append(col_node)

        # TABLE → COLUMN
        edges.append(NeuralEdge(
            source_id=table_id,
            target_id=col_id,
            edge_type=EdgeType.CONTAINS,
        ))

    return nodes, edges


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_orm_models(
    all_nodes: dict[str, NeuralNode],
) -> tuple[list[NeuralNode], list[NeuralEdge]]:
    """Scan all_nodes for ORM patterns; return database schema nodes + edges.

    Detection order: SQLAlchemy → Django → TypeORM → GORM → Diesel.
    A second pass wires FK REFERENCES edges between COLUMN nodes.
    """
    detectors = [
        _detect_sqlalchemy,
        _detect_django,
        _detect_typeorm,
        _detect_gorm,
        _detect_diesel,
    ]

    # Group schemas by language for DB node creation
    schemas_by_language: dict[str, list[TableSchema]] = {}

    for node in all_nodes.values():
        if node.node_type != NodeType.CLASS:
            continue
        for detect_fn in detectors:
            schema = detect_fn(node)
            if schema is not None:
                lang = schema.language
                schemas_by_language.setdefault(lang, []).append(schema)
                break  # Only one ORM per class

    if not schemas_by_language:
        return [], []

    result_nodes: list[NeuralNode] = []
    result_edges: list[NeuralEdge] = []

    # Build per-language DATABASE nodes
    for language, schemas in schemas_by_language.items():
        db_node_id = f"db::orm::{language}"
        db_node = NeuralNode(
            id=db_node_id,
            name=f"ORM ({language})",
            node_type=NodeType.DATABASE,
            file_path="",
            line_start=0,
            line_end=0,
            category="database",
            language=language,
            summary_short=f"ORM-detected database schema for {language}",
        )
        result_nodes.append(db_node)

        for schema in schemas:
            nodes, edges = _build_nodes_edges(schema, db_node_id)
            result_nodes.extend(nodes)
            result_edges.extend(edges)

    # Second pass: wire FK REFERENCES edges
    # Build lookup: table_name → table_node_id
    table_id_map = {
        n.name: n.id
        for n in result_nodes
        if n.node_type == NodeType.TABLE
    }

    for schema_list in schemas_by_language.values():
        for schema in schema_list:
            for col in schema.columns:
                if not col.foreign_key:
                    continue
                parts = col.foreign_key.split(".", 1)
                ref_table = parts[0]
                ref_col = parts[1] if len(parts) > 1 else ""

                source_col_id = f"col::{schema.table_name}::{col.name}"
                if ref_col:
                    target_col_id = f"col::{ref_table}::{ref_col}"
                else:
                    target_col_id = table_id_map.get(ref_table, f"table::{ref_table}")

                result_edges.append(NeuralEdge(
                    source_id=source_col_id,
                    target_id=target_col_id,
                    edge_type=EdgeType.REFERENCES,
                    context=f"FK {schema.table_name}.{col.name} → {col.foreign_key}",
                ))

    return result_nodes, result_edges
