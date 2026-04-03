"""Schema indexer — convert TableSchema objects into NeuralNodes/NeuralEdges.

Used by both the live DB connector (connector.py) and the MCP tool
neural_index_db (server.py).
"""

from __future__ import annotations

from ..models import NeuralNode, NeuralEdge, NodeType, EdgeType
from ..storage import Storage
from . import TableSchema


def index_db_schema(
    storage: Storage,
    schemas: list[TableSchema],
    db_name: str,
    source: str = "live_db",
) -> dict:
    """Convert TableSchema list → NeuralNodes/NeuralEdges, store them, return stats.

    Args:
        storage:  Open Storage instance.
        schemas:  List of TableSchema objects (from connector or ORM detector).
        db_name:  Human-readable label for the database (used in node name).
        source:   Provenance tag (e.g. 'live_db', 'orm', 'sql_file').

    Returns:
        dict with keys: tables_indexed, columns_indexed, edges_created
    """
    nodes: list[NeuralNode] = []
    edges: list[NeuralEdge] = []

    db_id = f"db::{source}::{db_name}"
    db_node = NeuralNode(
        id=db_id,
        name=db_name,
        node_type=NodeType.DATABASE,
        file_path="",
        line_start=0,
        line_end=0,
        category="database",
        summary_short=f"Database: {db_name} (source: {source})",
    )
    nodes.append(db_node)

    # Track all table/column IDs for FK edge wiring
    all_tables: dict[str, str] = {}   # table_name → table_node_id
    fk_hints: list[tuple[str, str]] = []  # (source_col_id, "ref_table.ref_col")

    for schema in schemas:
        table_id = f"table::{db_name}::{schema.table_name}"
        all_tables[schema.table_name] = table_id

        table_node = NeuralNode(
            id=table_id,
            name=schema.table_name,
            node_type=NodeType.TABLE,
            file_path=schema.file_path,
            line_start=0,
            line_end=0,
            category="database",
            language=schema.language,
            summary_short=f"Table: {schema.table_name} ({len(schema.columns)} columns)",
        )
        nodes.append(table_node)

        # DATABASE → TABLE
        edges.append(NeuralEdge(
            source_id=db_id,
            target_id=table_id,
            edge_type=EdgeType.CONTAINS,
        ))

        # If the schema came from ORM detection, wire source class → table
        if schema.source_node_id:
            edges.append(NeuralEdge(
                source_id=schema.source_node_id,
                target_id=table_id,
                edge_type=EdgeType.DEFINES,
            ))

        # COLUMN nodes
        for col in schema.columns:
            col_id = f"col::{db_name}::{schema.table_name}::{col.name}"
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
                summary_short=(
                    f"Column {col.name} ({col.col_type})"
                    + (" PK" if col.is_primary else "")
                    + ("" if col.is_nullable else " NOT NULL")
                ),
            )
            nodes.append(col_node)

            # TABLE → COLUMN
            edges.append(NeuralEdge(
                source_id=table_id,
                target_id=col_id,
                edge_type=EdgeType.CONTAINS,
            ))

            if col.foreign_key:
                fk_hints.append((col_id, col.foreign_key))

    # Second pass: wire FK REFERENCES
    for source_col_id, fk_ref in fk_hints:
        parts = fk_ref.split(".", 1)
        ref_table = parts[0]
        ref_col = parts[1] if len(parts) > 1 else ""

        if ref_col and ref_table in all_tables:
            target_id = f"col::{db_name}::{ref_table}::{ref_col}"
        elif ref_table in all_tables:
            target_id = all_tables[ref_table]
        else:
            # Best-effort fallback
            target_id = f"table::{db_name}::{ref_table}"

        edges.append(NeuralEdge(
            source_id=source_col_id,
            target_id=target_id,
            edge_type=EdgeType.REFERENCES,
            context=f"FK → {fk_ref}",
        ))

    storage.batch_upsert_nodes(nodes)
    storage.batch_upsert_edges(edges)

    tables_indexed = sum(1 for n in nodes if n.node_type == NodeType.TABLE)
    columns_indexed = sum(1 for n in nodes if n.node_type == NodeType.COLUMN)
    fk_edges = sum(1 for e in edges if e.edge_type == EdgeType.REFERENCES)

    return {
        "tables_indexed": tables_indexed,
        "columns_indexed": columns_indexed,
        "edges_created": len(edges),
        "fk_edges": fk_edges,
    }
