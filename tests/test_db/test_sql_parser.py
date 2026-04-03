"""Tests for the SQL/Prisma parser."""

from __future__ import annotations

import pytest

from neural_memory.models import NodeType, EdgeType
from neural_memory.parsers.languages.sql import SQLParser


PARSER = SQLParser()


# ---------------------------------------------------------------------------
# SQL DDL tests
# ---------------------------------------------------------------------------

SIMPLE_SQL = """
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email TEXT
);
"""

SQL_WITH_FK = """
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100) NOT NULL
);

CREATE TABLE posts (
    id INTEGER PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    title TEXT NOT NULL
);
"""

SQL_WITH_CONSTRAINT = """
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    total DECIMAL NOT NULL,
    status VARCHAR(50)
);
"""


def test_sql_parser_language_id():
    assert PARSER.language_id == "sql"


def test_sql_parser_extensions():
    assert ".sql" in PARSER.file_extensions
    assert ".prisma" in PARSER.file_extensions


def test_sql_create_table_basic():
    nodes, edges = PARSER.parse_file("schema.sql", source=SIMPLE_SQL)

    node_types = {n.node_type for n in nodes}
    assert NodeType.DATABASE in node_types
    assert NodeType.TABLE in node_types
    assert NodeType.COLUMN in node_types

    table_nodes = [n for n in nodes if n.node_type == NodeType.TABLE]
    assert len(table_nodes) == 1
    assert table_nodes[0].name == "users"


def test_sql_columns_parsed():
    nodes, _ = PARSER.parse_file("schema.sql", source=SIMPLE_SQL)
    col_names = {n.name for n in nodes if n.node_type == NodeType.COLUMN}
    assert "id" in col_names
    assert "name" in col_names
    assert "email" in col_names


def test_sql_primary_key_in_summary():
    nodes, _ = PARSER.parse_file("schema.sql", source=SIMPLE_SQL)
    col_map = {n.name: n for n in nodes if n.node_type == NodeType.COLUMN}
    assert "PK" in col_map["id"].summary_short


def test_sql_foreign_key_creates_references_edge():
    nodes, edges = PARSER.parse_file("schema.sql", source=SQL_WITH_FK)

    ref_edges = [e for e in edges if e.edge_type == EdgeType.REFERENCES]
    assert len(ref_edges) >= 1

    # user_id column in posts should reference users.id
    source_ids = {e.source_id for e in ref_edges}
    assert any("user_id" in sid for sid in source_ids)


def test_sql_multiple_tables():
    nodes, edges = PARSER.parse_file("schema.sql", source=SQL_WITH_FK)
    table_nodes = [n for n in nodes if n.node_type == NodeType.TABLE]
    assert len(table_nodes) == 2
    table_names = {t.name for t in table_nodes}
    assert "users" in table_names
    assert "posts" in table_names


def test_sql_contains_edges_exist():
    nodes, edges = PARSER.parse_file("schema.sql", source=SIMPLE_SQL)
    contains_edges = [e for e in edges if e.edge_type == EdgeType.CONTAINS]
    # At least: db→table + table→columns
    assert len(contains_edges) >= 4  # 1 db→table + 3 table→col


def test_sql_empty_file_returns_empty():
    nodes, edges = PARSER.parse_file("empty.sql", source="")
    assert nodes == []
    assert edges == []


def test_sql_empty_string_whitespace_returns_empty():
    nodes, edges = PARSER.parse_file("empty.sql", source="   \n  ")
    assert nodes == []
    assert edges == []


# ---------------------------------------------------------------------------
# Prisma tests
# ---------------------------------------------------------------------------

PRISMA_SCHEMA = """
datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

model User {
  id        Int      @id @default(autoincrement())
  email     String   @unique
  name      String?
  posts     Post[]
}

model Post {
  id        Int      @id @default(autoincrement())
  title     String
  content   String?
  author    User     @relation(fields: [authorId], references: [id])
  authorId  Int
}
"""


def test_prisma_models_parsed():
    nodes, edges = PARSER.parse_file("schema.prisma", source=PRISMA_SCHEMA)

    table_nodes = [n for n in nodes if n.node_type == NodeType.TABLE]
    assert len(table_nodes) >= 2
    names = {t.name for t in table_nodes}
    assert "user" in names
    assert "post" in names


def test_prisma_columns_parsed():
    nodes, edges = PARSER.parse_file("schema.prisma", source=PRISMA_SCHEMA)

    # Check User columns (excluding relation fields)
    user_cols = {
        n.name for n in nodes
        if n.node_type == NodeType.COLUMN and "user" in n.id
    }
    assert "id" in user_cols
    assert "email" in user_cols


def test_prisma_pk_column_detected():
    nodes, _ = PARSER.parse_file("schema.prisma", source=PRISMA_SCHEMA)
    id_cols = [
        n for n in nodes
        if n.node_type == NodeType.COLUMN and n.name == "id"
    ]
    assert len(id_cols) >= 1
    assert any("PK" in c.summary_short for c in id_cols)


def test_prisma_database_node_present():
    nodes, _ = PARSER.parse_file("schema.prisma", source=PRISMA_SCHEMA)
    db_nodes = [n for n in nodes if n.node_type == NodeType.DATABASE]
    assert len(db_nodes) == 1


def test_prisma_empty_file_returns_empty():
    nodes, edges = PARSER.parse_file("empty.prisma", source="")
    assert nodes == []
    assert edges == []
