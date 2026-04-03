"""Tests for ORM model detection (SQLAlchemy, Django, TypeORM, GORM, Diesel)."""

from __future__ import annotations

import pytest

from neural_memory.models import NeuralNode, NodeType, EdgeType
from neural_memory.db.orm_detector import detect_orm_models


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_class_node(
    node_id: str,
    name: str,
    raw_code: str = "",
    file_path: str = "models.py",
    language: str = "python",
) -> NeuralNode:
    return NeuralNode(
        id=node_id,
        name=name,
        node_type=NodeType.CLASS,
        file_path=file_path,
        line_start=1,
        line_end=20,
        raw_code=raw_code,
        language=language,
    )


# ---------------------------------------------------------------------------
# SQLAlchemy
# ---------------------------------------------------------------------------

SQLALCHEMY_SOURCE = '''
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    email = Column(String)
'''

SQLALCHEMY_USER_SOURCE = '''
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    email = Column(String)
'''

SQLALCHEMY_POST_SOURCE = '''
class Post(Base):
    __tablename__ = "posts"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String)
'''


def test_sqlalchemy_detection_creates_table_and_columns():
    nodes = {
        "User": _make_class_node("User", "User", SQLALCHEMY_USER_SOURCE),
    }
    db_nodes, db_edges = detect_orm_models(nodes)

    node_types = {n.node_type for n in db_nodes}
    assert NodeType.DATABASE in node_types
    assert NodeType.TABLE in node_types
    assert NodeType.COLUMN in node_types

    table_nodes = [n for n in db_nodes if n.node_type == NodeType.TABLE]
    assert len(table_nodes) == 1
    assert table_nodes[0].name == "users"

    col_names = {n.name for n in db_nodes if n.node_type == NodeType.COLUMN}
    assert "id" in col_names
    assert "name" in col_names
    assert "email" in col_names


def test_sqlalchemy_primary_key_detected():
    nodes = {
        "User": _make_class_node("User", "User", SQLALCHEMY_USER_SOURCE),
    }
    db_nodes, _ = detect_orm_models(nodes)
    col_nodes = {n.name: n for n in db_nodes if n.node_type == NodeType.COLUMN}
    # id column should have PK in summary
    assert "PK" in col_nodes["id"].summary_short


def test_sqlalchemy_foreign_key_creates_references_edge():
    nodes = {
        "User": _make_class_node("User", "User", SQLALCHEMY_USER_SOURCE),
        "Post": _make_class_node("Post", "Post", SQLALCHEMY_POST_SOURCE),
    }
    db_nodes, db_edges = detect_orm_models(nodes)

    ref_edges = [e for e in db_edges if e.edge_type == EdgeType.REFERENCES]
    assert len(ref_edges) >= 1

    # Source should be Post.user_id column
    source_ids = {e.source_id for e in ref_edges}
    assert any("user_id" in sid for sid in source_ids)


def test_sqlalchemy_defines_edge_from_class_to_table():
    nodes = {
        "User": _make_class_node("User", "User", SQLALCHEMY_USER_SOURCE),
    }
    db_nodes, db_edges = detect_orm_models(nodes)

    defines_edges = [e for e in db_edges if e.edge_type == EdgeType.DEFINES]
    assert len(defines_edges) >= 1
    assert defines_edges[0].source_id == "User"


# ---------------------------------------------------------------------------
# Django
# ---------------------------------------------------------------------------

DJANGO_SOURCE = '''
from django.db import models

class Article(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    published = models.BooleanField(default=False)
'''


def test_django_detection_creates_table():
    nodes = {
        "Article": _make_class_node("Article", "Article", DJANGO_SOURCE),
    }
    db_nodes, _ = detect_orm_models(nodes)

    table_nodes = [n for n in db_nodes if n.node_type == NodeType.TABLE]
    assert len(table_nodes) == 1
    # Django snake_cases class name for table
    assert table_nodes[0].name == "article"


def test_django_columns_detected():
    nodes = {
        "Article": _make_class_node("Article", "Article", DJANGO_SOURCE),
    }
    db_nodes, _ = detect_orm_models(nodes)
    col_names = {n.name for n in db_nodes if n.node_type == NodeType.COLUMN}
    assert "title" in col_names
    assert "content" in col_names


# ---------------------------------------------------------------------------
# TypeORM
# ---------------------------------------------------------------------------

TYPEORM_SOURCE = '''
import { Entity, Column, PrimaryGeneratedColumn } from "typeorm";

@Entity("products")
export class Product {
    @PrimaryGeneratedColumn()
    id: number;

    @Column()
    name: string;

    @Column({ nullable: true })
    description?: string;
}
'''


def test_typeorm_detection_creates_table():
    nodes = {
        "Product": _make_class_node(
            "Product", "Product", TYPEORM_SOURCE,
            file_path="product.ts", language="typescript"
        ),
    }
    db_nodes, _ = detect_orm_models(nodes)

    table_nodes = [n for n in db_nodes if n.node_type == NodeType.TABLE]
    assert len(table_nodes) == 1
    assert table_nodes[0].name == "products"


def test_typeorm_columns_detected():
    nodes = {
        "Product": _make_class_node(
            "Product", "Product", TYPEORM_SOURCE,
            file_path="product.ts", language="typescript"
        ),
    }
    db_nodes, _ = detect_orm_models(nodes)
    col_names = {n.name for n in db_nodes if n.node_type == NodeType.COLUMN}
    assert "id" in col_names
    assert "name" in col_names


# ---------------------------------------------------------------------------
# Negative case — plain Python class should NOT produce TABLE nodes
# ---------------------------------------------------------------------------

PLAIN_CLASS_SOURCE = '''
class MyHelper:
    def __init__(self):
        self.value = 42

    def compute(self):
        return self.value * 2
'''


def test_plain_class_generates_no_db_nodes():
    nodes = {
        "MyHelper": _make_class_node("MyHelper", "MyHelper", PLAIN_CLASS_SOURCE),
    }
    db_nodes, db_edges = detect_orm_models(nodes)
    assert db_nodes == []
    assert db_edges == []


def test_empty_all_nodes_returns_empty():
    db_nodes, db_edges = detect_orm_models({})
    assert db_nodes == []
    assert db_edges == []
