"""Tests for the Rust tree-sitter parser."""

import pytest
from neural_memory.models import NodeType, EdgeType
from neural_memory.parsers.registry import get_parser


RUST_SOURCE = """\
use std::fmt;
use crate::db::Database;

pub struct User {
    pub id: i64,
    pub name: String,
}

impl User {
    pub fn new(id: i64, name: String) -> Self {
        User { id, name }
    }

    pub fn greet(&self) -> String {
        format!("Hi {}", self.name)
    }
}

pub trait Repository {
    fn find(&self, id: i64) -> Option<User>;
    fn save(&self, user: &User) -> Result<(), Box<dyn std::error::Error>>;
}

pub enum Status {
    Active,
    Inactive,
}

pub fn create_user(name: &str) -> User {
    User::new(0, name.to_string())
}
"""


@pytest.fixture
def rs_parser():
    return get_parser("lib.rs")


def test_rust_module_node(rs_parser):
    nodes, _ = rs_parser.parse_file("lib.rs", source=RUST_SOURCE)
    modules = [n for n in nodes if n.node_type == NodeType.MODULE]
    assert len(modules) == 1
    assert modules[0].language == "rust"


def test_rust_struct_extracted(rs_parser):
    nodes, _ = rs_parser.parse_file("lib.rs", source=RUST_SOURCE)
    classes = [n for n in nodes if n.node_type == NodeType.CLASS]
    names = {n.name for n in classes}
    assert "User" in names
    assert "Repository" in names


def test_rust_impl_methods(rs_parser):
    nodes, _ = rs_parser.parse_file("lib.rs", source=RUST_SOURCE)
    methods = [n for n in nodes if n.node_type == NodeType.METHOD]
    method_names = {n.name for n in methods}
    assert "User.new" in method_names
    assert "User.greet" in method_names


def test_rust_enum(rs_parser):
    nodes, _ = rs_parser.parse_file("lib.rs", source=RUST_SOURCE)
    typedefs = [n for n in nodes if n.node_type == NodeType.TYPE_DEF]
    assert any(n.name == "Status" for n in typedefs)


def test_rust_function(rs_parser):
    nodes, _ = rs_parser.parse_file("lib.rs", source=RUST_SOURCE)
    funcs = [n for n in nodes if n.node_type == NodeType.FUNCTION]
    assert any(n.name == "create_user" for n in funcs)


def test_rust_import_edges(rs_parser):
    nodes, edges = rs_parser.parse_file("lib.rs", source=RUST_SOURCE)
    import_edges = [e for e in edges if e.edge_type == EdgeType.IMPORTS]
    assert len(import_edges) >= 1


def test_rust_language_set(rs_parser):
    nodes, _ = rs_parser.parse_file("lib.rs", source=RUST_SOURCE)
    for node in nodes:
        assert node.language == "rust"


def test_rust_empty_file(rs_parser):
    nodes, _ = rs_parser.parse_file("empty.rs", source="")
    assert any(n.node_type == NodeType.MODULE for n in nodes)
