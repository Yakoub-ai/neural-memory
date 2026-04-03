"""Tests for the Go tree-sitter parser."""

import pytest
from neural_memory.models import NodeType, EdgeType
from neural_memory.parsers.registry import get_parser


GO_SOURCE = """\
package main

import (
\t"fmt"
\t"github.com/myapp/db"
)

type User struct {
\tID   int
\tName string
}

type Repository interface {
\tFind(id int) (User, error)
\tSave(u User) error
}

type UserID int

func NewUser(name string) User {
\treturn User{Name: name}
}

func (u User) String() string {
\treturn fmt.Sprintf("User(%d, %s)", u.ID, u.Name)
}
"""


@pytest.fixture
def go_parser():
    return get_parser("main.go")


def test_go_module_node(go_parser):
    nodes, _ = go_parser.parse_file("main.go", source=GO_SOURCE)
    modules = [n for n in nodes if n.node_type == NodeType.MODULE]
    assert len(modules) == 1
    assert modules[0].language == "go"


def test_go_struct_extracted(go_parser):
    nodes, _ = go_parser.parse_file("main.go", source=GO_SOURCE)
    classes = [n for n in nodes if n.node_type == NodeType.CLASS]
    names = {n.name for n in classes}
    assert "User" in names
    assert "Repository" in names


def test_go_type_alias(go_parser):
    nodes, _ = go_parser.parse_file("main.go", source=GO_SOURCE)
    typedefs = [n for n in nodes if n.node_type == NodeType.TYPE_DEF]
    assert any(n.name == "UserID" for n in typedefs)


def test_go_functions(go_parser):
    nodes, _ = go_parser.parse_file("main.go", source=GO_SOURCE)
    funcs = [n for n in nodes if n.node_type == NodeType.FUNCTION]
    assert any(n.name == "NewUser" for n in funcs)


def test_go_method(go_parser):
    nodes, _ = go_parser.parse_file("main.go", source=GO_SOURCE)
    methods = [n for n in nodes if n.node_type == NodeType.METHOD]
    assert any(n.name == "String" for n in methods)


def test_go_import_edges(go_parser):
    nodes, edges = go_parser.parse_file("main.go", source=GO_SOURCE)
    import_edges = [e for e in edges if e.edge_type == EdgeType.IMPORTS]
    assert len(import_edges) >= 1


def test_go_language_set(go_parser):
    nodes, _ = go_parser.parse_file("main.go", source=GO_SOURCE)
    for node in nodes:
        assert node.language == "go"


def test_go_empty_file(go_parser):
    nodes, edges = go_parser.parse_file("empty.go", source="package main\n")
    assert any(n.node_type == NodeType.MODULE for n in nodes)
