"""Tests for the TypeScript/JavaScript tree-sitter parser."""

import pytest
from neural_memory.models import NodeType, EdgeType
from neural_memory.parsers.registry import get_parser


TS_SOURCE = """\
import { Database } from './db';
import axios from 'axios';

export interface UserShape {
    id: number;
    name: string;
}

export class UserService {
    constructor(private db: Database) {}

    async findUser(id: number): Promise<UserShape | null> {
        return this.db.query(id);
    }

    static create(db: Database): UserService {
        return new UserService(db);
    }
}

export function greet(name: string): string {
    return 'Hello ' + name;
}

type UserId = number;
"""

JS_SOURCE = """\
import { db } from './db.js';

class OrderService {
    placeOrder(item) {
        return db.save(item);
    }
}

function computeTotal(items) {
    return items.reduce((a, b) => a + b, 0);
}
"""


@pytest.fixture
def ts_parser():
    return get_parser("foo.ts")


@pytest.fixture
def js_parser():
    return get_parser("foo.js")


def test_ts_module_node(ts_parser):
    nodes, _ = ts_parser.parse_file("svc.ts", source=TS_SOURCE)
    modules = [n for n in nodes if n.node_type == NodeType.MODULE]
    assert len(modules) == 1
    assert modules[0].language == "typescript"
    assert modules[0].name == "svc"


def test_ts_class_extracted(ts_parser):
    nodes, _ = ts_parser.parse_file("svc.ts", source=TS_SOURCE)
    classes = [n for n in nodes if n.node_type == NodeType.CLASS]
    names = {n.name for n in classes}
    assert "UserService" in names
    assert "UserShape" in names


def test_ts_methods_extracted(ts_parser):
    nodes, _ = ts_parser.parse_file("svc.ts", source=TS_SOURCE)
    methods = [n for n in nodes if n.node_type == NodeType.METHOD]
    method_names = {n.name for n in methods}
    assert "UserService.findUser" in method_names
    assert "UserService.create" in method_names


def test_ts_function_extracted(ts_parser):
    nodes, _ = ts_parser.parse_file("svc.ts", source=TS_SOURCE)
    functions = [n for n in nodes if n.node_type == NodeType.FUNCTION]
    assert any(n.name == "greet" for n in functions)


def test_ts_type_alias_extracted(ts_parser):
    nodes, _ = ts_parser.parse_file("svc.ts", source=TS_SOURCE)
    typedefs = [n for n in nodes if n.node_type == NodeType.TYPE_DEF]
    assert any(n.name == "UserId" for n in typedefs)


def test_ts_import_edges(ts_parser):
    nodes, edges = ts_parser.parse_file("svc.ts", source=TS_SOURCE)
    import_edges = [e for e in edges if e.edge_type == EdgeType.IMPORTS]
    assert len(import_edges) >= 2


def test_ts_contains_edges(ts_parser):
    nodes, edges = ts_parser.parse_file("svc.ts", source=TS_SOURCE)
    contains = [e for e in edges if e.edge_type == EdgeType.CONTAINS]
    assert len(contains) >= 4  # module->class, class->methods, module->function


def test_ts_language_set(ts_parser):
    nodes, _ = ts_parser.parse_file("svc.ts", source=TS_SOURCE)
    for node in nodes:
        assert node.language == "typescript"


def test_ts_line_numbers(ts_parser):
    nodes, _ = ts_parser.parse_file("svc.ts", source=TS_SOURCE)
    svc = next(n for n in nodes if n.name == "UserService")
    assert svc.line_start > 0
    assert svc.line_end >= svc.line_start


def test_js_class_and_function(js_parser):
    nodes, _ = js_parser.parse_file("app.js", source=JS_SOURCE)
    names = {n.name for n in nodes}
    assert "OrderService" in names
    assert "computeTotal" in names
    for n in nodes:
        assert n.language == "javascript"


def test_ts_empty_file(ts_parser):
    nodes, edges = ts_parser.parse_file("empty.ts", source="")
    assert len(nodes) == 1  # just the module node
    assert nodes[0].node_type == NodeType.MODULE
