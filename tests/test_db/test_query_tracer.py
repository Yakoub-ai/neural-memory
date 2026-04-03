"""Tests for the query tracer — QUERIES and WRITES_TO edge generation."""

import pytest
from neural_memory.models import NeuralNode, NeuralEdge, NodeType, EdgeType
from neural_memory.db.query_tracer import trace_queries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fn(node_id: str, raw_code: str, language: str = "python") -> NeuralNode:
    return NeuralNode(
        id=node_id, name=node_id, node_type=NodeType.FUNCTION,
        file_path="app.py", line_start=1, line_end=10,
        raw_code=raw_code, language=language,
    )


def _method(node_id: str, raw_code: str, language: str = "python") -> NeuralNode:
    return NeuralNode(
        id=node_id, name=node_id, node_type=NodeType.METHOD,
        file_path="app.py", line_start=1, line_end=10,
        raw_code=raw_code, language=language,
    )


def _cls(node_id: str, name: str) -> NeuralNode:
    return NeuralNode(
        id=node_id, name=name, node_type=NodeType.CLASS,
        file_path="models.py", line_start=1, line_end=20,
        raw_code="", language="python",
    )


def _table(name: str) -> NeuralNode:
    return NeuralNode(
        id=f"table::{name}", name=name, node_type=NodeType.TABLE,
        file_path="models.py", line_start=1, line_end=1,
        raw_code="", category="database",
    )


def _defines_edge(cls_id: str, table_name: str) -> NeuralEdge:
    return NeuralEdge(
        source_id=cls_id, target_id=f"table::{table_name}",
        edge_type=EdgeType.DEFINES,
    )


def _edge_types(edges: list[NeuralEdge]) -> list[str]:
    return sorted(f"{e.source_id}->{e.target_id}:{e.edge_type.value}" for e in edges)


# ---------------------------------------------------------------------------
# No tables → no edges
# ---------------------------------------------------------------------------

def test_no_tables_returns_empty():
    fn = _fn("get_user", "session.query(User).filter_by(id=1).first()")
    assert trace_queries({"get_user": fn}, []) == []


# ---------------------------------------------------------------------------
# SQLAlchemy
# ---------------------------------------------------------------------------

def test_sqlalchemy_session_query_produces_queries_edge():
    nodes = {
        "UserModel": _cls("UserModel", "User"),
        "table::users": _table("users"),
        "get_user": _fn("get_user", "return session.query(User).filter_by(id=1).first()"),
    }
    edges = [_defines_edge("UserModel", "users")]
    result = trace_queries(nodes, edges)
    assert any(e.edge_type == EdgeType.QUERIES and e.target_id == "table::users" for e in result)


def test_sqlalchemy_dot_query_filter_produces_queries_edge():
    nodes = {
        "OrderModel": _cls("OrderModel", "Order"),
        "table::orders": _table("orders"),
        "list_orders": _fn("list_orders", "return Order.query.filter(Order.status == 'open').all()"),
    }
    edges = [_defines_edge("OrderModel", "orders")]
    result = trace_queries(nodes, edges)
    assert any(e.edge_type == EdgeType.QUERIES and e.source_id == "list_orders" for e in result)


def test_sqlalchemy_session_add_produces_writes_to_edge():
    nodes = {
        "UserModel": _cls("UserModel", "User"),
        "table::users": _table("users"),
        "create_user": _fn("create_user", "user = User(name='Alice')\nsession.add(user)\nsession.commit()"),
    }
    edges = [_defines_edge("UserModel", "users")]
    result = trace_queries(nodes, edges)
    assert any(e.edge_type == EdgeType.WRITES_TO and e.target_id == "table::users" for e in result)


def test_sqlalchemy_session_delete_produces_writes_to_edge():
    nodes = {
        "UserModel": _cls("UserModel", "User"),
        "table::users": _table("users"),
        "delete_user": _fn("delete_user", "session.delete(user)\nsession.commit()"),
    }
    edges = [_defines_edge("UserModel", "users")]
    result = trace_queries(nodes, edges)
    assert any(e.edge_type == EdgeType.WRITES_TO and e.source_id == "delete_user" for e in result)


def test_sqlalchemy_select_produces_queries_edge():
    nodes = {
        "PostModel": _cls("PostModel", "Post"),
        "table::posts": _table("posts"),
        "get_posts": _fn("get_posts", "stmt = select(Post).where(Post.published == True)"),
    }
    edges = [_defines_edge("PostModel", "posts")]
    result = trace_queries(nodes, edges)
    assert any(e.edge_type == EdgeType.QUERIES and e.target_id == "table::posts" for e in result)


# ---------------------------------------------------------------------------
# Django
# ---------------------------------------------------------------------------

def test_django_objects_filter_produces_queries_edge():
    nodes = {
        "UserModel": _cls("UserModel", "User"),
        "table::users": _table("users"),
        "search_users": _fn("search_users", "return User.objects.filter(active=True)"),
    }
    edges = [_defines_edge("UserModel", "users")]
    result = trace_queries(nodes, edges)
    assert any(e.edge_type == EdgeType.QUERIES and e.target_id == "table::users" for e in result)


def test_django_objects_create_produces_writes_to_edge():
    nodes = {
        "OrderModel": _cls("OrderModel", "Order"),
        "table::orders": _table("orders"),
        "place_order": _fn("place_order", "return Order.objects.create(user=user, total=100)"),
    }
    edges = [_defines_edge("OrderModel", "orders")]
    result = trace_queries(nodes, edges)
    assert any(e.edge_type == EdgeType.WRITES_TO and e.target_id == "table::orders" for e in result)


def test_django_objects_all_produces_queries_edge():
    nodes = {
        "ProductModel": _cls("ProductModel", "Product"),
        "table::products": _table("products"),
        "list_products": _fn("list_products", "qs = Product.objects.all().order_by('name')"),
    }
    edges = [_defines_edge("ProductModel", "products")]
    result = trace_queries(nodes, edges)
    assert any(e.edge_type == EdgeType.QUERIES and e.target_id == "table::products" for e in result)


def test_django_objects_delete_produces_writes_to_edge():
    nodes = {
        "SessionModel": _cls("SessionModel", "Session"),
        "table::sessions": _table("sessions"),
        "purge_sessions": _fn("purge_sessions", "Session.objects.delete(expired=True)"),
    }
    edges = [_defines_edge("SessionModel", "sessions")]
    result = trace_queries(nodes, edges)
    assert any(e.edge_type == EdgeType.WRITES_TO and e.target_id == "table::sessions" for e in result)


# ---------------------------------------------------------------------------
# Raw SQL
# ---------------------------------------------------------------------------

def test_raw_sql_select_produces_queries_edge():
    nodes = {
        "table::payments": _table("payments"),
        "get_payments": _fn("get_payments", 'rows = db.execute("SELECT id, amount FROM payments WHERE user_id = ?", [uid])'),
    }
    result = trace_queries(nodes, [])
    assert any(e.edge_type == EdgeType.QUERIES and e.target_id == "table::payments" for e in result)


def test_raw_sql_insert_produces_writes_to_edge():
    nodes = {
        "table::audit_log": _table("audit_log"),
        "log_event": _fn("log_event", 'db.execute("INSERT INTO audit_log (event, ts) VALUES (?, ?)", [evt, ts])'),
    }
    result = trace_queries(nodes, [])
    assert any(e.edge_type == EdgeType.WRITES_TO and e.target_id == "table::audit_log" for e in result)


def test_raw_sql_update_produces_writes_to_edge():
    nodes = {
        "table::users": _table("users"),
        "update_email": _fn("update_email", 'db.execute("UPDATE users SET email = ? WHERE id = ?", [email, uid])'),
    }
    result = trace_queries(nodes, [])
    assert any(e.edge_type == EdgeType.WRITES_TO and e.target_id == "table::users" for e in result)


def test_raw_sql_delete_produces_writes_to_edge():
    nodes = {
        "table::tokens": _table("tokens"),
        "revoke_token": _fn("revoke_token", 'cur.execute("DELETE FROM tokens WHERE value = %s", [tok])'),
    }
    result = trace_queries(nodes, [])
    assert any(e.edge_type == EdgeType.WRITES_TO and e.target_id == "table::tokens" for e in result)


def test_raw_sql_table_not_in_graph_ignored():
    nodes = {
        "table::users": _table("users"),
        "mystery": _fn("mystery", 'db.execute("SELECT * FROM unknown_table")'),
    }
    result = trace_queries(nodes, [])
    assert result == []


# ---------------------------------------------------------------------------
# GORM (Go)
# ---------------------------------------------------------------------------

def test_gorm_find_produces_queries_edge():
    nodes = {
        "UserModel": _cls("UserModel", "User"),
        "table::users": _table("users"),
        "get_user": _fn("get_user", "db.Find(&user, id)", language="go"),
    }
    edges = [_defines_edge("UserModel", "users")]
    result = trace_queries(nodes, edges)
    assert any(e.edge_type == EdgeType.QUERIES and e.target_id == "table::users" for e in result)


def test_gorm_create_produces_writes_to_edge():
    nodes = {
        "UserModel": _cls("UserModel", "User"),
        "table::users": _table("users"),
        "create_user": _fn("create_user", "result := db.Create(&user)", language="go"),
    }
    edges = [_defines_edge("UserModel", "users")]
    result = trace_queries(nodes, edges)
    assert any(e.edge_type == EdgeType.WRITES_TO and e.target_id == "table::users" for e in result)


# ---------------------------------------------------------------------------
# Diesel (Rust)
# ---------------------------------------------------------------------------

def test_diesel_filter_produces_queries_edge():
    nodes = {
        "table::posts": _table("posts"),
        "get_posts": _fn("get_posts", "posts::table.filter(posts::published.eq(true)).load::<Post>(&conn)", language="rust"),
    }
    result = trace_queries(nodes, [])
    assert any(e.edge_type == EdgeType.QUERIES and e.target_id == "table::posts" for e in result)


def test_diesel_insert_produces_writes_to_edge():
    nodes = {
        "table::comments": _table("comments"),
        "add_comment": _fn("add_comment", "diesel::insert_into(comments::table).values(&new_comment).execute(&conn)", language="rust"),
    }
    result = trace_queries(nodes, [])
    assert any(e.edge_type == EdgeType.WRITES_TO and e.target_id == "table::comments" for e in result)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def test_duplicate_calls_produce_single_edge():
    """Same function calling the same table twice should produce only one edge."""
    nodes = {
        "UserModel": _cls("UserModel", "User"),
        "table::users": _table("users"),
        "double_query": _fn(
            "double_query",
            "a = session.query(User).first()\nb = session.query(User).count()",
        ),
    }
    edges = [_defines_edge("UserModel", "users")]
    result = trace_queries(nodes, edges)
    queries_to_users = [e for e in result if e.edge_type == EdgeType.QUERIES and e.target_id == "table::users"]
    assert len(queries_to_users) == 1


# ---------------------------------------------------------------------------
# Methods are traced same as functions
# ---------------------------------------------------------------------------

def test_method_node_is_traced():
    nodes = {
        "UserModel": _cls("UserModel", "User"),
        "table::users": _table("users"),
        "UserService.get_all": _method("UserService.get_all", "return User.objects.all()"),
    }
    edges = [_defines_edge("UserModel", "users")]
    result = trace_queries(nodes, edges)
    assert any(e.source_id == "UserService.get_all" and e.edge_type == EdgeType.QUERIES for e in result)


# ---------------------------------------------------------------------------
# Non-function nodes are ignored
# ---------------------------------------------------------------------------

def test_class_nodes_are_not_traced():
    nodes = {
        "table::users": _table("users"),
        "SomeClass": NeuralNode(
            id="SomeClass", name="SomeClass", node_type=NodeType.CLASS,
            file_path="app.py", line_start=1, line_end=10,
            raw_code='db.execute("SELECT * FROM users")',
        ),
    }
    result = trace_queries(nodes, [])
    assert result == []
