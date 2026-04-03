"""Tests for the database connector (detection + schema fetching)."""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

from neural_memory.db.connector import detect_connection_string, fetch_schema
from neural_memory.db import TableSchema


# ---------------------------------------------------------------------------
# detect_connection_string
# ---------------------------------------------------------------------------

def test_detect_from_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///test.db")
    result = detect_connection_string(str(tmp_path))
    assert result == "sqlite:///test.db"


def test_detect_from_dot_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text('DATABASE_URL=postgresql://user:pass@localhost/mydb\n')
    result = detect_connection_string(str(tmp_path))
    assert result == "postgresql://user:pass@localhost/mydb"


def test_detect_from_dot_env_with_quotes(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text('DATABASE_URL="sqlite:///data.db"\n')
    result = detect_connection_string(str(tmp_path))
    assert result == "sqlite:///data.db"


def test_detect_returns_none_when_nothing_found(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DB_URL", raising=False)
    result = detect_connection_string(str(tmp_path))
    assert result is None


def test_detect_env_var_takes_priority_over_file(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///env_wins.db")
    env_file = tmp_path / ".env"
    env_file.write_text('DATABASE_URL=sqlite:///file.db\n')
    result = detect_connection_string(str(tmp_path))
    assert result == "sqlite:///env_wins.db"


# ---------------------------------------------------------------------------
# fetch_schema — SQLite
# ---------------------------------------------------------------------------

@pytest.fixture
def sqlite_db(tmp_path):
    """Create a SQLite database with two tables and a FK relationship."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT
        );

        CREATE TABLE posts (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)
    conn.close()
    return db_path


def test_fetch_schema_sqlite_returns_tables(sqlite_db):
    cs = f"sqlite:///{sqlite_db}"
    schemas = fetch_schema(cs)

    assert len(schemas) == 2
    table_names = {s.table_name for s in schemas}
    assert "users" in table_names
    assert "posts" in table_names


def test_fetch_schema_sqlite_columns(sqlite_db):
    cs = f"sqlite:///{sqlite_db}"
    schemas = fetch_schema(cs)
    users = next(s for s in schemas if s.table_name == "users")

    col_names = {c.name for c in users.columns}
    assert "id" in col_names
    assert "name" in col_names
    assert "email" in col_names


def test_fetch_schema_sqlite_primary_key(sqlite_db):
    cs = f"sqlite:///{sqlite_db}"
    schemas = fetch_schema(cs)
    users = next(s for s in schemas if s.table_name == "users")

    id_col = next(c for c in users.columns if c.name == "id")
    assert id_col.is_primary is True


def test_fetch_schema_sqlite_foreign_key(sqlite_db):
    cs = f"sqlite:///{sqlite_db}"
    schemas = fetch_schema(cs)
    posts = next(s for s in schemas if s.table_name == "posts")

    user_id_col = next(c for c in posts.columns if c.name == "user_id")
    assert user_id_col.foreign_key == "users.id"


def test_fetch_schema_sqlite_returns_tableschema_objects(sqlite_db):
    cs = f"sqlite:///{sqlite_db}"
    schemas = fetch_schema(cs)
    for s in schemas:
        assert isinstance(s, TableSchema)
        assert s.table_name
        assert isinstance(s.columns, list)


# ---------------------------------------------------------------------------
# fetch_schema — unsupported scheme raises ValueError
# ---------------------------------------------------------------------------

def test_fetch_schema_unsupported_scheme_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        fetch_schema("mongodb://localhost/mydb")


def test_fetch_schema_connection_string_parsing():
    """Verify connection string scheme routing (sqlite only, no real DB needed)."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        # Create an empty DB
        conn = sqlite3.connect(db_path)
        conn.close()
        schemas = fetch_schema(f"sqlite:///{db_path}")
        assert isinstance(schemas, list)
    finally:
        os.unlink(db_path)
