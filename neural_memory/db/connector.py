"""Database connector — detect connection strings and fetch live schemas.

Supports:
- SQLite (via stdlib sqlite3 — no extra deps)
- PostgreSQL (via psycopg2 — optional)
- MySQL (via pymysql — optional)
"""

from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path
from typing import Optional

from . import ColumnDef, TableSchema


# ---------------------------------------------------------------------------
# Connection string detection
# ---------------------------------------------------------------------------

def detect_connection_string(project_root: str = ".") -> Optional[str]:
    """Detect a database connection string from the environment or project files.

    Check order:
    1. DATABASE_URL environment variable
    2. .env file in project_root
    3. docker-compose.yml in project_root
    """
    # 1. Environment variable
    env_url = os.environ.get("DATABASE_URL")
    if env_url:
        return env_url

    root = Path(project_root)

    # 2. .env file
    env_file = root / ".env"
    if env_file.exists():
        try:
            for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                if key.strip().upper() in ("DATABASE_URL", "DB_URL", "DATABASE_URI"):
                    url = value.strip().strip('"').strip("'")
                    if url:
                        return url
        except OSError:
            pass

    # 3. docker-compose.yml
    for compose_name in ("docker-compose.yml", "docker-compose.yaml"):
        compose_file = root / compose_name
        if compose_file.exists():
            try:
                content = compose_file.read_text(encoding="utf-8", errors="replace")
                m = re.search(
                    r"DATABASE_URL[:\s]+=?\s*['\"]?([^\s'\"]+://[^\s'\"]+)",
                    content,
                    re.IGNORECASE,
                )
                if m:
                    return m.group(1)
            except OSError:
                pass

    return None


# ---------------------------------------------------------------------------
# Schema fetching
# ---------------------------------------------------------------------------

def _fetch_sqlite_schema(path: str) -> list[TableSchema]:
    """Introspect a SQLite database file."""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()

        schemas: list[TableSchema] = []
        for table_row in tables:
            table_name = table_row["name"]
            if table_name.startswith("sqlite_"):
                continue

            schema = TableSchema(table_name=table_name, language="sql", file_path=path)

            # PRAGMA table_info returns: cid, name, type, notnull, dflt_value, pk
            cols = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            for col in cols:
                schema.columns.append(ColumnDef(
                    name=col["name"],
                    col_type=col["type"] or "TEXT",
                    is_primary=bool(col["pk"]),
                    is_nullable=not bool(col["notnull"]),
                ))

            # PRAGMA foreign_key_list: id, seq, table, from, to, ...
            fks = conn.execute(f"PRAGMA foreign_key_list({table_name})").fetchall()
            fk_map: dict[str, str] = {}
            for fk in fks:
                fk_map[fk["from"]] = f"{fk['table']}.{fk['to']}"

            for col_def in schema.columns:
                if col_def.name in fk_map:
                    col_def.foreign_key = fk_map[col_def.name]

            schemas.append(schema)

        return schemas
    finally:
        conn.close()


def _fetch_postgresql_schema(connection_string: str) -> list[TableSchema]:
    """Introspect a PostgreSQL database using psycopg2."""
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError as e:
        raise ImportError(
            "psycopg2 is required for PostgreSQL support. "
            "Install with: pip install neural-memory-mcp[db]"
        ) from e

    conn = psycopg2.connect(connection_string)
    conn.autocommit = True
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    try:
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """)
        table_names = [row["table_name"] for row in cur.fetchall()]

        schemas: list[TableSchema] = []
        for table_name in table_names:
            schema = TableSchema(table_name=table_name, language="sql", file_path=connection_string)

            cur.execute("""
                SELECT column_name, data_type, is_nullable,
                       (SELECT kcu.column_name
                        FROM information_schema.table_constraints tc
                        JOIN information_schema.key_column_usage kcu
                          ON tc.constraint_name = kcu.constraint_name
                         AND tc.table_schema = kcu.table_schema
                        WHERE tc.constraint_type = 'PRIMARY KEY'
                          AND tc.table_name = c.table_name
                          AND kcu.column_name = c.column_name
                        LIMIT 1) AS pk_col
                FROM information_schema.columns c
                WHERE table_name = %s AND table_schema = 'public'
                ORDER BY ordinal_position
            """, (table_name,))

            col_rows = cur.fetchall()
            for col in col_rows:
                schema.columns.append(ColumnDef(
                    name=col["column_name"],
                    col_type=col["data_type"],
                    is_primary=col["pk_col"] is not None,
                    is_nullable=col["is_nullable"] == "YES",
                ))

            # Foreign keys
            cur.execute("""
                SELECT
                    kcu.column_name AS from_col,
                    ccu.table_name AS to_table,
                    ccu.column_name AS to_col
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage ccu
                  ON ccu.constraint_name = tc.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY'
                  AND tc.table_name = %s
            """, (table_name,))

            for fk in cur.fetchall():
                for col_def in schema.columns:
                    if col_def.name == fk["from_col"]:
                        col_def.foreign_key = f"{fk['to_table']}.{fk['to_col']}"

            schemas.append(schema)

        return schemas
    finally:
        cur.close()
        conn.close()


def _fetch_mysql_schema(connection_string: str) -> list[TableSchema]:
    """Introspect a MySQL database using pymysql."""
    try:
        import pymysql
        import pymysql.cursors
    except ImportError as e:
        raise ImportError(
            "pymysql is required for MySQL support. "
            "Install with: pip install neural-memory-mcp[db]"
        ) from e

    # Parse connection string: mysql://user:pass@host:port/dbname
    m = re.match(
        r"mysql(?:\+pymysql)?://([^:@]+)(?::([^@]*))?@([^:/]+)(?::(\d+))?/(\w+)",
        connection_string,
    )
    if not m:
        raise ValueError(f"Cannot parse MySQL connection string: {connection_string}")

    user, password, host, port, database = m.groups()
    conn = pymysql.connect(
        host=host,
        user=user,
        password=password or "",
        database=database,
        port=int(port) if port else 3306,
        cursorclass=pymysql.cursors.DictCursor,
    )
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT TABLE_NAME FROM information_schema.TABLES
                WHERE TABLE_SCHEMA = %s AND TABLE_TYPE = 'BASE TABLE'
                ORDER BY TABLE_NAME
            """, (database,))
            table_names = [row["TABLE_NAME"] for row in cur.fetchall()]

            schemas: list[TableSchema] = []
            for table_name in table_names:
                schema = TableSchema(table_name=table_name, language="sql", file_path=connection_string)

                cur.execute("""
                    SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_KEY
                    FROM information_schema.COLUMNS
                    WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
                    ORDER BY ORDINAL_POSITION
                """, (database, table_name))

                for col in cur.fetchall():
                    schema.columns.append(ColumnDef(
                        name=col["COLUMN_NAME"],
                        col_type=col["DATA_TYPE"],
                        is_primary=col["COLUMN_KEY"] == "PRI",
                        is_nullable=col["IS_NULLABLE"] == "YES",
                    ))

                # Foreign keys
                cur.execute("""
                    SELECT kcu.COLUMN_NAME, kcu.REFERENCED_TABLE_NAME, kcu.REFERENCED_COLUMN_NAME
                    FROM information_schema.KEY_COLUMN_USAGE kcu
                    JOIN information_schema.TABLE_CONSTRAINTS tc
                      ON kcu.CONSTRAINT_NAME = tc.CONSTRAINT_NAME
                     AND kcu.TABLE_SCHEMA = tc.TABLE_SCHEMA
                     AND kcu.TABLE_NAME = tc.TABLE_NAME
                    WHERE tc.CONSTRAINT_TYPE = 'FOREIGN KEY'
                      AND kcu.TABLE_SCHEMA = %s AND kcu.TABLE_NAME = %s
                """, (database, table_name))

                for fk in cur.fetchall():
                    for col_def in schema.columns:
                        if col_def.name == fk["COLUMN_NAME"] and fk["REFERENCED_TABLE_NAME"]:
                            col_def.foreign_key = f"{fk['REFERENCED_TABLE_NAME']}.{fk['REFERENCED_COLUMN_NAME']}"

                schemas.append(schema)

        return schemas
    finally:
        conn.close()


def fetch_schema(connection_string: str) -> list[TableSchema]:
    """Connect to a database and return its schema as TableSchema objects.

    Supports sqlite:///, postgresql://, and mysql:// connection strings.
    """
    cs = connection_string.strip()

    if cs.startswith("sqlite:///"):
        db_path = cs[len("sqlite:///"):]
        # Handle absolute vs relative paths
        if not os.path.isabs(db_path):
            db_path = os.path.join(os.getcwd(), db_path)
        return _fetch_sqlite_schema(db_path)

    if cs.startswith("postgresql://") or cs.startswith("postgres://"):
        return _fetch_postgresql_schema(cs)

    if cs.startswith("mysql://") or cs.startswith("mysql+pymysql://"):
        return _fetch_mysql_schema(cs)

    raise ValueError(
        f"Unsupported connection string scheme: {cs!r}. "
        "Expected sqlite:///, postgresql://, or mysql://"
    )
