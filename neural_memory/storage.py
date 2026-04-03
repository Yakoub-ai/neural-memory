"""SQLite + JSON blob storage for the neural memory graph."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

from .config import get_memory_dir, DB_FILE
from .models import NeuralNode, NeuralEdge, IndexState, NodeType, EdgeType


class Storage:
    """SQLite-backed storage for neural memory graph."""

    def __init__(self, project_root: str = "."):
        self.db_path = get_memory_dir(project_root) / DB_FILE
        self.conn: Optional[sqlite3.Connection] = None

    def open(self) -> None:
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    def _create_tables(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                node_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                line_start INTEGER,
                line_end INTEGER,
                content_hash TEXT,
                importance REAL DEFAULT 0.0,
                data JSON NOT NULL
            );

            CREATE TABLE IF NOT EXISTS edges (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                context TEXT DEFAULT '',
                weight REAL DEFAULT 1.0,
                PRIMARY KEY (source_id, target_id, edge_type),
                FOREIGN KEY (source_id) REFERENCES nodes(id) ON DELETE CASCADE,
                FOREIGN KEY (target_id) REFERENCES nodes(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS index_state (
                key TEXT PRIMARY KEY DEFAULT 'main',
                data JSON NOT NULL
            );

            CREATE TABLE IF NOT EXISTS file_hashes (
                file_path TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                last_indexed TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_nodes_file ON nodes(file_path);
            CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);
            CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name);
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
        """)
        self.conn.commit()

    # ── Node operations ──

    def upsert_node(self, node: NeuralNode) -> None:
        self.conn.execute(
            """INSERT INTO nodes (id, name, node_type, file_path, line_start, line_end,
                                  content_hash, importance, data)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   name=excluded.name, node_type=excluded.node_type,
                   file_path=excluded.file_path, line_start=excluded.line_start,
                   line_end=excluded.line_end, content_hash=excluded.content_hash,
                   importance=excluded.importance, data=excluded.data""",
            (node.id, node.name, node.node_type.value, node.file_path,
             node.line_start, node.line_end, node.content_hash,
             node.importance, json.dumps(node.to_dict()))
        )
        self.conn.commit()

    def get_node(self, node_id: str) -> Optional[NeuralNode]:
        row = self.conn.execute(
            "SELECT data FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        if row:
            return NeuralNode.from_dict(json.loads(row["data"]))
        return None

    def get_nodes_by_file(self, file_path: str) -> list[NeuralNode]:
        rows = self.conn.execute(
            "SELECT data FROM nodes WHERE file_path = ?", (file_path,)
        ).fetchall()
        return [NeuralNode.from_dict(json.loads(r["data"])) for r in rows]

    def get_nodes_by_type(self, node_type: NodeType) -> list[NeuralNode]:
        rows = self.conn.execute(
            "SELECT data FROM nodes WHERE node_type = ?", (node_type.value,)
        ).fetchall()
        return [NeuralNode.from_dict(json.loads(r["data"])) for r in rows]

    def search_nodes(self, query: str, limit: int = 20) -> list[NeuralNode]:
        """Search nodes by name or summary content."""
        pattern = f"%{query}%"
        rows = self.conn.execute(
            """SELECT data FROM nodes
               WHERE name LIKE ? OR data LIKE ?
               ORDER BY importance DESC
               LIMIT ?""",
            (pattern, pattern, limit)
        ).fetchall()
        return [NeuralNode.from_dict(json.loads(r["data"])) for r in rows]

    def delete_nodes_by_file(self, file_path: str) -> int:
        cursor = self.conn.execute(
            "DELETE FROM nodes WHERE file_path = ?", (file_path,)
        )
        self.conn.commit()
        return cursor.rowcount

    def get_all_node_ids(self) -> list[str]:
        rows = self.conn.execute("SELECT id FROM nodes").fetchall()
        return [r["id"] for r in rows]

    def get_all_nodes(self) -> list[NeuralNode]:
        rows = self.conn.execute("SELECT data FROM nodes").fetchall()
        return [NeuralNode.from_dict(json.loads(r["data"])) for r in rows]

    # ── Edge operations ──

    def upsert_edge(self, edge: NeuralEdge) -> None:
        self.conn.execute(
            """INSERT INTO edges (source_id, target_id, edge_type, context, weight)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(source_id, target_id, edge_type) DO UPDATE SET
                   context=excluded.context, weight=excluded.weight""",
            (edge.source_id, edge.target_id, edge.edge_type.value,
             edge.context, edge.weight)
        )
        self.conn.commit()

    def get_edges_from(self, node_id: str) -> list[NeuralEdge]:
        rows = self.conn.execute(
            "SELECT * FROM edges WHERE source_id = ?", (node_id,)
        ).fetchall()
        return [NeuralEdge(
            source_id=r["source_id"], target_id=r["target_id"],
            edge_type=EdgeType(r["edge_type"]),
            context=r["context"], weight=r["weight"]
        ) for r in rows]

    def get_edges_to(self, node_id: str) -> list[NeuralEdge]:
        rows = self.conn.execute(
            "SELECT * FROM edges WHERE target_id = ?", (node_id,)
        ).fetchall()
        return [NeuralEdge(
            source_id=r["source_id"], target_id=r["target_id"],
            edge_type=EdgeType(r["edge_type"]),
            context=r["context"], weight=r["weight"]
        ) for r in rows]

    def delete_edges_for_node(self, node_id: str) -> None:
        self.conn.execute(
            "DELETE FROM edges WHERE source_id = ? OR target_id = ?",
            (node_id, node_id)
        )
        self.conn.commit()

    def delete_edges_by_file(self, file_path: str) -> None:
        """Delete all edges involving nodes from a specific file."""
        self.conn.execute(
            """DELETE FROM edges WHERE source_id IN
               (SELECT id FROM nodes WHERE file_path = ?)
               OR target_id IN
               (SELECT id FROM nodes WHERE file_path = ?)""",
            (file_path, file_path)
        )
        self.conn.commit()

    # ── Index state ──

    def get_index_state(self) -> IndexState:
        row = self.conn.execute(
            "SELECT data FROM index_state WHERE key = 'main'"
        ).fetchone()
        if row:
            return IndexState.from_dict(json.loads(row["data"]))
        return IndexState()

    def save_index_state(self, state: IndexState) -> None:
        self.conn.execute(
            """INSERT INTO index_state (key, data) VALUES ('main', ?)
               ON CONFLICT(key) DO UPDATE SET data=excluded.data""",
            (json.dumps(state.to_dict()),)
        )
        self.conn.commit()

    # ── File hashes ──

    def get_file_hash(self, file_path: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT content_hash FROM file_hashes WHERE file_path = ?",
            (file_path,)
        ).fetchone()
        return row["content_hash"] if row else None

    def save_file_hash(self, file_path: str, content_hash: str, timestamp: str) -> None:
        self.conn.execute(
            """INSERT INTO file_hashes (file_path, content_hash, last_indexed)
               VALUES (?, ?, ?)
               ON CONFLICT(file_path) DO UPDATE SET
                   content_hash=excluded.content_hash,
                   last_indexed=excluded.last_indexed""",
            (file_path, content_hash, timestamp)
        )
        self.conn.commit()

    def delete_file_hash(self, file_path: str) -> None:
        self.conn.execute(
            "DELETE FROM file_hashes WHERE file_path = ?", (file_path,)
        )
        self.conn.commit()

    def get_all_indexed_files(self) -> dict[str, str]:
        """Returns {file_path: content_hash} for all indexed files."""
        rows = self.conn.execute(
            "SELECT file_path, content_hash FROM file_hashes"
        ).fetchall()
        return {r["file_path"]: r["content_hash"] for r in rows}

    # ── Stats ──

    def get_stats(self) -> dict:
        node_count = self.conn.execute("SELECT COUNT(*) as c FROM nodes").fetchone()["c"]
        edge_count = self.conn.execute("SELECT COUNT(*) as c FROM edges").fetchone()["c"]
        file_count = self.conn.execute("SELECT COUNT(DISTINCT file_path) as c FROM nodes").fetchone()["c"]
        type_counts = {}
        for row in self.conn.execute("SELECT node_type, COUNT(*) as c FROM nodes GROUP BY node_type").fetchall():
            type_counts[row["node_type"]] = row["c"]
        return {
            "total_nodes": node_count,
            "total_edges": edge_count,
            "total_files": file_count,
            "nodes_by_type": type_counts,
        }
