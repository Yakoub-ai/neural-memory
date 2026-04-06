"""SQLite + JSON blob storage for the neural memory graph."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

from .config import get_memory_dir, DB_FILE
from .models import NeuralNode, NeuralEdge, IndexState, EmbeddingMeta, NodeType, EdgeType


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
                category TEXT DEFAULT 'codebase',
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
                last_indexed TEXT,
                last_verified TEXT
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
                node_id UNINDEXED,
                name,
                summary_short,
                docstring,
                tokenize='unicode61'
            );

            CREATE INDEX IF NOT EXISTS idx_nodes_file ON nodes(file_path);
            CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);
            CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name);
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);

            CREATE TABLE IF NOT EXISTS embeddings (
                node_id TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                content_hash TEXT NOT NULL,
                FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS embedding_meta (
                key TEXT PRIMARY KEY DEFAULT 'main',
                data JSON NOT NULL
            );

            CREATE TABLE IF NOT EXISTS schema_version (
                key TEXT PRIMARY KEY DEFAULT 'main',
                version INTEGER NOT NULL DEFAULT 2
            );

            CREATE TABLE IF NOT EXISTS package_docs (
                package_name TEXT NOT NULL,
                registry TEXT NOT NULL,
                version TEXT DEFAULT '',
                summary TEXT DEFAULT '',
                description TEXT DEFAULT '',
                homepage_url TEXT DEFAULT '',
                doc_url TEXT DEFAULT '',
                fetched_at TEXT NOT NULL,
                PRIMARY KEY (package_name, registry)
            );

            CREATE INDEX IF NOT EXISTS idx_pkgdocs_registry ON package_docs(registry);

            CREATE TABLE IF NOT EXISTS session_log (
                session_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                summary TEXT DEFAULT '',
                files_touched TEXT DEFAULT '[]',
                insights_added TEXT DEFAULT '[]',
                data JSON DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_session_log_started ON session_log(started_at);

        """)
        self._migrate_schema()
        self.conn.commit()

    def _migrate_schema(self) -> None:
        """Apply incremental schema migrations for existing databases."""
        # v1→v2: add category column to nodes table
        try:
            self.conn.execute("ALTER TABLE nodes ADD COLUMN category TEXT DEFAULT 'codebase'")
            self.conn.commit()
        except Exception:
            pass  # Column already exists — safe to ignore
        # Create category index after column is guaranteed to exist
        try:
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_category ON nodes(category)")
            self.conn.commit()
        except Exception:
            pass
        # v2→v3: add language column to nodes table
        try:
            self.conn.execute("ALTER TABLE nodes ADD COLUMN language TEXT DEFAULT ''")
            self.conn.commit()
        except Exception:
            pass
        # v4→v5: add archived column to nodes table
        try:
            self.conn.execute("ALTER TABLE nodes ADD COLUMN archived INTEGER DEFAULT 0")
            self.conn.commit()
        except Exception:
            pass
        try:
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_archived ON nodes(archived)")
            self.conn.commit()
        except Exception:
            pass

        # v3→v4: add package_docs table
        try:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS package_docs (
                    package_name TEXT NOT NULL,
                    registry TEXT NOT NULL,
                    version TEXT DEFAULT '',
                    summary TEXT DEFAULT '',
                    description TEXT DEFAULT '',
                    homepage_url TEXT DEFAULT '',
                    doc_url TEXT DEFAULT '',
                    fetched_at TEXT NOT NULL,
                    PRIMARY KEY (package_name, registry)
                );
                CREATE INDEX IF NOT EXISTS idx_pkgdocs_registry ON package_docs(registry);
            """)
            self.conn.commit()
        except Exception:
            pass

        # v5: add last_verified column to file_hashes for freshness cache
        try:
            self.conn.execute("ALTER TABLE file_hashes ADD COLUMN last_verified TEXT")
            self.conn.commit()
        except Exception:
            pass

        # v5: add FTS5 search table for nodes
        try:
            self.conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
                    node_id UNINDEXED,
                    name,
                    summary_short,
                    docstring,
                    tokenize='unicode61'
                )
            """)
            self.conn.commit()
        except Exception:
            pass

        # v6: add session_log table for cross-session memory
        try:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS session_log (
                    session_id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    summary TEXT DEFAULT '',
                    files_touched TEXT DEFAULT '[]',
                    insights_added TEXT DEFAULT '[]',
                    data JSON DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_session_log_started ON session_log(started_at);
            """)
            self.conn.commit()
        except Exception:
            pass

    # ── Node operations ──

    def upsert_node(self, node: NeuralNode) -> None:
        self.conn.execute(
            """INSERT INTO nodes (id, name, node_type, file_path, line_start, line_end,
                                  content_hash, importance, category, archived, data)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   name=excluded.name, node_type=excluded.node_type,
                   file_path=excluded.file_path, line_start=excluded.line_start,
                   line_end=excluded.line_end, content_hash=excluded.content_hash,
                   importance=excluded.importance, category=excluded.category,
                   archived=excluded.archived, data=excluded.data""",
            (node.id, node.name, node.node_type.value, node.file_path,
             node.line_start, node.line_end, node.content_hash,
             node.importance, node.category, 1 if node.archived else 0,
             json.dumps(node.to_dict()))
        )
        # Always auto-commits. For bulk writes use batch_upsert_nodes() instead.
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

    def get_nodes_by_category(self, category: str) -> list[NeuralNode]:
        rows = self.conn.execute(
            "SELECT data FROM nodes WHERE category = ?", (category,)
        ).fetchall()
        return [NeuralNode.from_dict(json.loads(r["data"])) for r in rows]

    def get_active_items(self, category: str) -> list[NeuralNode]:
        """Return non-archived nodes for a category (bugs or tasks), ordered by importance."""
        rows = self.conn.execute(
            "SELECT data FROM nodes WHERE category = ? AND (archived IS NULL OR archived = 0) ORDER BY importance DESC",
            (category,)
        ).fetchall()
        return [NeuralNode.from_dict(json.loads(r["data"])) for r in rows]

    def get_tasks(
        self,
        status: str | None = None,
        priority: str | None = None,
        include_archived: bool = False,
    ) -> list[NeuralNode]:
        """Return task nodes with optional filters.

        Uses json_extract() on the data blob for status/priority filtering.
        Results ordered by importance DESC.
        """
        conditions = ["category = 'tasks'"]
        params: list = []

        if not include_archived:
            conditions.append("(archived IS NULL OR archived = 0)")

        if status is not None:
            conditions.append("json_extract(data, '$.task_status') = ?")
            params.append(status)

        if priority is not None:
            conditions.append("json_extract(data, '$.priority') = ?")
            params.append(priority)

        where = " AND ".join(conditions)
        rows = self.conn.execute(
            f"SELECT data FROM nodes WHERE {where} ORDER BY importance DESC",
            params,
        ).fetchall()
        return [NeuralNode.from_dict(json.loads(r["data"])) for r in rows]

    def get_insights(self, topic: str | None = None) -> list[NeuralNode]:
        """Return insight nodes, optionally filtered by topic.

        Topic is encoded in the node name as 'insight/{topic}: ...', so filtering
        uses a LIKE prefix match on the name column. Results ordered by importance DESC.
        """
        conditions = ["category = 'insights'"]
        params: list = []

        if topic is not None:
            conditions.append("name LIKE ?")
            params.append(f"insight/{topic.strip().lower()}%")

        where = " AND ".join(conditions)
        rows = self.conn.execute(
            f"SELECT data FROM nodes WHERE {where} ORDER BY importance DESC",
            params,
        ).fetchall()
        return [NeuralNode.from_dict(json.loads(r["data"])) for r in rows]

    def update_node_field(self, node_id: str, field: str, value: str) -> bool:
        """Update a single field in a node's JSON data blob.

        Reads the node, updates the field in the deserialized dict,
        then writes the full JSON blob back. Returns True if node found.
        Only allows updating known safe fields (task_status, priority, bug_status, severity).
        """
        _ALLOWED_FIELDS = {"task_status", "priority", "bug_status", "severity"}
        if field not in _ALLOWED_FIELDS:
            raise ValueError(f"Field '{field}' is not updatable. Allowed: {_ALLOWED_FIELDS}")

        node = self.get_node(node_id)
        if not node:
            return False

        setattr(node, field, value)
        self.conn.execute(
            "UPDATE nodes SET data = ? WHERE id = ?",
            (json.dumps(node.to_dict()), node_id)
        )
        self.conn.commit()
        return True

    def archive_node(self, node_id: str) -> bool:
        """Mark a single node as archived and decay its importance by 0.3×."""
        node = self.get_node(node_id)
        if not node:
            return False
        node.archived = True
        node.importance = round(node.importance * 0.3, 3)
        self.conn.execute(
            "UPDATE nodes SET archived = 1, importance = ?, data = ? WHERE id = ?",
            (node.importance, json.dumps(node.to_dict()), node_id)
        )
        self.conn.commit()
        return True

    def unarchive_node(self, node_id: str) -> bool:
        """Unarchive a node (restore to active state — importance is NOT restored)."""
        node = self.get_node(node_id)
        if not node:
            return False
        node.archived = False
        self.conn.execute(
            "UPDATE nodes SET archived = 0, data = ? WHERE id = ?",
            (json.dumps(node.to_dict()), node_id)
        )
        self.conn.commit()
        return True

    def archive_completed(self) -> int:
        """Bulk-archive all done tasks and fixed bugs. Returns count archived."""
        rows = self.conn.execute(
            """SELECT data FROM nodes
               WHERE (archived IS NULL OR archived = 0)
               AND category IN ('bugs', 'tasks')
               AND (
                   (category = 'bugs' AND json_extract(data, '$.bug_status') = 'fixed')
                   OR (category = 'tasks' AND json_extract(data, '$.task_status') = 'done')
               )"""
        ).fetchall()
        count = 0
        for row in rows:
            node = NeuralNode.from_dict(json.loads(row["data"]))
            node.archived = True
            node.importance = round(node.importance * 0.3, 3)
            self.conn.execute(
                "UPDATE nodes SET archived = 1, importance = ?, data = ? WHERE id = ?",
                (node.importance, json.dumps(node.to_dict()), node.id)
            )
            count += 1
        if count:
            self.conn.commit()
        return count

    def get_nodes_by_type(self, node_type: NodeType) -> list[NeuralNode]:
        rows = self.conn.execute(
            "SELECT data FROM nodes WHERE node_type = ?", (node_type.value,)
        ).fetchall()
        return [NeuralNode.from_dict(json.loads(r["data"])) for r in rows]

    def check_file_freshness(
        self,
        file_paths: list[str],
        project_root: str,
        skip_if_verified_within_seconds: int = 300,
    ) -> dict[str, str]:
        """Check if indexed files are still fresh by comparing disk hash vs stored hash.

        Returns dict mapping file_path -> "fresh" | "stale" | "deleted" | "new".
        Files verified within skip_if_verified_within_seconds are assumed fresh without re-hashing.
        Cost: ~1ms per uncached file (SHA-256 of file contents).
        """
        if not file_paths:
            return {}

        now = datetime.now(timezone.utc)
        root = Path(project_root).resolve()
        result: dict[str, str] = {}

        # Load stored hashes + last_verified for all requested files in one query
        placeholders = ",".join("?" * len(file_paths))
        rows = self.conn.execute(
            f"SELECT file_path, content_hash, last_verified FROM file_hashes WHERE file_path IN ({placeholders})",
            file_paths,
        ).fetchall()
        stored = {r["file_path"]: (r["content_hash"], r["last_verified"]) for r in rows}

        to_verify: list[tuple[str, str, Path]] = []

        for fp in file_paths:
            if fp not in stored:
                result[fp] = "new"
                continue
            stored_hash, last_verified = stored[fp]
            full_path = root / fp
            if not full_path.exists():
                result[fp] = "deleted"
                continue
            # Freshness cache: skip re-hashing if verified recently
            if last_verified:
                try:
                    lv_dt = datetime.fromisoformat(last_verified)
                    if lv_dt.tzinfo is None:
                        lv_dt = lv_dt.replace(tzinfo=timezone.utc)
                    if (now - lv_dt).total_seconds() < skip_if_verified_within_seconds:
                        result[fp] = "fresh"
                        continue
                except Exception:
                    pass
            to_verify.append((fp, stored_hash, full_path))

        # Re-hash files that need verification
        fresh_now = now.isoformat()
        verified_fps: list[str] = []
        for fp, stored_hash, full_path in to_verify:
            try:
                current_hash = hashlib.sha256(full_path.read_bytes()).hexdigest()[:16]
                if current_hash == stored_hash:
                    result[fp] = "fresh"
                    verified_fps.append(fp)
                else:
                    result[fp] = "stale"
            except Exception:
                result[fp] = "stale"

        # Batch-update last_verified for confirmed-fresh files
        if verified_fps:
            self.conn.executemany(
                "UPDATE file_hashes SET last_verified = ? WHERE file_path = ?",
                [(fresh_now, fp) for fp in verified_fps],
            )
            self.conn.commit()

        return result

    def populate_fts(self) -> None:
        """Rebuild the FTS5 search index from current nodes. Call after index/update."""
        try:
            self.conn.execute("DELETE FROM nodes_fts")
            self.conn.execute("""
                INSERT INTO nodes_fts(node_id, name, summary_short, docstring)
                SELECT id,
                       name,
                       COALESCE(json_extract(data, '$.summary_short'), ''),
                       COALESCE(json_extract(data, '$.docstring'), '')
                FROM nodes
            """)
            self.conn.commit()
        except Exception as e:
            logger.warning("populate_fts failed: %s", e)

    def update_fts_for_nodes(self, node_ids: list[str]) -> None:
        """Incrementally update FTS5 entries for the given node IDs only.

        Much cheaper than a full populate_fts() rebuild — used by micro_update()
        to keep search accurate after single-file re-parses.
        """
        if not node_ids:
            return
        try:
            placeholders = ",".join("?" * len(node_ids))
            self.conn.execute(
                f"DELETE FROM nodes_fts WHERE node_id IN ({placeholders})", node_ids
            )
            self.conn.execute(
                f"""INSERT INTO nodes_fts(node_id, name, summary_short, docstring)
                    SELECT id,
                           name,
                           COALESCE(json_extract(data, '$.summary_short'), ''),
                           COALESCE(json_extract(data, '$.docstring'), '')
                    FROM nodes WHERE id IN ({placeholders})""",
                node_ids,
            )
            self.conn.commit()
        except Exception as e:
            logger.warning("update_fts_for_nodes failed: %s", e)

    def search_nodes(self, query: str, limit: int = 20) -> list[NeuralNode]:
        """Search nodes by name or summary — uses FTS5 BM25 ranking when available."""
        # Attempt FTS5 search (BM25 ranking, much better than LIKE)
        try:
            words = [w.strip() for w in query.split() if w.strip()]
            if words:
                # Prefix-match each word for typo-tolerance on partials
                fts_query = " ".join(f'"{w}"*' for w in words)
                rows = self.conn.execute(
                    """SELECT n.data FROM nodes_fts
                       JOIN nodes n ON n.id = nodes_fts.node_id
                       WHERE nodes_fts MATCH ?
                       ORDER BY rank, n.importance DESC
                       LIMIT ?""",
                    (fts_query, limit),
                ).fetchall()
                if rows:
                    return [NeuralNode.from_dict(json.loads(r["data"])) for r in rows]
        except Exception as e:
            logger.warning("FTS5 search failed, falling back to LIKE: %s", e)

        # Fallback: LIKE search on name + JSON blob
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

    def get_all_edges(self) -> list[NeuralEdge]:
        rows = self.conn.execute("SELECT * FROM edges").fetchall()
        return [NeuralEdge(
            source_id=r["source_id"], target_id=r["target_id"],
            edge_type=EdgeType(r["edge_type"]),
            context=r["context"], weight=r["weight"]
        ) for r in rows]

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
            """INSERT INTO file_hashes (file_path, content_hash, last_indexed, last_verified)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(file_path) DO UPDATE SET
                   content_hash=excluded.content_hash,
                   last_indexed=excluded.last_indexed,
                   last_verified=excluded.last_verified""",
            (file_path, content_hash, timestamp, timestamp)
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

    # ── Embeddings ──

    def upsert_embedding(self, node_id: str, vector_bytes: bytes, content_hash: str) -> None:
        self.conn.execute(
            """INSERT INTO embeddings (node_id, vector, content_hash)
               VALUES (?, ?, ?)
               ON CONFLICT(node_id) DO UPDATE SET
                   vector=excluded.vector,
                   content_hash=excluded.content_hash""",
            (node_id, vector_bytes, content_hash)
        )
        self.conn.commit()

    def get_embedding(self, node_id: str) -> Optional[bytes]:
        row = self.conn.execute(
            "SELECT vector FROM embeddings WHERE node_id = ?", (node_id,)
        ).fetchone()
        return bytes(row["vector"]) if row else None

    def get_all_embeddings(self) -> dict[str, bytes]:
        """Returns {node_id: vector_bytes} for all embedded nodes."""
        rows = self.conn.execute(
            "SELECT node_id, vector FROM embeddings"
        ).fetchall()
        return {r["node_id"]: bytes(r["vector"]) for r in rows}

    def get_embedding_hashes(self) -> dict[str, str]:
        """Returns {node_id: content_hash} for staleness checking."""
        rows = self.conn.execute(
            "SELECT node_id, content_hash FROM embeddings"
        ).fetchall()
        return {r["node_id"]: r["content_hash"] for r in rows}

    def delete_embeddings_by_file(self, file_path: str) -> None:
        self.conn.execute(
            """DELETE FROM embeddings WHERE node_id IN
               (SELECT id FROM nodes WHERE file_path = ?)""",
            (file_path,)
        )
        self.conn.commit()

    def save_embedding_meta(self, meta: EmbeddingMeta) -> None:
        self.conn.execute(
            """INSERT INTO embedding_meta (key, data) VALUES ('main', ?)
               ON CONFLICT(key) DO UPDATE SET data=excluded.data""",
            (json.dumps(meta.to_dict()),)
        )
        self.conn.commit()

    def get_embedding_meta(self) -> Optional[EmbeddingMeta]:
        row = self.conn.execute(
            "SELECT data FROM embedding_meta WHERE key = 'main'"
        ).fetchone()
        if row:
            return EmbeddingMeta.from_dict(json.loads(row["data"]))
        return None

    # ── Package docs ──

    def upsert_package_doc(self, package_name: str, registry: str, data: dict, fetched_at: str) -> None:
        """Store or update a fetched package doc. data contains: version, summary, description, homepage_url, doc_url"""
        self.conn.execute(
            """INSERT INTO package_docs (package_name, registry, version, summary, description, homepage_url, doc_url, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(package_name, registry) DO UPDATE SET
                   version=excluded.version, summary=excluded.summary,
                   description=excluded.description, homepage_url=excluded.homepage_url,
                   doc_url=excluded.doc_url, fetched_at=excluded.fetched_at""",
            (
                package_name, registry,
                data.get("version", ""), data.get("summary", ""),
                data.get("description", ""), data.get("homepage_url", ""),
                data.get("doc_url", ""), fetched_at,
            )
        )
        self.conn.commit()

    def get_package_doc(self, package_name: str, registry: str) -> Optional[dict]:
        """Retrieve a cached package doc. Returns dict with all fields, or None."""
        row = self.conn.execute(
            "SELECT * FROM package_docs WHERE package_name = ? AND registry = ?",
            (package_name, registry)
        ).fetchone()
        if row:
            return dict(row)
        return None

    def get_all_package_docs(self) -> list[dict]:
        """Return all cached package docs as list of dicts."""
        rows = self.conn.execute("SELECT * FROM package_docs").fetchall()
        return [dict(r) for r in rows]

    # ── Batch operations ──

    @contextmanager
    def transaction(self):
        """Wrap a block in a single SQLite transaction for batch performance."""
        try:
            yield
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def batch_upsert_nodes(self, nodes: list[NeuralNode]) -> None:
        """Upsert multiple nodes in a single transaction."""
        if not nodes:
            return
        with self.transaction():
            self.conn.executemany(
                """INSERT INTO nodes (id, name, node_type, file_path, line_start, line_end,
                                      content_hash, importance, category, archived, data)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                       name=excluded.name, node_type=excluded.node_type,
                       file_path=excluded.file_path, line_start=excluded.line_start,
                       line_end=excluded.line_end, content_hash=excluded.content_hash,
                       importance=excluded.importance, category=excluded.category,
                       archived=excluded.archived, data=excluded.data""",
                [(n.id, n.name, n.node_type.value, n.file_path,
                  n.line_start, n.line_end, n.content_hash,
                  n.importance, n.category, 1 if n.archived else 0,
                  json.dumps(n.to_dict()))
                 for n in nodes]
            )

    def batch_upsert_edges(self, edges: list[NeuralEdge]) -> None:
        """Upsert multiple edges in a single transaction."""
        if not edges:
            return
        with self.transaction():
            self.conn.executemany(
                """INSERT INTO edges (source_id, target_id, edge_type, context, weight)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(source_id, target_id, edge_type) DO UPDATE SET
                       context=excluded.context, weight=excluded.weight""",
                [(e.source_id, e.target_id, e.edge_type.value, e.context, e.weight)
                 for e in edges]
            )

    def batch_save_file_hashes(self, entries: list[tuple[str, str, str]]) -> None:
        """Save multiple file hashes in a single transaction.

        Args:
            entries: list of (file_path, content_hash, timestamp)
        """
        if not entries:
            return
        with self.transaction():
            # Expand entries to include last_verified = timestamp (just indexed = just verified)
            expanded = [(fp, ch, ts, ts) for fp, ch, ts in entries]
            self.conn.executemany(
                """INSERT INTO file_hashes (file_path, content_hash, last_indexed, last_verified)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(file_path) DO UPDATE SET
                       content_hash=excluded.content_hash,
                       last_indexed=excluded.last_indexed,
                       last_verified=excluded.last_verified""",
                expanded
            )

    def get_all_edges_by_node(self) -> dict[str, dict[str, list[NeuralEdge]]]:
        """Return {node_id: {'incoming': [...], 'outgoing': [...]}} for all nodes in one query."""
        rows = self.conn.execute("SELECT * FROM edges").fetchall()
        result: dict[str, dict[str, list[NeuralEdge]]] = {}
        for r in rows:
            edge = NeuralEdge(
                source_id=r["source_id"], target_id=r["target_id"],
                edge_type=EdgeType(r["edge_type"]),
                context=r["context"], weight=r["weight"]
            )
            # outgoing for source
            if edge.source_id not in result:
                result[edge.source_id] = {"incoming": [], "outgoing": []}
            result[edge.source_id]["outgoing"].append(edge)
            # incoming for target
            if edge.target_id not in result:
                result[edge.target_id] = {"incoming": [], "outgoing": []}
            result[edge.target_id]["incoming"].append(edge)
        return result

    def get_all_degree_counts(self) -> dict[str, tuple[int, int]]:
        """Return {node_id: (in_degree, out_degree)} for all nodes in one query."""
        rows = self.conn.execute("""
            SELECT n.id,
                   COALESCE(ein.cnt, 0) AS in_deg,
                   COALESCE(eout.cnt, 0) AS out_deg
            FROM nodes n
            LEFT JOIN (
                SELECT target_id, COUNT(*) AS cnt FROM edges GROUP BY target_id
            ) ein ON ein.target_id = n.id
            LEFT JOIN (
                SELECT source_id, COUNT(*) AS cnt FROM edges GROUP BY source_id
            ) eout ON eout.source_id = n.id
        """).fetchall()
        return {r["id"]: (r["in_deg"], r["out_deg"]) for r in rows}

    # ── Session log ──

    def save_session(
        self,
        session_id: str,
        started_at: str,
        ended_at: Optional[str],
        summary: str,
        files_touched: list[str],
        insights_added: list[str],
        data: dict,
    ) -> None:
        """Write or update a session log entry."""
        self.conn.execute(
            """INSERT INTO session_log
                   (session_id, started_at, ended_at, summary, files_touched, insights_added, data)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(session_id) DO UPDATE SET
                   ended_at=excluded.ended_at,
                   summary=excluded.summary,
                   files_touched=excluded.files_touched,
                   insights_added=excluded.insights_added,
                   data=excluded.data""",
            (
                session_id, started_at, ended_at, summary,
                json.dumps(files_touched), json.dumps(insights_added),
                json.dumps(data),
            ),
        )
        self.conn.commit()

    def get_recent_sessions(self, limit: int = 3) -> list[dict]:
        """Return the N most recent session log entries, newest first."""
        rows = self.conn.execute(
            "SELECT * FROM session_log ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
        result = []
        for r in rows:
            result.append({
                "session_id": r["session_id"],
                "started_at": r["started_at"],
                "ended_at": r["ended_at"],
                "summary": r["summary"],
                "files_touched": json.loads(r["files_touched"] or "[]"),
                "insights_added": json.loads(r["insights_added"] or "[]"),
                "data": json.loads(r["data"] or "{}"),
            })
        return result

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
