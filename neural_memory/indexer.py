"""Indexer — orchestrates full and incremental indexing of a codebase."""

from __future__ import annotations

import fnmatch
import hashlib
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import NeuralConfig, load_config
from .context_parser import import_context_logs
from .embeddings import compute_all_embeddings, update_embeddings, is_available as embeddings_available
from .lsp_client import LSPClient, is_lsp_available
from .models import NeuralNode, NeuralEdge, IndexMode, IndexState
from .overview import generate_and_store_overviews
from .parser import parse_file, resolve_edges
from .redactor import Redactor
from .storage import Storage
from .summarizer import summarize_node
from .graph import compute_importance


def _discover_files(config: NeuralConfig) -> list[str]:
    """Discover all Python files matching include/exclude patterns."""
    root = Path(config.project_root).resolve()
    files = []

    for include in config.include_patterns:
        for path in root.glob(include):
            rel_path = str(path.relative_to(root))
            # Check excludes
            excluded = any(
                fnmatch.fnmatch(rel_path, exc) or fnmatch.fnmatch(str(path), exc)
                for exc in config.exclude_patterns
            )
            if not excluded and path.is_file():
                files.append(rel_path)

    return sorted(set(files))


def _file_hash(file_path: str, project_root: str) -> str:
    """Compute hash of a file's contents."""
    full_path = Path(project_root).resolve() / file_path
    with open(full_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def _get_git_head(project_root: str) -> Optional[str]:
    """Get current git HEAD commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=project_root
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        pass
    return None


def _get_git_changed_files(since_commit: str, project_root: str) -> list[str]:
    """Get files changed since a specific commit."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", since_commit, "HEAD"],
            capture_output=True, text=True, cwd=project_root
        )
        if result.returncode == 0:
            return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
    except FileNotFoundError:
        pass
    return []


def _get_commits_behind(since_commit: str, project_root: str) -> int:
    """Count commits between since_commit and HEAD."""
    try:
        result = subprocess.run(
            ["git", "rev-list", "--count", f"{since_commit}..HEAD"],
            capture_output=True, text=True, cwd=project_root
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except (FileNotFoundError, ValueError):
        pass
    return 0


def _enrich_with_lsp(
    storage,
    all_nodes: dict,
    project_root: str,
    lsp_server: str = "auto",
) -> None:
    """Enrich high-importance code nodes with LSP hover info and diagnostics."""
    high_importance = [
        n for n in all_nodes.values()
        if n.importance >= 0.3 and n.file_path and not n.file_path.startswith("__")
    ]
    if not high_importance:
        return

    with LSPClient(project_root, server=lsp_server) as lsp:
        if not lsp._proc:
            return  # LSP failed to start
        for node in high_importance:
            try:
                hover_text = lsp.hover(node.file_path, node.line_start or 1)
                diags = lsp.diagnostics(node.file_path)
                if hover_text or diags:
                    node.lsp_hover_doc = (hover_text or "")[:500]
                    node.lsp_diagnostics = diags[:10]
                    storage.upsert_node(node)
            except Exception:
                pass


def full_index(config: Optional[NeuralConfig] = None, project_root: str = ".") -> dict:
    """Perform a full index of the codebase.

    Returns stats about the indexing operation.
    """
    if config is None:
        config = load_config(project_root)

    files = _discover_files(config)
    redactor = Redactor(config.redaction)

    all_nodes: dict[str, NeuralNode] = {}
    all_edges: list[NeuralEdge] = []
    stats = {
        "files_processed": 0,
        "files_skipped": 0,
        "nodes_created": 0,
        "edges_created": 0,
        "redactions": 0,
        "embeddings_computed": 0,
        "bugs_imported": 0,
        "tasks_imported": 0,
        "overview_nodes": 0,
        "errors": [],
    }

    root = Path(config.project_root).resolve()

    # Phase 1: Parse all files
    for rel_path in files:
        full_path = root / rel_path
        try:
            nodes, edges = parse_file(str(rel_path), source=full_path.read_text(encoding="utf-8", errors="replace"))
            for node in nodes:
                all_nodes[node.id] = node
            all_edges.extend(edges)
            stats["files_processed"] += 1
        except Exception as e:
            stats["errors"].append(f"{rel_path}: {str(e)}")
            stats["files_skipped"] += 1

    # Phase 2: Resolve cross-file edges
    all_edges = resolve_edges(all_nodes, all_edges)

    # Phase 3: Redact sensitive content
    for node in all_nodes.values():
        if node.raw_code:
            redacted_code, redacted_summary, had_redactions = redactor.redact_node_content(
                node.raw_code, node.summary_short
            )
            if had_redactions:
                node.raw_code = redacted_code
                node.summary_short = redacted_summary
                node.has_redacted_content = True
                stats["redactions"] += 1

    # Phase 4: Store everything
    with Storage(config.project_root) as storage:
        now = datetime.now(timezone.utc).isoformat()

        for node in all_nodes.values():
            storage.upsert_node(node)
            stats["nodes_created"] += 1

        for edge in all_edges:
            storage.upsert_edge(edge)
            stats["edges_created"] += 1

        # Save file hashes
        for rel_path in files:
            try:
                fh = _file_hash(rel_path, config.project_root)
                storage.save_file_hash(rel_path, fh, now)
            except Exception:
                pass

        # Phase 4.5: Import context logs (bugs, tasks, phases)
        try:
            log_stats = import_context_logs(storage, config.project_root)
            stats["bugs_imported"] = log_stats["bugs_imported"]
            stats["tasks_imported"] = log_stats["tasks_imported"] + log_stats["phases_imported"]
        except Exception as e:
            stats["errors"].append(f"context_logs: {e}")

        # Phase 5: Compute importance
        compute_importance(storage)

        # Phase 5.1: LSP enrichment — add type info + diagnostics to high-importance nodes
        lsp_server_pref = config.lsp_server if hasattr(config, "lsp_server") else "auto"
        lsp_on = getattr(config, "lsp_enabled", True)
        if lsp_on and lsp_server_pref != "none" and is_lsp_available():
            try:
                _enrich_with_lsp(storage, all_nodes, config.project_root, lsp_server_pref)
            except Exception as e:
                stats["errors"].append(f"lsp: {e}")

        # Phase 5.2: Generate overview nodes (project + directory summaries)
        try:
            ov_stats = generate_and_store_overviews(storage)
            stats["overview_nodes"] = ov_stats["project_overviews"] + ov_stats["directory_overviews"]
        except Exception as e:
            stats["errors"].append(f"overviews: {e}")

        # Phase 5.5: Compute embeddings for ALL nodes (code + bugs + tasks + overviews)
        if embeddings_available():
            all_storage_nodes = storage.get_all_nodes()
            stats["embeddings_computed"] = compute_all_embeddings(storage, all_storage_nodes)

        # Phase 6: Generate summaries (after importance is computed)
        if config.index_mode != IndexMode.AST_ONLY:
            # Only API-summarize high-importance nodes to save tokens
            for node in all_nodes.values():
                if node.importance >= 0.3:
                    # Get context nodes for better summaries
                    context = []
                    for edge in storage.get_edges_from(node.id):
                        ctx_node = storage.get_node(edge.target_id)
                        if ctx_node:
                            context.append(ctx_node)
                    summarize_node(node, config.index_mode, context[:3])
                    storage.upsert_node(node)

        # Save index state
        state = IndexState(
            last_full_index=now,
            last_commit_hash=_get_git_head(config.project_root),
            total_nodes=stats["nodes_created"],
            total_edges=stats["edges_created"],
            total_files=stats["files_processed"],
            index_mode=config.index_mode,
        )
        storage.save_index_state(state)

    return stats


def incremental_update(config: Optional[NeuralConfig] = None, project_root: str = ".") -> dict:
    """Perform an incremental update based on git diff and file hash changes.

    Returns stats about what was updated.
    """
    if config is None:
        config = load_config(project_root)

    redactor = Redactor(config.redaction)
    root = Path(config.project_root).resolve()
    stats = {
        "files_updated": 0,
        "files_added": 0,
        "files_removed": 0,
        "nodes_updated": 0,
        "errors": [],
    }

    with Storage(config.project_root) as storage:
        state = storage.get_index_state()
        indexed_files = storage.get_all_indexed_files()

        # Determine changed files
        changed_files: set[str] = set()

        # Method 1: Git diff
        if state.last_commit_hash:
            git_changes = _get_git_changed_files(state.last_commit_hash, config.project_root)
            for f in git_changes:
                if any(fnmatch.fnmatch(f, pat) for pat in config.include_patterns):
                    changed_files.add(f)

        # Method 2: File hash comparison
        current_files = set(_discover_files(config))

        # New files
        new_files = current_files - set(indexed_files.keys())
        changed_files.update(new_files)

        # Changed files (hash mismatch)
        for f in current_files & set(indexed_files.keys()):
            try:
                current_hash = _file_hash(f, config.project_root)
                if current_hash != indexed_files[f]:
                    changed_files.add(f)
            except Exception:
                pass

        # Removed files — only remove files whose extension is covered by include_patterns.
        # Context logs (.md), overview markers, etc. are stored in file_hashes but are NOT
        # Python source files; removing them would delete bug/task nodes.
        # Extract allowed extensions from include patterns (e.g. "**/*.py" → ".py").
        tracked_exts: set[str] = set()
        for pat in config.include_patterns:
            if "." in pat:
                tracked_exts.add("." + pat.rsplit(".", 1)[1])
        candidate_removed = set(indexed_files.keys()) - current_files
        removed_files = {
            f for f in candidate_removed
            if any(f.endswith(ext) for ext in tracked_exts)
        }

        # Process removals
        for f in removed_files:
            storage.delete_edges_by_file(f)
            storage.delete_nodes_by_file(f)
            storage.delete_file_hash(f)
            stats["files_removed"] += 1

        # Process changes and additions
        all_existing_nodes = {n.id: n for n in storage.get_all_nodes()}

        for rel_path in changed_files:
            full_path = root / rel_path
            if not full_path.exists():
                continue

            try:
                # Remove old data for this file
                storage.delete_edges_by_file(rel_path)
                storage.delete_nodes_by_file(rel_path)

                # Re-parse
                source = full_path.read_text(encoding="utf-8", errors="replace")
                nodes, edges = parse_file(str(rel_path), source=source)

                # Resolve edges against all known nodes
                for node in nodes:
                    all_existing_nodes[node.id] = node
                edges = resolve_edges(all_existing_nodes, edges)

                # Redact
                for node in nodes:
                    if node.raw_code:
                        rc, rs, had = redactor.redact_node_content(node.raw_code, node.summary_short)
                        if had:
                            node.raw_code = rc
                            node.summary_short = rs
                            node.has_redacted_content = True

                # Store
                for node in nodes:
                    storage.upsert_node(node)
                    stats["nodes_updated"] += 1

                for edge in edges:
                    storage.upsert_edge(edge)

                # Update file hash
                now = datetime.now(timezone.utc).isoformat()
                fh = _file_hash(rel_path, config.project_root)
                storage.save_file_hash(rel_path, fh, now)

                if rel_path in new_files:
                    stats["files_added"] += 1
                else:
                    stats["files_updated"] += 1

            except Exception as e:
                stats["errors"].append(f"{rel_path}: {str(e)}")

        # Recompute importance
        compute_importance(storage)

        # Regenerate overview nodes (cheap, always refresh)
        try:
            generate_and_store_overviews(storage)
        except Exception:
            pass

        # Re-embed changed nodes (incremental projection)
        if embeddings_available() and changed_files:
            changed_ids: set[str] = set()
            for f in changed_files:
                for node in storage.get_nodes_by_file(f):
                    changed_ids.add(node.id)
            update_embeddings(storage, changed_ids)

        # Update state
        now = datetime.now(timezone.utc).isoformat()
        state.last_incremental_update = now
        state.last_commit_hash = _get_git_head(config.project_root)
        db_stats = storage.get_stats()
        state.total_nodes = db_stats["total_nodes"]
        state.total_edges = db_stats["total_edges"]
        state.total_files = db_stats["total_files"]
        storage.save_index_state(state)

    return stats
