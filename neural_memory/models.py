"""Data models for neural memory nodes and edges."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class NodeType(str, Enum):
    # ── Codebase (AST-parsed) ──
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    CONFIG = "config"
    EXPORT = "export"
    TYPE_DEF = "type_def"
    OTHER = "other"
    # ── Multi-language constructs ──
    INTERFACE = "interface"          # TS/Go/PHP interfaces
    TRAIT = "trait"                  # Rust traits
    STRUCT = "struct"               # Rust/Go structs
    ENUM = "enum"                   # Rust/TS/Ruby enums
    TYPE_ALIAS = "type_alias"       # TS type aliases, Rust type aliases
    CONSTANT = "constant"           # Module-level constants
    # ── Database / SQL schema ──
    DATABASE = "database"
    TABLE = "table"
    COLUMN = "column"
    VIEW = "view"
    STORED_PROCEDURE = "stored_procedure"
    # ── Codebase (generated overviews) ──
    PROJECT_OVERVIEW = "project_overview"
    DIRECTORY_OVERVIEW = "directory_overview"
    # ── Bugs ──
    BUG = "bug"
    # ── Tasks / Project lifecycle ──
    PHASE = "phase"
    TASK = "task"
    SUBTASK = "subtask"
    # ── Insights (accumulated technical knowledge) ──
    INSIGHT = "insight"


class EdgeType(str, Enum):
    # ── Code structure ──
    CALLS = "calls"
    IMPORTS = "imports"
    INHERITS = "inherits"
    IMPLEMENTS = "implements"       # Rust impl, TS/Java implements
    DEFINES = "defines"
    USES = "uses"
    CONTAINS = "contains"
    # ── Cross-layer (code ↔ bugs/tasks) ──
    RELATES_TO = "relates_to"       # bug/task → code node
    FIXED_BY = "fixed_by"           # bug → code node that fixed it
    # ── Task hierarchy ──
    PHASE_CONTAINS = "phase_contains"   # phase → task
    TASK_CONTAINS = "task_contains"     # task → subtask
    REFERENCES = "references"          # FK relationship between tables
    QUERIES = "queries"                # function/method → table (read access)
    WRITES_TO = "writes_to"           # function/method → table (write access)


# ── Task status constants ──────────────────────────────────────────────────────
VALID_TASK_STATUSES: frozenset[str] = frozenset({"pending", "in_progress", "testing", "done"})
TASK_STATUS_ALIASES: dict[str, str] = {"new": "pending"}

VALID_BUG_STATUSES: frozenset[str] = frozenset({"open", "fixed"})
VALID_PRIORITIES: frozenset[str] = frozenset({"low", "medium", "high"})


class SummaryMode(str, Enum):
    """How the summary was generated."""
    HEURISTIC = "heuristic"
    API = "api"
    BOTH = "both"


class IndexMode(str, Enum):
    """Indexing strategy."""
    AST_ONLY = "ast_only"
    API_ONLY = "api_only"
    BOTH = "both"


@dataclass
class NeuralNode:
    """A single node in the neural memory graph."""
    id: str
    name: str
    node_type: NodeType
    file_path: str
    line_start: int
    line_end: int
    # Layered summaries
    summary_short: str = ""          # 1-2 sentence overview
    summary_detailed: str = ""       # Full explanation with context
    summary_mode: SummaryMode = SummaryMode.HEURISTIC
    # Signature / interface
    signature: str = ""              # e.g., def foo(x: int, y: str) -> bool
    docstring: str = ""
    # Metadata
    complexity: int = 0              # Cyclomatic complexity estimate
    importance: float = 0.0          # 0-1, computed from connectivity + size
    is_public: bool = True
    decorators: list[str] = field(default_factory=list)
    # Content hash for change detection
    content_hash: str = ""
    # Language identifier (populated by multi-language parser)
    language: str = ""             # "python", "typescript", "rust", etc.
    # Raw code (stored but not surfaced unless requested)
    raw_code: str = ""
    # Redacted flag
    has_redacted_content: bool = False
    # ── Layer / category ──────────────────────────────────────────────────────
    # "codebase" | "bugs" | "tasks"  — used for visualization filtering
    category: str = "codebase"
    # ── Lifecycle ─────────────────────────────────────────────────────────────
    # Archived = completed task (done) or fixed bug. Still queryable, but excluded
    # from active context fetches and decayed in importance (×0.3).
    archived: bool = False
    # ── Bug-specific fields ───────────────────────────────────────────────────
    severity: str = ""              # low / medium / high / critical
    bug_status: str = ""            # open / fixed
    # ── Task-specific fields ──────────────────────────────────────────────────
    task_status: str = ""           # pending / in_progress / testing / done  (see VALID_TASK_STATUSES)
    priority: str = ""              # low / medium / high
    # ── LSP-enrichment ────────────────────────────────────────────────────────
    lsp_type_info: str = ""         # resolved type signatures from hover
    lsp_diagnostics: list[str] = field(default_factory=list)
    lsp_hover_doc: str = ""         # documentation text from LSP hover

    def compute_hash(self, source: str) -> str:
        """Compute content hash from source code."""
        self.content_hash = hashlib.sha256(source.encode()).hexdigest()[:16]
        return self.content_hash

    def to_dict(self) -> dict:
        d = asdict(self)
        d["node_type"] = self.node_type.value
        d["summary_mode"] = self.summary_mode.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> NeuralNode:
        import dataclasses
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        filtered["node_type"] = NodeType(filtered["node_type"])
        filtered["summary_mode"] = SummaryMode(filtered["summary_mode"])
        return cls(**filtered)


@dataclass
class NeuralEdge:
    """A directed edge between two neural nodes."""
    source_id: str
    target_id: str
    edge_type: EdgeType
    context: str = ""        # e.g., "called at line 42"
    weight: float = 1.0      # Frequency / importance

    def to_dict(self) -> dict:
        d = asdict(self)
        d["edge_type"] = self.edge_type.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> NeuralEdge:
        data["edge_type"] = EdgeType(data["edge_type"])
        return cls(**data)


@dataclass
class IndexState:
    """Tracks the state of the neural memory index."""
    last_full_index: Optional[str] = None       # ISO timestamp
    last_incremental_update: Optional[str] = None
    last_commit_hash: Optional[str] = None
    total_nodes: int = 0
    total_edges: int = 0
    total_files: int = 0
    index_mode: IndexMode = IndexMode.BOTH
    stale_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["index_mode"] = self.index_mode.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> IndexState:
        data["index_mode"] = IndexMode(data["index_mode"])
        return cls(**data)


@dataclass
class EmbeddingMeta:
    """Corpus-level metadata for the embedding space.

    Stores the vocabulary, IDF weights, and SVD projection matrix so that
    new queries and incremental nodes can be projected into the same space
    without re-fitting the whole corpus.
    """
    vocab: list[str]            # ordered token list (index → token)
    idf: list[float]            # per-token IDF weights (len == len(vocab))
    svd_components: list[list[float]]  # shape [n_components, vocab_size]
    n_components: int           # content vector dimensions (e.g. 100)
    model_version: str = "tfidf-svd-v2"
    total_nodes: int = 0        # node count when this was last fit

    def to_dict(self) -> dict:
        return {
            "vocab": self.vocab,
            "idf": self.idf,
            "svd_components": self.svd_components,
            "n_components": self.n_components,
            "model_version": self.model_version,
            "total_nodes": self.total_nodes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EmbeddingMeta":
        return cls(**data)
