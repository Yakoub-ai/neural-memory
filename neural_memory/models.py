"""Data models for neural memory nodes and edges."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class NodeType(str, Enum):
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    CONFIG = "config"
    EXPORT = "export"
    TYPE_DEF = "type_def"
    OTHER = "other"


class EdgeType(str, Enum):
    CALLS = "calls"
    IMPORTS = "imports"
    INHERITS = "inherits"
    DEFINES = "defines"
    USES = "uses"
    CONTAINS = "contains"


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
    # Raw code (stored but not surfaced unless requested)
    raw_code: str = ""
    # Redacted flag
    has_redacted_content: bool = False

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
        data["node_type"] = NodeType(data["node_type"])
        data["summary_mode"] = SummaryMode(data["summary_mode"])
        return cls(**data)


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
