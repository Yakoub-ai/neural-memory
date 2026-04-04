"""Vectorized node embeddings for semantic search and visualization.

Computes a composite float32 vector per node from two sources:
  - Content vector: TF-IDF on tokenized text, projected via truncated SVD
  - Structural vector: graph topology features (edge profile, importance, etc.)

The combined vector captures both *what* a node does (semantics) and
*where* it sits in the architecture (graph role).

Requires: numpy (optional extra). All public functions check is_available()
and silently skip if numpy is not installed.
"""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .models import NeuralNode, EmbeddingMeta
    from .storage import Storage

# ── Constants ──────────────────────────────────────────────────────────────────

_MODEL_VERSION = "tfidf-svd-v2"   # bumped: 23 node types, 11 edge types
_SVD_COMPONENTS = 100    # content vector dimensions (kept compact)
_STRUCT_DIMS = 38        # 23 node one-hot + 2 degrees + 11 edge profile + 2 meta
_TOTAL_DIMS = _SVD_COMPONENTS + _STRUCT_DIMS  # 138 floats = 552 bytes per node
_PRUNE_THRESHOLD = 0.10  # min cosine similarity to explore a branch
_BRANCH_DECAY = 0.5      # score multiplier per graph hop

# Edge-type ordering for the structural profile vector (11 types)
_EDGE_TYPES = [
    "calls", "imports", "inherits", "implements", "defines", "uses", "contains",
    "relates_to", "fixed_by", "phase_contains", "task_contains",
]

# Node-type ordering for the one-hot portion (23 types)
_NODE_TYPES = [
    # Core AST types
    "module", "class", "function", "method", "config", "export", "type_def", "other",
    # Multi-language constructs
    "interface", "trait", "struct", "enum", "type_alias", "constant",
    # SQL constructs
    "table", "view", "stored_procedure",
    # Generated overviews
    "project_overview", "directory_overview",
    # Bugs / tasks
    "bug", "phase", "task", "subtask",
]


# ── Availability check ─────────────────────────────────────────────────────────

def is_available() -> bool:
    """Return True if numpy is importable (vectors extra installed)."""
    try:
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


# ── Tokenizer ─────────────────────────────────────────────────────────────────

_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_NON_WORD_RE = re.compile(r"[^a-zA-Z0-9]+")

def _tokenize(text: str) -> list[str]:
    """Code-aware tokenizer.

    Splits on: snake_case underscores, camelCase boundaries, dots,
    and non-word characters. Returns lowercase tokens of length >= 2.
    """
    # Split camelCase first, then on non-word chars
    text = _CAMEL_RE.sub(" ", text)
    tokens = _NON_WORD_RE.split(text.lower())
    return [t for t in tokens if len(t) >= 2]


def _compose_text(node: "NeuralNode") -> str:
    """Build the text document to embed for a node."""
    parts = [
        node.name,
        node.signature,
        node.docstring,
        node.summary_short,
        node.summary_detailed,
        # LSP enrichment — richer type info for code nodes
        getattr(node, "lsp_type_info", ""),
        getattr(node, "lsp_hover_doc", ""),
    ]
    return " ".join(p for p in parts if p)


# ── TF-IDF corpus fitting ─────────────────────────────────────────────────────

def _build_tfidf(
    nodes: list["NeuralNode"],
) -> tuple[list[list[float]], list[str], list[float]]:
    """Compute TF-IDF matrix for a list of nodes.

    Returns:
        tfidf_matrix: shape [n_nodes, vocab_size]  (sparse as list-of-dicts internally)
        vocab: ordered list of tokens
        idf: IDF weights, parallel to vocab
    """
    import numpy as np

    # Build corpus: tokenized documents
    docs: list[list[str]] = [_tokenize(_compose_text(n)) for n in nodes]

    # Collect global vocab
    vocab_set: set[str] = set()
    for tokens in docs:
        vocab_set.update(tokens)
    vocab = sorted(vocab_set)
    token_to_idx = {t: i for i, t in enumerate(vocab)}
    V = len(vocab)
    N = len(nodes)

    if V == 0 or N == 0:
        return [], [], []

    # TF matrix (raw counts, then normalize to term frequency)
    tf = np.zeros((N, V), dtype=np.float32)
    for d_idx, tokens in enumerate(docs):
        for tok in tokens:
            tf[d_idx, token_to_idx[tok]] += 1
        row_sum = tf[d_idx].sum()
        if row_sum > 0:
            tf[d_idx] /= row_sum

    # IDF: log((N + 1) / (df + 1)) + 1  (scikit-learn smooth variant)
    df = (tf > 0).sum(axis=0).astype(np.float32)  # shape [V]
    idf = np.log((N + 1) / (df + 1)) + 1.0

    # TF-IDF
    tfidf = tf * idf  # broadcast over rows

    return tfidf, vocab, idf.tolist()


# ── Truncated SVD ──────────────────────────────────────────────────────────────

def _truncated_svd(
    matrix,  # np.ndarray [n, V]
    n_components: int,
) -> tuple:  # (reduced [n, k], components [k, V])
    """Project matrix to n_components dimensions via SVD.

    Uses numpy's full SVD on small corpora. For large corpora the top-k
    approximation is achieved by slicing U, S, Vt.
    """
    import numpy as np

    n, V = matrix.shape
    k = min(n_components, n, V)

    # Mean-center
    mean = matrix.mean(axis=0)
    centered = matrix - mean

    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    # U: [n, min(n,V)], S: [min(n,V)], Vt: [min(n,V), V]

    reduced = U[:, :k] * S[:k]          # [n, k]
    components = Vt[:k]                  # [k, V]

    return reduced, components, mean


# ── Structural feature vector ──────────────────────────────────────────────────

def _structural_features(
    node: "NeuralNode",
    storage: "Storage",
    edges_by_node: dict | None = None,
) -> list[float]:
    """Build a fixed-size structural feature vector from graph topology.

    Dimensions (len(_NODE_TYPES) + 5 total):
      [0:N]   node_type one-hot (N = len(_NODE_TYPES) types)
      [N]     in_degree (log-normalized)
      [N+1]   out_degree (log-normalized)
      [N+2:N+12] edge_type profile (10 types, fraction of edges per type)
      [N+12]  complexity (log-normalized, max assumed 50)
      [N+13]  importance (already 0-1)

    Args:
        node: The node to compute features for.
        storage: Storage instance (used for per-node queries when edges_by_node is None).
        edges_by_node: Optional pre-fetched dict from storage.get_all_edges_by_node().
            When provided, avoids per-node SQL queries (bulk path).
    """
    # node_type one-hot
    one_hot = [0.0] * len(_NODE_TYPES)
    try:
        one_hot[_NODE_TYPES.index(node.node_type.value)] = 1.0
    except ValueError:
        pass

    if edges_by_node is not None:
        node_edges = edges_by_node.get(node.id, {"incoming": [], "outgoing": []})
        in_edges = node_edges["incoming"]
        out_edges = node_edges["outgoing"]
    else:
        in_edges = storage.get_edges_to(node.id)
        out_edges = storage.get_edges_from(node.id)

    in_deg = math.log1p(len(in_edges))
    out_deg = math.log1p(len(out_edges))

    # Edge-type profile
    all_edges = in_edges + out_edges
    total_e = max(len(all_edges), 1)
    edge_profile = [0.0] * len(_EDGE_TYPES)
    for edge in all_edges:
        try:
            edge_profile[_EDGE_TYPES.index(edge.edge_type.value)] += 1.0
        except ValueError:
            pass
    edge_profile = [c / total_e for c in edge_profile]

    complexity_norm = math.log1p(node.complexity) / math.log1p(50)

    return one_hot + [in_deg, out_deg] + edge_profile + [complexity_norm, node.importance]


# ── Vector packing / unpacking ─────────────────────────────────────────────────

def _pack(vector) -> bytes:
    """Pack a float32 numpy array to bytes."""
    import numpy as np
    arr = np.asarray(vector, dtype=np.float32)
    return arr.tobytes()


def _unpack(data: bytes, dims: int) -> "np.ndarray":
    """Unpack bytes to a float32 numpy array."""
    import numpy as np
    return np.frombuffer(data, dtype=np.float32)[:dims]


def _cosine(a, b) -> float:
    """Cosine similarity between two 1-D numpy arrays."""
    import numpy as np
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ── Batch embedding computation ────────────────────────────────────────────────

def compute_all_embeddings(storage: "Storage", nodes: list["NeuralNode"]) -> int:
    """Compute and store composite embeddings for all nodes.

    Fits TF-IDF + SVD on the full corpus, then stores per-node vectors
    and the corpus metadata (vocab, IDF, SVD components) to storage.

    Returns the number of nodes embedded.
    """
    if not is_available():
        return 0
    if not nodes:
        return 0

    import numpy as np
    from .models import EmbeddingMeta

    # 1. Fit TF-IDF on full corpus
    tfidf_matrix, vocab, idf = _build_tfidf(nodes)
    if not vocab:
        return 0

    tfidf_np = np.array(tfidf_matrix, dtype=np.float32)

    # 2. Truncated SVD
    k = min(_SVD_COMPONENTS, len(nodes) - 1, len(vocab))
    content_reduced, svd_components, svd_mean = _truncated_svd(tfidf_np, k)
    # content_reduced: [n_nodes, k]

    # Pad or trim to exactly _SVD_COMPONENTS dims
    n = len(nodes)
    content_padded = np.zeros((n, _SVD_COMPONENTS), dtype=np.float32)
    content_padded[:, :k] = content_reduced

    # 3. Structural features — pre-fetch all edges in one query to avoid N+1 pattern
    edges_by_node = storage.get_all_edges_by_node()
    struct_matrix = np.zeros((n, _STRUCT_DIMS), dtype=np.float32)
    for i, node in enumerate(nodes):
        struct_matrix[i] = _structural_features(node, storage, edges_by_node=edges_by_node)

    # 4. Combine + L2-normalize
    combined = np.concatenate([content_padded, struct_matrix], axis=1)  # [n, TOTAL_DIMS]
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    combined /= norms

    # 5. Store per-node vectors
    for i, node in enumerate(nodes):
        storage.upsert_embedding(node.id, _pack(combined[i]), node.content_hash)

    # 6. Store corpus metadata (vocab + IDF + SVD components)
    # Pad SVD components matrix to _SVD_COMPONENTS rows if needed
    svd_components_full = np.zeros((_SVD_COMPONENTS, len(vocab)), dtype=np.float32)
    svd_components_full[:k] = svd_components
    svd_mean_full = np.zeros(len(vocab), dtype=np.float32)
    svd_mean_full[:len(vocab)] = svd_mean

    meta = EmbeddingMeta(
        vocab=vocab,
        idf=idf,
        svd_components=svd_components_full.tolist(),
        n_components=_SVD_COMPONENTS,
        model_version=_MODEL_VERSION,
        total_nodes=len(nodes),
    )
    storage.save_embedding_meta(meta)

    return len(nodes)


def update_embeddings(storage: "Storage", changed_node_ids: set[str]) -> int:
    """Incrementally re-embed changed nodes, projecting into the existing space.

    Does NOT refit the corpus — projects into the stored SVD space.
    If too many nodes changed (>20% of corpus), or if the model version has
    changed (e.g. v1→v2), triggers a full refit.

    Returns the number of nodes re-embedded.
    """
    if not is_available():
        return 0
    if not changed_node_ids:
        return 0

    import numpy as np

    meta = storage.get_embedding_meta()
    if meta is None:
        # No existing space — need full compute
        all_nodes = storage.get_all_nodes()
        return compute_all_embeddings(storage, all_nodes)

    # Version mismatch: force full recompute
    if getattr(meta, "model_version", None) != _MODEL_VERSION:
        all_nodes = storage.get_all_nodes()
        return compute_all_embeddings(storage, all_nodes)

    # Check if full refit is warranted
    total = meta.total_nodes or 1
    if len(changed_node_ids) / total > 0.2:
        all_nodes = storage.get_all_nodes()
        return compute_all_embeddings(storage, all_nodes)

    # Reproject changed nodes into existing SVD space
    vocab = meta.vocab
    token_to_idx = {t: i for i, t in enumerate(vocab)}
    idf = np.array(meta.idf, dtype=np.float32)
    svd_components = np.array(meta.svd_components, dtype=np.float32)  # [k, V]
    k = min(_SVD_COMPONENTS, svd_components.shape[0])

    count = 0
    for node_id in changed_node_ids:
        node = storage.get_node(node_id)
        if node is None:
            continue

        # TF for this node
        tokens = _tokenize(_compose_text(node))
        tf_vec = np.zeros(len(vocab), dtype=np.float32)
        for tok in tokens:
            if tok in token_to_idx:
                tf_vec[token_to_idx[tok]] += 1
        s = tf_vec.sum()
        if s > 0:
            tf_vec /= s
        tfidf_vec = tf_vec * idf

        # Project via stored SVD components: x_reduced = (x - mean) @ Vt.T
        # We don't store mean separately (approximation: use zero mean for incremental)
        content_reduced = tfidf_vec @ svd_components[:k].T  # [k]
        content_padded = np.zeros(_SVD_COMPONENTS, dtype=np.float32)
        content_padded[:k] = content_reduced

        struct_vec = np.array(_structural_features(node, storage), dtype=np.float32)

        combined = np.concatenate([content_padded, struct_vec])
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined /= norm

        storage.upsert_embedding(node_id, _pack(combined), node.content_hash)
        count += 1

    return count


# ── Query projection ───────────────────────────────────────────────────────────

def embed_query(query: str, meta: "EmbeddingMeta") -> Optional["np.ndarray"]:
    """Project a query string into the stored embedding space.

    Returns a float32 array of shape [TOTAL_DIMS], or None if unavailable.
    """
    if not is_available():
        return None

    import numpy as np

    vocab = meta.vocab
    if not vocab:
        return None

    token_to_idx = {t: i for i, t in enumerate(vocab)}
    idf = np.array(meta.idf, dtype=np.float32)
    svd_components = np.array(meta.svd_components, dtype=np.float32)
    k = min(_SVD_COMPONENTS, svd_components.shape[0])

    tokens = _tokenize(query)
    tf_vec = np.zeros(len(vocab), dtype=np.float32)
    for tok in tokens:
        if tok in token_to_idx:
            tf_vec[token_to_idx[tok]] += 1
    s = tf_vec.sum()
    if s > 0:
        tf_vec /= s
    tfidf_vec = tf_vec * idf

    content_reduced = tfidf_vec @ svd_components[:k].T
    content_padded = np.zeros(_SVD_COMPONENTS, dtype=np.float32)
    content_padded[:k] = content_reduced

    # Structural part for a query is all zeros (unknown topology)
    struct_zeros = np.zeros(_STRUCT_DIMS, dtype=np.float32)

    combined = np.concatenate([content_padded, struct_zeros])
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined /= norm

    return combined


# ── Search result ──────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """A ranked result from semantic_search()."""
    node: "NeuralNode"
    score: float                # final combined score (0-1)
    vector_similarity: float    # raw cosine similarity to query
    match_type: str             # "semantic", "branch", or "name_match"
    connections_summary: str    # compact 1-line edge overview


def _connections_summary(node: "NeuralNode", storage: "Storage") -> str:
    """Build a compact edge overview string, e.g. '→ 3 calls, ← 2 callers'."""
    out_edges = storage.get_edges_from(node.id)
    in_edges = storage.get_edges_to(node.id)
    parts = []
    if out_edges:
        parts.append(f"→ {len(out_edges)} out")
    if in_edges:
        parts.append(f"← {len(in_edges)} in")
    return ", ".join(parts) if parts else "no connections"


# ── Semantic + branch search ───────────────────────────────────────────────────

def semantic_search(
    storage: "Storage",
    query: str,
    limit: int = 10,
    weights: Optional[dict] = None,
) -> list[SearchResult]:
    """Three-phase semantic search with graph-guided branch expansion.

    Phase 1 — Seed: cosine similarity to find top-K anchor nodes.
    Phase 2 — Expand: walk edges from seeds, pruning divergent branches.
    Phase 3 — Rank: combine vector similarity + importance + graph score.

    Falls back to an empty list if numpy / embeddings unavailable
    (caller should fall back to storage.search_nodes()).
    """
    if not is_available():
        return []

    import numpy as np

    meta = storage.get_embedding_meta()
    if meta is None:
        return []

    query_vec = embed_query(query, meta)
    if query_vec is None:
        return []

    w = {"semantic": 0.5, "importance": 0.3, "branch": 0.2}
    if weights:
        w.update(weights)

    # ── Phase 1: Seed ────────────────────────────────────────────────────────
    all_embeddings = storage.get_all_embeddings()
    if not all_embeddings:
        return []

    node_ids = list(all_embeddings.keys())
    matrix = np.stack(
        [_unpack(all_embeddings[nid], _TOTAL_DIMS) for nid in node_ids]
    )  # [n, TOTAL_DIMS]

    sims = matrix @ query_vec  # cosine (vectors are L2-normalized)

    seed_k = min(10, len(node_ids))
    top_idx = np.argpartition(sims, -seed_k)[-seed_k:]
    top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

    seeds: dict[str, float] = {}  # node_id → vector_similarity
    for idx in top_idx:
        sim = float(sims[idx])
        if sim > 0:
            seeds[node_ids[idx]] = sim

    # Supplement seeds with any nodes whose name contains the query as a substring.
    # This ensures "authentication" still surfaces "authenticate_user" even when
    # vocabulary is disjoint (no stemming). We load all nodes only once here.
    query_lower = query.lower()
    if not seeds:
        # Only do the full scan when vector seeds are empty (zero-sim query)
        for nid in node_ids:
            node_temp = storage.get_node(nid)
            if node_temp and query_lower in node_temp.name.lower():
                seeds[nid] = 0.0  # no vector similarity, but name match

    # ── Phase 2: Branch expansion ─────────────────────────────────────────────
    node_sim_map = {node_ids[i]: float(sims[i]) for i in range(len(node_ids))}
    branch_scores: dict[str, float] = {}  # node_id → branch propagation score

    def _expand(node_id: str, seed_score: float, depth: int) -> None:
        if depth == 0:
            return
        for edge in storage.get_edges_from(node_id) + storage.get_edges_to(node_id):
            neighbor_id = edge.target_id if edge.source_id == node_id else edge.source_id
            if neighbor_id not in all_embeddings:
                continue
            # Prune divergent branches
            neighbor_sim = node_sim_map.get(neighbor_id, 0.0)
            if neighbor_sim < _PRUNE_THRESHOLD:
                continue
            prop_score = seed_score * (_BRANCH_DECAY ** (3 - depth)) * edge.weight
            if prop_score > branch_scores.get(neighbor_id, 0.0):
                branch_scores[neighbor_id] = prop_score
                _expand(neighbor_id, prop_score, depth - 1)

    for seed_id, seed_sim in seeds.items():
        _expand(seed_id, seed_sim, depth=2)

    # ── Phase 3: Rank ─────────────────────────────────────────────────────────
    candidate_ids = set(seeds.keys()) | set(branch_scores.keys())

    results: list[SearchResult] = []

    for node_id in candidate_ids:
        node = storage.get_node(node_id)
        if node is None:
            continue

        vec_sim = node_sim_map.get(node_id, 0.0)
        b_score = branch_scores.get(node_id, 0.0)
        name_bonus = 0.15 if query_lower in node.name.lower() else 0.0

        score = (
            w["semantic"] * vec_sim
            + w["importance"] * node.importance
            + w["branch"] * b_score
            + name_bonus
        )

        if node_id in seeds:
            match_type = "semantic"
        else:
            match_type = "branch"
        if name_bonus:
            match_type = "name_match"

        results.append(SearchResult(
            node=node,
            score=score,
            vector_similarity=vec_sim,
            match_type=match_type,
            connections_summary=_connections_summary(node, storage),
        ))

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:limit]
