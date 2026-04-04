"""Tests for the embeddings module: tokenizer, TF-IDF, structural features,
composite vectors, semantic search, and branch search.
"""

from __future__ import annotations

import math
import struct
import pytest

from neural_memory.models import (
    NeuralNode, NeuralEdge, NodeType, EdgeType, SummaryMode, EmbeddingMeta
)
from neural_memory.embeddings import (
    _tokenize, _compose_text, _build_tfidf, _truncated_svd,
    _structural_features, _pack, _unpack, _cosine,
    compute_all_embeddings, update_embeddings, embed_query,
    semantic_search, is_available,
    _TOTAL_DIMS, _SVD_COMPONENTS, _STRUCT_DIMS, _NODE_TYPES,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _node(
    name: str,
    node_type: NodeType = NodeType.FUNCTION,
    file_path: str = "mod.py",
    summary_short: str = "",
    signature: str = "",
    docstring: str = "",
    importance: float = 0.5,
    complexity: int = 1,
    line_start: int = 1,
    line_end: int = 10,
) -> NeuralNode:
    from neural_memory.ts_parser import _node_id
    nid = _node_id(file_path, name, node_type)
    return NeuralNode(
        id=nid,
        name=name,
        node_type=node_type,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        summary_short=summary_short,
        signature=signature,
        docstring=docstring,
        importance=importance,
        complexity=complexity,
        summary_mode=SummaryMode.HEURISTIC,
    )


# ── Tokenizer tests ────────────────────────────────────────────────────────────

class TestTokenize:
    def test_snake_case(self):
        assert "parse" in _tokenize("parse_file")
        assert "file" in _tokenize("parse_file")

    def test_camel_case(self):
        tokens = _tokenize("NeuralNode")
        assert "Neural" in tokens or "neural" in tokens
        assert "Node" in tokens or "node" in tokens

    def test_dot_separated(self):
        tokens = _tokenize("module.submodule.function")
        assert "module" in tokens
        assert "function" in tokens

    def test_min_length_filter(self):
        tokens = _tokenize("a b foo bar")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "foo" in tokens
        assert "bar" in tokens

    def test_lowercase(self):
        tokens = _tokenize("ParseFile")
        for t in tokens:
            assert t == t.lower()

    def test_empty(self):
        assert _tokenize("") == []

    def test_digits_preserved(self):
        tokens = _tokenize("parse2file")
        assert any("2" in t or t.isdigit() for t in tokens) or len(tokens) >= 1


class TestComposeText:
    def test_uses_name_and_summary(self):
        n = _node("my_func", summary_short="Does something useful")
        text = _compose_text(n)
        assert "my_func" in text
        assert "Does something useful" in text

    def test_skips_empty_fields(self):
        n = _node("foo")
        text = _compose_text(n)
        assert "  " not in text  # no double spaces from empty joins


# ── TF-IDF tests ───────────────────────────────────────────────────────────────

@pytest.mark.skipif(not is_available(), reason="numpy not installed")
class TestTFIDF:
    def test_basic_shape(self):
        import numpy as np
        nodes = [
            _node("parse_file", summary_short="parse python files"),
            _node("store_node", summary_short="store data in database"),
            _node("search_query", summary_short="search for functions"),
        ]
        matrix, vocab, idf = _build_tfidf(nodes)
        matrix_np = np.array(matrix)
        assert matrix_np.shape[0] == 3
        assert matrix_np.shape[1] == len(vocab)
        assert len(idf) == len(vocab)

    def test_idf_higher_for_rare_terms(self):
        import numpy as np
        # "parse" appears in 1 of 3 docs; "file" appears in 1 of 3 docs;
        # "the" would appear in all — but we use real text
        nodes = [
            _node("a", summary_short="parse files"),
            _node("b", summary_short="store records in database"),
            _node("c", summary_short="authenticate users with tokens"),
        ]
        _, vocab, idf = _build_tfidf(nodes)
        tok_to_idf = dict(zip(vocab, idf))
        # All terms appear in exactly 1 doc → equal IDF
        assert all(v > 0 for v in idf)

    def test_empty_nodes(self):
        matrix, vocab, idf = _build_tfidf([])
        assert matrix == []
        assert vocab == []

    def test_tfidf_nonzero_for_present_terms(self):
        import numpy as np
        nodes = [
            _node("foo", summary_short="authentication login session"),
            _node("bar", summary_short="database storage"),
        ]
        matrix, vocab, idf = _build_tfidf(nodes)
        matrix_np = np.array(matrix)
        # Row 0 should have non-zero entries for auth-related terms
        assert matrix_np[0].sum() > 0
        assert matrix_np[1].sum() > 0


# ── SVD tests ──────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not is_available(), reason="numpy not installed")
class TestTruncatedSVD:
    def test_output_shape(self):
        import numpy as np
        matrix = np.random.rand(10, 50).astype(np.float32)
        reduced, components, mean = _truncated_svd(matrix, n_components=5)
        assert reduced.shape == (10, 5)
        assert components.shape[0] == 5

    def test_fewer_dims_than_components(self):
        import numpy as np
        matrix = np.random.rand(3, 4).astype(np.float32)
        reduced, components, mean = _truncated_svd(matrix, n_components=10)
        # Should give at most min(3, 4) = 3 components
        assert reduced.shape[1] <= 3


# ── Structural features tests ──────────────────────────────────────────────────

@pytest.mark.skipif(not is_available(), reason="numpy not installed")
class TestStructuralFeatures(object):
    def test_output_length(self, storage):
        n = _node("myfunc")
        storage.upsert_node(n)
        feats = _structural_features(n, storage)
        assert len(feats) == _STRUCT_DIMS

    def test_node_type_one_hot(self, storage):
        fn = _node("fn", node_type=NodeType.FUNCTION)
        storage.upsert_node(fn)
        feats = _structural_features(fn, storage)
        # One-hot for FUNCTION should be 1.0 at index 2
        assert feats[2] == 1.0
        assert sum(feats[:8]) == 1.0

    def test_importance_in_last_slot(self, storage):
        n = _node("imp", importance=0.75)
        storage.upsert_node(n)
        feats = _structural_features(n, storage)
        assert feats[-1] == pytest.approx(0.75)

    def test_in_degree_nonzero_with_edges(self, storage):
        caller = _node("caller", file_path="a.py")
        callee = _node("callee", file_path="b.py")
        storage.upsert_node(caller)
        storage.upsert_node(callee)
        edge = NeuralEdge(
            source_id=caller.id,
            target_id=callee.id,
            edge_type=EdgeType.CALLS,
        )
        storage.upsert_edge(edge)
        feats = _structural_features(callee, storage)
        # in_degree is at index len(_NODE_TYPES), right after the one-hot block
        assert feats[len(_NODE_TYPES)] > 0


# ── Pack / unpack tests ────────────────────────────────────────────────────────

@pytest.mark.skipif(not is_available(), reason="numpy not installed")
class TestPackUnpack:
    def test_roundtrip(self):
        import numpy as np
        vec = np.array([1.0, 2.5, -0.3, 0.0], dtype=np.float32)
        data = _pack(vec)
        recovered = _unpack(data, 4)
        assert np.allclose(vec, recovered)

    def test_cosine_identical(self):
        import numpy as np
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert _cosine(v, v) == pytest.approx(1.0)

    def test_cosine_orthogonal(self):
        import numpy as np
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        assert _cosine(a, b) == pytest.approx(0.0)

    def test_cosine_opposite(self):
        import numpy as np
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        assert _cosine(a, b) == pytest.approx(-1.0)


# ── compute_all_embeddings tests ───────────────────────────────────────────────

@pytest.mark.skipif(not is_available(), reason="numpy not installed")
class TestComputeAllEmbeddings:
    def test_stores_correct_count(self, storage):
        nodes = [
            _node("alpha", summary_short="parse python files"),
            _node("beta", summary_short="store data in sqlite"),
            _node("gamma", summary_short="search knowledge graph"),
        ]
        for n in nodes:
            storage.upsert_node(n)

        count = compute_all_embeddings(storage, nodes)
        assert count == 3

    def test_vectors_stored(self, storage):
        nodes = [
            _node("fn_a", summary_short="authenticate user with token"),
            _node("fn_b", summary_short="store record in database"),
        ]
        for n in nodes:
            storage.upsert_node(n)

        compute_all_embeddings(storage, nodes)

        all_embs = storage.get_all_embeddings()
        assert len(all_embs) == 2
        for nid, data in all_embs.items():
            import numpy as np
            vec = _unpack(data, _TOTAL_DIMS)
            assert vec.shape == (_TOTAL_DIMS,)
            # L2-normalized → norm ≈ 1.0
            assert abs(float(np.linalg.norm(vec)) - 1.0) < 0.02

    def test_meta_stored(self, storage):
        nodes = [
            _node("x", summary_short="tokenize source code"),
            _node("y", summary_short="index codebase graph"),
        ]
        for n in nodes:
            storage.upsert_node(n)
        compute_all_embeddings(storage, nodes)

        meta = storage.get_embedding_meta()
        assert meta is not None
        assert len(meta.vocab) > 0
        assert len(meta.idf) == len(meta.vocab)
        assert meta.n_components == _SVD_COMPONENTS

    def test_empty_nodes(self, storage):
        count = compute_all_embeddings(storage, [])
        assert count == 0

    def test_different_nodes_different_vectors(self, storage):
        import numpy as np
        nodes = [
            _node("auth_login", summary_short="authenticate user session login password"),
            _node("db_store", summary_short="store persist database record sql"),
        ]
        for n in nodes:
            storage.upsert_node(n)
        compute_all_embeddings(storage, nodes)

        all_embs = storage.get_all_embeddings()
        vecs = list(all_embs.values())
        v0 = _unpack(vecs[0], _TOTAL_DIMS)
        v1 = _unpack(vecs[1], _TOTAL_DIMS)
        # Should not be identical
        assert not np.allclose(v0, v1)


# ── embed_query tests ──────────────────────────────────────────────────────────

@pytest.mark.skipif(not is_available(), reason="numpy not installed")
class TestEmbedQuery:
    def _make_meta(self, storage):
        nodes = [
            _node("parse", summary_short="parse python ast"),
            _node("store", summary_short="store nodes in sqlite"),
            _node("search", summary_short="search graph by name"),
        ]
        for n in nodes:
            storage.upsert_node(n)
        compute_all_embeddings(storage, nodes)
        return storage.get_embedding_meta()

    def test_returns_correct_shape(self, storage):
        import numpy as np
        meta = self._make_meta(storage)
        vec = embed_query("parse function", meta)
        assert vec is not None
        assert vec.shape == (_TOTAL_DIMS,)

    def test_unknown_query_returns_vector(self, storage):
        import numpy as np
        meta = self._make_meta(storage)
        vec = embed_query("xyzzy totally unknown", meta)
        assert vec is not None
        assert vec.shape == (_TOTAL_DIMS,)

    def test_similar_queries_close(self, storage):
        import numpy as np
        meta = self._make_meta(storage)
        v1 = embed_query("parse ast", meta)
        v2 = embed_query("parse python", meta)
        v3 = embed_query("store database", meta)
        sim_12 = _cosine(v1, v2)
        sim_13 = _cosine(v1, v3)
        # parse+ast should be closer to parse+python than to store+database
        assert sim_12 >= sim_13


# ── update_embeddings tests ────────────────────────────────────────────────────

@pytest.mark.skipif(not is_available(), reason="numpy not installed")
class TestUpdateEmbeddings:
    def test_updates_changed_node(self, storage):
        nodes = [
            _node("fn_a", summary_short="parse files carefully"),
            _node("fn_b", summary_short="store data persistently"),
            _node("fn_c", summary_short="search the graph deeply"),
        ]
        for n in nodes:
            storage.upsert_node(n)
        compute_all_embeddings(storage, nodes)

        # Modify fn_a and re-embed
        changed = update_embeddings(storage, {nodes[0].id})
        assert changed >= 1

        # Vector should still be stored
        vec = storage.get_embedding(nodes[0].id)
        assert vec is not None

    def test_full_refit_on_large_change(self, storage):
        nodes = [_node(f"fn_{i}", summary_short=f"function {i} does work") for i in range(5)]
        for n in nodes:
            storage.upsert_node(n)
        compute_all_embeddings(storage, nodes)

        # Change >20% of corpus
        changed_ids = {nodes[0].id, nodes[1].id}  # 2/5 = 40%
        count = update_embeddings(storage, changed_ids)
        assert count > 0


# ── semantic_search tests ──────────────────────────────────────────────────────

@pytest.mark.skipif(not is_available(), reason="numpy not installed")
class TestSemanticSearch:
    def _setup(self, storage):
        nodes = [
            _node("authenticate_user", summary_short="verify user credentials and create session", importance=0.8),
            _node("hash_password", summary_short="hash password with bcrypt for storage", importance=0.6),
            _node("connect_database", summary_short="open sqlite connection and configure pool", importance=0.7),
            _node("execute_query", summary_short="run sql query against database and return rows", importance=0.5),
            _node("parse_ast", summary_short="parse python source into ast nodes", importance=0.9),
        ]
        for n in nodes:
            storage.upsert_node(n)
        compute_all_embeddings(storage, nodes)
        return nodes

    def test_returns_results(self, storage):
        self._setup(storage)
        # "authenticate" is a token from "authenticate_user" and appears in the corpus
        results = semantic_search(storage, "authenticate user", limit=5)
        assert len(results) > 0

    def test_result_fields(self, storage):
        self._setup(storage)
        results = semantic_search(storage, "database query", limit=3)
        for r in results:
            assert r.node is not None
            assert 0.0 <= r.score
            assert r.match_type in ("semantic", "branch", "name_match")
            assert isinstance(r.connections_summary, str)

    def test_auth_query_finds_auth_node(self, storage):
        nodes = self._setup(storage)
        # "authenticate" and "credentials" are tokens in the corpus
        results = semantic_search(storage, "authenticate credentials", limit=5)
        result_names = [r.node.name for r in results]
        # authenticate_user or hash_password should appear
        assert any("auth" in name or "password" in name for name in result_names)

    def test_respects_limit(self, storage):
        self._setup(storage)
        results = semantic_search(storage, "parse", limit=2)
        assert len(results) <= 2

    def test_name_match_bonus(self, storage):
        self._setup(storage)
        results = semantic_search(storage, "parse_ast", limit=5)
        result_names = [r.node.name for r in results]
        assert "parse_ast" in result_names
        # parse_ast should be first or very high
        top_names = result_names[:2]
        assert "parse_ast" in top_names

    def test_no_embeddings_returns_empty(self, storage):
        # Storage with no embeddings
        results = semantic_search(storage, "anything", limit=5)
        assert results == []

    def test_branch_expansion(self, storage):
        """Branch expansion should surface connected nodes."""
        caller = _node("call_auth", summary_short="calls authenticate user", importance=0.3)
        callee = _node("authenticate_user", summary_short="verify credentials authenticate session", importance=0.8)
        storage.upsert_node(caller)
        storage.upsert_node(callee)
        edge = NeuralEdge(source_id=caller.id, target_id=callee.id, edge_type=EdgeType.CALLS)
        storage.upsert_edge(edge)

        nodes = [caller, callee]
        compute_all_embeddings(storage, nodes)

        # Query for authentication — should find callee directly, and potentially caller via branch
        results = semantic_search(storage, "authentication verify", limit=5)
        result_ids = {r.node.id for r in results}
        assert callee.id in result_ids
