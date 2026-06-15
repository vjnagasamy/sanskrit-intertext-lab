"""Unit tests for FAISS approximate nearest-neighbour search."""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from sanskrit_pipeline.ann_search import ANNIndex


class _FakeFlatIndex:
    def __init__(self, dim: int) -> None:
        self.d = dim
        self.ntotal = 0
        self._vectors: np.ndarray | None = None

    def add(self, vectors: np.ndarray) -> None:
        self._vectors = vectors
        self.ntotal = vectors.shape[0]

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        scores = queries @ self._vectors.T
        indices = np.argsort(-scores, axis=1)[:, :k]
        top_scores = np.take_along_axis(scores, indices, axis=1)
        return top_scores.astype(np.float32), indices.astype(np.int64)


class _FakeFaissModule(types.ModuleType):
    METRIC_INNER_PRODUCT = 0

    IndexFlatIP = _FakeFlatIndex

    class IndexIVFFlat(_FakeFlatIndex):
        def __init__(self, quantizer: object, dim: int, nlist: int, metric: int) -> None:
            super().__init__(dim)
            self.nlist = nlist
            self.nprobe = 1

        def train(self, vectors: np.ndarray) -> None:
            self._train_vectors = vectors

    @staticmethod
    def get_num_gpus() -> int:
        return 0

    @staticmethod
    def write_index(index: object, path: str) -> None:
        pass

    @staticmethod
    def read_index(path: str) -> object:
        return _FakeFlatIndex(3)


def _install_fake_faiss() -> None:
    sys.modules["faiss"] = _FakeFaissModule("faiss")


class ANNSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        _install_fake_faiss()

    def test_flat_index_matches_brute_force_cosine(self) -> None:
        rng = np.random.default_rng(0)
        corpus = rng.normal(size=(8, 16)).astype(np.float32)
        queries = rng.normal(size=(4, 16)).astype(np.float32)

        index = ANNIndex(corpus, use_gpu=False)
        scores, indices = index.search(queries, top_k=3)

        corpus_norm = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
        query_norm = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        expected = query_norm @ corpus_norm.T
        for row in range(queries.shape[0]):
            expected_indices = np.argsort(-expected[row])[:3]
            np.testing.assert_array_equal(indices[row], expected_indices)
            np.testing.assert_allclose(scores[row], expected[row, expected_indices], rtol=1e-5)

    def test_ivfflat_branch_selected_when_n_exceeds_threshold(self) -> None:
        from sanskrit_pipeline import ann_search

        corpus = np.random.default_rng(1).normal(size=(5, 8)).astype(np.float32)
        with patch.object(ann_search, "_FLAT_THRESHOLD", 4):
            index = ann_search.ANNIndex(corpus, use_gpu=False)
        self.assertEqual(index._index_type, "ivf")

    def test_run_ann_returns_ranked_pairs(self) -> None:
        from sanskrit_pipeline.pairwise_run import make_segments, run_ann

        segments_a = make_segments(["a", "b"])
        segments_b = make_segments(["x", "y"])
        embeddings_a = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        embeddings_b = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        result = run_ann(
            segments_a,
            embeddings_a,
            segments_b,
            embeddings_b,
            top_k=2,
            use_gpu=False,
        )
        self.assertEqual(len(result.matches), 2)
        self.assertEqual(result.matches[0].segment_a.text, "a")
        self.assertEqual(result.matches[0].segment_b.text, "x")

    def test_missing_faiss_raises_clear_error(self) -> None:
        original = sys.modules.pop("faiss", None)
        try:
            importlib.reload(importlib.import_module("sanskrit_pipeline.ann_search"))
            from sanskrit_pipeline.ann_search import ANNIndex as ReloadedANNIndex

            with self.assertRaises(ImportError) as ctx:
                ReloadedANNIndex(np.ones((2, 3), dtype=np.float32))
            self.assertIn("faiss-cpu", str(ctx.exception))
        finally:
            if original is not None:
                sys.modules["faiss"] = original
            else:
                _install_fake_faiss()
            importlib.reload(importlib.import_module("sanskrit_pipeline.ann_search"))


if __name__ == "__main__":
    unittest.main()
