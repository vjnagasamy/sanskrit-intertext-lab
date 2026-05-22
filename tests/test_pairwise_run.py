"""Tests for canonical Pairwise Similarity Run core logic."""

from __future__ import annotations

import unittest

import numpy as np

from sanskrit_pipeline.pairwise_run import PairwiseMetrics, make_segments, run_pairwise_similarity_core, top_k_match_records


class PairwiseRunCoreTests(unittest.TestCase):
    def test_core_preserves_segment_metadata_and_metrics(self) -> None:
        segments_a = make_segments(["a0", "a1"], spans=[(0, 2), (3, 5)])
        segments_b = make_segments(["b0", "b1"], spans=[(10, 12), (13, 15)])
        embeddings_a = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        embeddings_b = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32)

        result = run_pairwise_similarity_core(
            segments_a,
            embeddings_a,
            segments_b,
            embeddings_b,
            top_k=3,
        )

        self.assertEqual(result.segments_a[0].start, 0)
        self.assertEqual(result.segments_b[1].end, 15)
        self.assertEqual(result.similarity_matrix.shape, (2, 2))
        self.assertEqual(result.matches[0].segment_a.index, 0)
        self.assertEqual(result.matches[0].segment_b.index, 0)
        self.assertIsInstance(result.metrics, PairwiseMetrics)
        self.assertAlmostEqual(result.metrics.max_score, 1.0, places=6)

    def test_core_allows_empty_input(self) -> None:
        result = run_pairwise_similarity_core(
            [],
            np.empty((0, 4), dtype=np.float32),
            [],
            np.empty((0, 4), dtype=np.float32),
            top_k=5,
        )

        self.assertEqual(result.similarity_matrix.shape, (0, 0))
        self.assertEqual(result.matches, [])
        self.assertEqual(result.metrics.mean_score, 0.0)

    def test_top_k_modes_can_require_unique_or_diverse_sides(self) -> None:
        matrix = np.array(
            [
                [0.99, 0.98, 0.10, 0.09],
                [0.97, 0.96, 0.50, 0.08],
                [0.20, 0.19, 0.95, 0.94],
                [0.18, 0.17, 0.93, 0.92],
            ],
            dtype=np.float32,
        )
        segments_a = make_segments(["a0", "a1", "a2", "a3"])
        segments_b = make_segments(["b0", "b1", "b2", "b3"])

        raw = top_k_match_records(matrix, segments_a, segments_b, 4, mode="raw")
        self.assertEqual([(m.segment_a.index, m.segment_b.index) for m in raw], [(0, 0), (0, 1), (1, 0), (1, 1)])

        unique_a = top_k_match_records(matrix, segments_a, segments_b, 4, mode="unique_a")
        self.assertEqual([m.segment_a.index for m in unique_a], [0, 1, 2, 3])

        unique_b = top_k_match_records(matrix, segments_a, segments_b, 4, mode="unique_b")
        self.assertEqual([m.segment_b.index for m in unique_b], [0, 1, 2, 3])

        unique_both = top_k_match_records(matrix, segments_a, segments_b, 4, mode="unique_both")
        self.assertEqual([(m.segment_a.index, m.segment_b.index) for m in unique_both], [(0, 0), (1, 1), (2, 2), (3, 3)])

        diverse_both = top_k_match_records(
            matrix,
            segments_a,
            segments_b,
            4,
            mode="diverse_both",
            diversity_radius=1,
        )
        self.assertEqual([(m.segment_a.index, m.segment_b.index) for m in diverse_both], [(0, 0), (2, 2)])


if __name__ == "__main__":
    unittest.main()
