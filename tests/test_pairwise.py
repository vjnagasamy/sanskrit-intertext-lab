"""Tests for pairwise sentence similarity utilities."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sanskrit_pipeline.embeddings import EmbeddingResult
from sanskrit_pipeline.pairwise import cosine_similarity_matrix, global_top_k_matches, run_pairwise_similarity


class PairwiseTests(unittest.TestCase):
    def test_cosine_similarity_matrix_shape_and_values(self) -> None:
        a = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        b = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        matrix = cosine_similarity_matrix(a, b)
        self.assertEqual(matrix.shape, (2, 2))
        self.assertAlmostEqual(float(matrix[0, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(matrix[1, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(matrix[1, 1]), 0.7071067, places=5)

    def test_global_top_k_matches_returns_descending_scores(self) -> None:
        matrix = np.array([[0.8, 0.2], [0.9, 0.7]], dtype=np.float32)
        matches = global_top_k_matches(
            matrix=matrix,
            sentences_a=["a0", "a1"],
            sentences_b=["b0", "b1"],
            k=3,
        )
        self.assertEqual(len(matches), 3)
        self.assertEqual((matches[0].i, matches[0].j), (1, 0))
        self.assertEqual((matches[1].i, matches[1].j), (0, 0))
        self.assertEqual((matches[2].i, matches[2].j), (1, 1))
        self.assertGreaterEqual(matches[0].score, matches[1].score)
        self.assertGreaterEqual(matches[1].score, matches[2].score)

    def test_run_pairwise_similarity_writes_csv_jsonl_and_manifest(self) -> None:
        sentences_a = ["धर्मो रक्षति॥", "सत्यमेव जयते॥"]
        sentences_b = ["धर्मो रक्षति॥", "अहिंसा परमो धर्मः॥", "सत्यमेव जयते॥"]

        def fake_encode(texts: list[str]) -> EmbeddingResult:
            if texts == sentences_a:
                embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            elif texts == sentences_b:
                embeddings = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=np.float32)
            else:
                embeddings = np.zeros((len(texts), 2), dtype=np.float32)
            return EmbeddingResult("fake/model", embeddings)

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("sanskrit_pipeline.pairwise.segment_text_to_sentences", side_effect=[sentences_a, sentences_b]):
                with patch("sanskrit_pipeline.pairwise.TextEmbedder.encode_queries", side_effect=fake_encode):
                    with patch("sanskrit_pipeline.pairwise.TextEmbedder.encode_corpus", side_effect=fake_encode):
                        artifacts = run_pairwise_similarity(
                            text_a="unused-a",
                            text_b="unused-b",
                            output_dir=temp_dir,
                            model_id="fake/model",
                            top_k=4,
                        )

            self.assertTrue(artifacts.topk_csv.exists())
            self.assertTrue(artifacts.topk_jsonl.exists())
            self.assertTrue(artifacts.manifest_json.exists())

            with artifacts.topk_csv.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0]["i"], "0")
            self.assertEqual(rows[0]["j"], "0")
            self.assertEqual(rows[0]["sentence_a"], "धर्मो रक्षति॥")
            self.assertEqual(rows[0]["sentence_b"], "धर्मो रक्षति॥")

            with artifacts.topk_jsonl.open(encoding="utf-8") as handle:
                jsonl_rows = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(len(jsonl_rows), 4)

            manifest = json.loads(Path(artifacts.manifest_json).read_text(encoding="utf-8"))
            self.assertEqual(manifest["model_id"], "fake/model")
            self.assertEqual(manifest["device"], "auto")
            self.assertEqual(manifest["segment_count_a"], 2)
            self.assertEqual(manifest["segment_count_b"], 3)
            self.assertEqual(manifest["top_k_requested"], 4)
            self.assertEqual(manifest["top_k_returned"], 4)
            self.assertIn("max_score", manifest)
            self.assertIn("mean_best_a_to_b", manifest)

    def test_run_pairwise_similarity_passes_device_to_embedder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("sanskrit_pipeline.pairwise.segment_text_to_sentences", side_effect=[["a"], ["b"]]):
                with patch("sanskrit_pipeline.pairwise.TextEmbedder") as mock_embedder_cls:
                    mock_embedder = mock_embedder_cls.return_value
                    mock_embedder.encode_queries.return_value = EmbeddingResult("fake/model", np.array([[1.0, 0.0]], dtype=np.float32))
                    mock_embedder.encode_corpus.return_value = EmbeddingResult("fake/model", np.array([[1.0, 0.0]], dtype=np.float32))
                    run_pairwise_similarity(
                        text_a="unused-a",
                        text_b="unused-b",
                        output_dir=temp_dir,
                        model_id="fake/model",
                        device="cpu",
                        top_k=1,
                    )

            mock_embedder_cls.assert_called_once_with(
                model_id="fake/model",
                batch_size=8,
                normalize_embeddings=True,
                device="cpu",
                embedding_progress="off",
                torch_dtype=None,
                device_map=None,
                load_in_8bit=False,
                low_cpu_mem_usage=None,
            )


if __name__ == "__main__":
    unittest.main()
