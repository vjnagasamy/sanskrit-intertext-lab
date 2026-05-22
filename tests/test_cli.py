"""CLI tests for the Sanskrit pipeline."""

from __future__ import annotations

import csv
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import run_bidirectional_corpus_pairwise, run_corpus_pairwise_similarity
from sanskrit_pipeline.cli import run
from sanskrit_pipeline.embeddings import EmbeddingResult
from sanskrit_pipeline.pipeline import PipelineArtifacts
from sanskrit_pipeline.segmenters.base import BaseSegmenter, Segment


class FakeSegmenter(BaseSegmenter):
    engine_name = "dandas"

    def segment(self, text: str) -> list[Segment]:
        midpoint = max(1, len(text) // 2)
        return [
            Segment(text[:midpoint].strip(), 0, midpoint),
            Segment(text[midpoint:].strip(), midpoint, len(text)),
        ]


class CLITests(unittest.TestCase):
    def test_segmentation_only_writes_review_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.csv"
            input_path.write_text("input_text\nधर्मो रक्षति रक्षितः॥ सत्यमेव जयते॥\n", encoding="utf-8")
            args = Namespace(
                input=str(input_path),
                output_dir=str(Path(temp_dir) / "out"),
                input_format="devanagari",
                engine="dandas",
                text_column="input_text",
                limit=None,
                split_on_single_danda=False,
                embed=False,
                model_id="unused",
            )

            with patch("sanskrit_pipeline.cli.resolve_segmenter", return_value=FakeSegmenter()):
                artifacts = run(args)

            self.assertIsInstance(artifacts, PipelineArtifacts)
            self.assertTrue(artifacts.review_csv.exists())
            with artifacts.review_csv.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["engine"], "dandas")

    def test_embedding_stage_writes_numpy_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.csv"
            input_path.write_text("input_text\nधर्मो रक्षति रक्षितः॥ सत्यमेव जयते॥\n", encoding="utf-8")
            args = Namespace(
                input=str(input_path),
                output_dir=str(Path(temp_dir) / "out"),
                input_format="devanagari",
                engine="dandas",
                text_column="input_text",
                limit=None,
                split_on_single_danda=False,
                embed=True,
                model_id="fake/model",
            )

            with patch("sanskrit_pipeline.cli.resolve_segmenter", return_value=FakeSegmenter()):
                with patch("sanskrit_pipeline.pipeline.TextEmbedder.encode", return_value=EmbeddingResult("fake/model", np.ones((2, 3), dtype=np.float32))):
                    artifacts = run(args)

            self.assertTrue(artifacts.embeddings_npy.exists())
            self.assertTrue(artifacts.embeddings_metadata_json.exists())

    def test_corpus_pairwise_dry_run_reports_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dir_a = root / "a"
            dir_b = root / "b"
            dir_a.mkdir()
            dir_b.mkdir()
            (dir_a / "a1.txt").write_text("a1", encoding="utf-8")
            (dir_a / "a2.txt").write_text("a2", encoding="utf-8")
            (dir_b / "b1.txt").write_text("b1", encoding="utf-8")
            args = [
                "--dir-a",
                str(dir_a),
                "--dir-b",
                str(dir_b),
                "--output-dir",
                str(root / "out"),
                "--limit-a",
                "1",
                "--dry-run",
            ]

            with patch("builtins.print") as mock_print:
                exit_code = run_corpus_pairwise_similarity.main(args)

            self.assertEqual(exit_code, 0)
            printed = [call.args[0] for call in mock_print.call_args_list]
            self.assertIn("doc_count_a=1", printed)
            self.assertIn("doc_count_b=1", printed)
            self.assertIn("pair_count=1", printed)

    def test_bidirectional_corpus_pairwise_dry_run_reports_both_directions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dir_a = root / "a"
            dir_b = root / "b"
            dir_a.mkdir()
            dir_b.mkdir()
            (dir_a / "a1.txt").write_text("a1", encoding="utf-8")
            (dir_a / "a2.txt").write_text("a2", encoding="utf-8")
            (dir_b / "b1.txt").write_text("b1", encoding="utf-8")
            args = [
                "--dir-a",
                str(dir_a),
                "--dir-b",
                str(dir_b),
                "--output-dir",
                str(root / "out"),
                "--label-a",
                "Corpus A",
                "--label-b",
                "Corpus B",
                "--limit-a",
                "1",
                "--dry-run",
            ]

            with patch("builtins.print") as mock_print:
                exit_code = run_bidirectional_corpus_pairwise.main(args)

            self.assertEqual(exit_code, 0)
            printed = [call.args[0] for call in mock_print.call_args_list]
            self.assertIn("forward=Corpus A -> Corpus B", printed)
            self.assertIn("reverse=Corpus B -> Corpus A", printed)
            self.assertIn("doc_count_a=1", printed)
            self.assertIn("doc_count_b=1", printed)
            self.assertIn("pair_count_per_direction=1", printed)


if __name__ == "__main__":
    unittest.main()
