"""Tests for bidirectional corpus pairwise workflows."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.test_corpus_pairwise import FakeSDK
from sanskrit_pipeline.corpus_bidirectional import run_bidirectional_corpus_pairwise


class BidirectionalCorpusPairwiseTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeSDK.instances.clear()

    def test_bidirectional_workflow_runs_both_directions_and_generates_three_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dir_a = root / "corpus_a"
            dir_b = root / "corpus_b"
            out = root / "bidirectional"
            dir_a.mkdir()
            dir_b.mkdir()
            (dir_a / "text_a1.txt").write_text("alpha|shared", encoding="utf-8")
            (dir_b / "text_b1.txt").write_text("shared|gamma", encoding="utf-8")

            with patch("sanskrit_pipeline.corpus_pairwise.SanskritResearchSDK", FakeSDK):
                artifacts = run_bidirectional_corpus_pairwise(
                    dir_a=dir_a,
                    dir_b=dir_b,
                    output_dir=out,
                    label_a="Corpus A",
                    label_b="Corpus B",
                    model_id="fake/model",
                    device="cpu",
                    top_k=2,
                    generate_reports=True,
                    report_heatmap_size=4,
                    report_max_topk=2,
                )

            self.assertTrue(artifacts.forward["manifest_json"].exists())
            self.assertTrue(artifacts.reverse["manifest_json"].exists())
            self.assertTrue(artifacts.forward_report_html.exists())
            self.assertTrue(artifacts.reverse_report_html.exists())
            self.assertTrue(artifacts.synthesis_report_html.exists())
            self.assertTrue(artifacts.synthesis_csv.exists())
            self.assertTrue(artifacts.manifest_json.exists())
            self.assertIn("Corpus A surroundings", artifacts.forward_report_html.read_text(encoding="utf-8"))
            self.assertIn("click for ±3 context", artifacts.forward_report_html.read_text(encoding="utf-8"))

            manifest = json.loads(artifacts.manifest_json.read_text(encoding="utf-8"))
            self.assertEqual(manifest["label_a"], "Corpus A")
            self.assertEqual(manifest["label_b"], "Corpus B")
            self.assertEqual(manifest["forward_run_dir"], str(out / "forward"))
            self.assertEqual(manifest["reverse_run_dir"], str(out / "reverse"))

            forward_manifest = json.loads(artifacts.forward["manifest_json"].read_text(encoding="utf-8"))
            reverse_manifest = json.loads(artifacts.reverse["manifest_json"].read_text(encoding="utf-8"))
            self.assertEqual(forward_manifest["dir_a"], str(dir_a))
            self.assertEqual(forward_manifest["dir_b"], str(dir_b))
            self.assertEqual(reverse_manifest["dir_a"], str(dir_b))
            self.assertEqual(reverse_manifest["dir_b"], str(dir_a))

            self.assertEqual(len(FakeSDK.instances), 2)
            self.assertEqual([is_query for _, is_query in FakeSDK.instances[0].embed_calls], [True, False])
            self.assertEqual([is_query for _, is_query in FakeSDK.instances[1].embed_calls], [True, False])


if __name__ == "__main__":
    unittest.main()
