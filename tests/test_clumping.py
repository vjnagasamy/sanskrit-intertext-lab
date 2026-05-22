"""Tests for clumped pseudo-evaluation utilities."""

from __future__ import annotations

import unittest

from sanskrit_pipeline.clumping import build_clumped_records
from sanskrit_pipeline.io import InputRecord
from sanskrit_pipeline.pipeline import PipelineResult
from sanskrit_pipeline.pseudo_eval import compare_clump_to_prediction


class ClumpingTests(unittest.TestCase):
    def test_build_clumped_records_groups_neighbors(self) -> None:
        records = [
            InputRecord(record_id="0", text="a"),
            InputRecord(record_id="1", text="b"),
            InputRecord(record_id="2", text="c"),
        ]
        clumps = build_clumped_records(records, clump_size=2, stride=2)

        self.assertEqual(len(clumps), 2)
        self.assertEqual(clumps[0].text, "a b")
        self.assertEqual(clumps[0].source_sentences, ["a", "b"])

    def test_compare_clump_to_prediction_scores_exact_overlap(self) -> None:
        clump = build_clumped_records(
            [
                InputRecord(record_id="0", text="s1"),
                InputRecord(record_id="1", text="s2"),
                InputRecord(record_id="2", text="s3"),
            ],
            clump_size=3,
        )[0]
        result = PipelineResult(
            record_id=clump.record_id,
            original_text=clump.text,
            normalized_text=clump.text,
            segments=["s1", "s2", "extra"],
            segment_spans=[(0, 2), (3, 4), (5, 8)],
            source_format="devanagari",
            engine_name="dandas",
        )
        row = compare_clump_to_prediction(clump, result)

        self.assertEqual(row.exact_match_count, 2)
        self.assertAlmostEqual(row.source_recall, 2 / 3)
        self.assertAlmostEqual(row.predicted_precision, 2 / 3)
        self.assertEqual(row.boundary_true_positive, 1)
        self.assertEqual(row.boundary_false_positive, 1)
        self.assertEqual(row.boundary_false_negative, 1)
        self.assertAlmostEqual(row.boundary_precision, 0.5)
        self.assertAlmostEqual(row.boundary_recall, 0.5)
        self.assertAlmostEqual(row.boundary_f1, 0.5)


if __name__ == "__main__":
    unittest.main()
