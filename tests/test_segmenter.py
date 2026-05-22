"""Segmenter contract tests for the Sanskrit DandasSegmenter."""

from __future__ import annotations

import unittest

from sanskrit_pipeline.segmenters.dandas import DandasSegmenter


class DandasSegmenterTests(unittest.TestCase):
    def test_empty_text_returns_no_segments(self) -> None:
        segmenter = DandasSegmenter()
        self.assertEqual(segmenter.segment(""), [])

    def test_double_danda_creates_sentence_boundary(self) -> None:
        segmenter = DandasSegmenter()
        text = "धर्मो रक्षति रक्षितः॥ सत्यमेव जयते॥"
        segments = segmenter.segment(text)
        self.assertEqual(len(segments), 2)
        self.assertIn("धर्मो", segments[0].text)
        self.assertIn("सत्यमेव", segments[1].text)

    def test_single_danda_not_split_by_default(self) -> None:
        segmenter = DandasSegmenter(split_on_single_danda=False)
        text = "धर्मो रक्षति रक्षितः। सत्यमेव जयते॥"
        segments = segmenter.segment(text)
        self.assertEqual(len(segments), 1)

    def test_single_danda_split_when_flag_set(self) -> None:
        segmenter = DandasSegmenter(split_on_single_danda=True)
        text = "धर्मो रक्षति रक्षितः। सत्यमेव जयते॥"
        segments = segmenter.segment(text)
        self.assertEqual(len(segments), 2)

    def test_verse_numbers_are_filtered(self) -> None:
        segmenter = DandasSegmenter()
        text = "धर्मो रक्षति रक्षितः॥ १॥"
        segments = segmenter.segment(text)
        self.assertEqual(len(segments), 1)
        self.assertIn("धर्मो", segments[0].text)

    def test_non_devanagari_segments_are_filtered(self) -> None:
        segmenter = DandasSegmenter()
        text = "धर्मो रक्षति रक्षितः॥ some latin text॥"
        segments = segmenter.segment(text)
        self.assertEqual(len(segments), 1)
        self.assertIn("धर्मो", segments[0].text)

    def test_engine_name_is_dandas(self) -> None:
        segmenter = DandasSegmenter()
        self.assertEqual(segmenter.engine_name, "dandas")

    def test_whitespace_only_input_returns_no_segments(self) -> None:
        segmenter = DandasSegmenter()
        self.assertEqual(segmenter.segment("   \n\t  "), [])


if __name__ == "__main__":
    unittest.main()
