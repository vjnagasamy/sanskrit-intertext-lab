"""Segmenter contract tests for the Sanskrit DandasSegmenter."""

from __future__ import annotations

import unittest

from sanskrit_pipeline.segmenters.dandas import DandasSegmenter
from sanskrit_pipeline.segmenters.lines import LineSegmenter


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


class LineSegmenterTests(unittest.TestCase):
    def test_engine_name_and_line_structure_flag(self) -> None:
        segmenter = LineSegmenter()
        self.assertEqual(segmenter.engine_name, "lines")
        self.assertTrue(segmenter.requires_line_structure)

    def test_one_segment_per_line(self) -> None:
        segmenter = LineSegmenter()
        text = "alpha\nbeta\ngamma"
        segments = segmenter.segment(text)
        self.assertEqual([s.text for s in segments], ["alpha", "beta", "gamma"])

    def test_blank_and_comment_lines_dropped(self) -> None:
        segmenter = LineSegmenter()
        text = "%% header note\n\nalpha\n%comment\nbeta\n"
        segments = segmenter.segment(text)
        self.assertEqual([s.text for s in segments], ["alpha", "beta"])

    def test_strips_colon_verse_label_iast(self) -> None:
        segmenter = LineSegmenter()
        segments = segmenter.segment("1.1ab: athato rahasyam\n1.1cd: sriherukasamyogam")
        self.assertEqual(
            [s.text for s in segments],
            ["athato rahasyam", "sriherukasamyogam"],
        )

    def test_strips_colon_verse_label_transliterated(self) -> None:
        segmenter = LineSegmenter()
        segments = segmenter.segment("१.१अब्: अथतो रहस्यम्")
        self.assertEqual(segments[0].text, "अथतो रहस्यम्")

    def test_strips_space_only_numeric_label(self) -> None:
        segmenter = LineSegmenter()
        segments = segmenter.segment("1.2 srimaddhimavatah\nsantanapuramadhyagam")
        self.assertEqual(
            [s.text for s in segments],
            ["srimaddhimavatah", "santanapuramadhyagam"],
        )

    def test_mid_line_colon_is_not_stripped(self) -> None:
        segmenter = LineSegmenter()
        segments = segmenter.segment("alpha beta: gamma")
        self.assertEqual(segments[0].text, "alpha beta: gamma")

    def test_spans_point_into_source_text(self) -> None:
        segmenter = LineSegmenter()
        text = "1.1ab: athato"
        segments = segmenter.segment(text)
        start, end = segments[0].start, segments[0].end
        self.assertEqual(text[start:end], "athato")

    def test_empty_input_returns_no_segments(self) -> None:
        self.assertEqual(LineSegmenter().segment("   \n  \n"), [])


if __name__ == "__main__":
    unittest.main()
