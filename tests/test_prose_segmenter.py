"""Unit tests for prose and hybrid Sanskrit segmenters."""

from __future__ import annotations

import unittest

from sanskrit_pipeline.segmenters.dandas import DandasSegmenter
from sanskrit_pipeline.segmenters.hybrid import HybridSegmenter
from sanskrit_pipeline.segmenters.prose import ProseSegmenter


def _long_devanagari_clause(repeats: int = 80) -> str:
    unit = "धर्मो रक्षति रक्षितः "
    return unit * repeats


class ProseSegmenterTests(unittest.TestCase):
    def test_long_verse_chunks_respect_max_tokens(self) -> None:
        segmenter = ProseSegmenter(max_tokens=50, overlap_tokens=5, chars_per_token=3.5)
        text = f"{_long_devanagari_clause()}॥"
        segments = segmenter.segment(text)
        self.assertGreater(len(segments), 1)
        for segment in segments:
            estimated_tokens = len(segment.text) / segmenter.chars_per_token
            self.assertLessEqual(estimated_tokens, segmenter.max_tokens + 5)

    def test_short_text_matches_dandas_output(self) -> None:
        text = "धर्मो रक्षति रक्षितः॥ सत्यमेव जयते॥"
        dandas = DandasSegmenter()
        prose = ProseSegmenter(max_tokens=400)
        dandas_segments = [segment.text for segment in dandas.segment(text)]
        prose_segments = [segment.text for segment in prose.segment(text)]
        self.assertEqual(prose_segments, dandas_segments)

    def test_overlap_preserves_shared_context(self) -> None:
        segmenter = ProseSegmenter(max_tokens=40, overlap_tokens=10, chars_per_token=3.5)
        text = f"{_long_devanagari_clause(40)}॥"
        segments = [segment.text for segment in segmenter.segment(text)]
        self.assertGreaterEqual(len(segments), 2)
        overlap_chars = int(segmenter.overlap_tokens * segmenter.chars_per_token)
        tail = segments[0][-overlap_chars:]
        self.assertIn(tail.strip()[: max(5, overlap_chars // 3)], segments[1])

    def test_hybrid_only_splits_long_chunks(self) -> None:
        short = "धर्मो रक्षति रक्षितः॥ सत्यमेव जयते॥"
        long_text = f"{_long_devanagari_clause(60)}॥ {short}"
        hybrid = HybridSegmenter(max_tokens=50, overlap_tokens=5)
        dandas = DandasSegmenter()
        hybrid_segments = [segment.text for segment in hybrid.segment(long_text)]
        dandas_segments = [segment.text for segment in dandas.segment(long_text)]
        self.assertGreater(len(hybrid_segments), len(dandas_segments))
        self.assertEqual(hybrid_segments[-1], dandas_segments[-1])


if __name__ == "__main__":
    unittest.main()
