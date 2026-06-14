"""Tests for segmenter resolver engine mapping."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from sanskrit_pipeline.pipeline import resolve_segmenter


class ResolveSegmentersTests(unittest.TestCase):
    def test_dandas_engine_returns_dandas_segmenter(self) -> None:
        with patch("sanskrit_pipeline.pipeline.DandasSegmenter", return_value="dandas_instance") as mock_cls:
            resolved = resolve_segmenter("dandas")
        self.assertEqual(resolved, "dandas_instance")
        mock_cls.assert_called_once_with(split_on_single_danda=False, strip_dandas=False)

    def test_dandas_engine_passes_split_flag(self) -> None:
        with patch("sanskrit_pipeline.pipeline.DandasSegmenter", return_value="dandas_pada") as mock_cls:
            resolved = resolve_segmenter("dandas", split_on_single_danda=True)
        self.assertEqual(resolved, "dandas_pada")
        mock_cls.assert_called_once_with(split_on_single_danda=True, strip_dandas=False)

    def test_dandas_engine_passes_strip_flag(self) -> None:
        with patch("sanskrit_pipeline.pipeline.DandasSegmenter", return_value="dandas_stripped") as mock_cls:
            resolved = resolve_segmenter("dandas", strip_dandas=True)
        self.assertEqual(resolved, "dandas_stripped")
        mock_cls.assert_called_once_with(split_on_single_danda=False, strip_dandas=True)

    def test_lines_engine_returns_line_segmenter(self) -> None:
        from sanskrit_pipeline.segmenters.lines import LineSegmenter

        resolved = resolve_segmenter("lines")
        self.assertIsInstance(resolved, LineSegmenter)
        self.assertTrue(resolved.requires_line_structure)

    def test_stanza_engine_raises_when_not_installed(self) -> None:
        with patch("sanskrit_pipeline.segmenters.stanza_segmenter._STANZA_AVAILABLE", False):
            with self.assertRaises(ImportError):
                resolve_segmenter("stanza")

    def test_stanza_engine_uses_devanagari_default(self) -> None:
        with patch("sanskrit_pipeline.pipeline.StanzaSegmenter", return_value="stanza_instance") as mock_cls:
            resolved = resolve_segmenter("stanza")
        self.assertEqual(resolved, "stanza_instance")
        mock_cls.assert_called_once_with()

    def test_stanza_iast_engine_keeps_source_script(self) -> None:
        with patch("sanskrit_pipeline.pipeline.StanzaSegmenter", return_value="stanza_iast_instance") as mock_cls:
            resolved = resolve_segmenter("stanza_iast")
        self.assertEqual(resolved, "stanza_iast_instance")
        mock_cls.assert_called_once_with(keep_source_script=True)

    def test_unknown_engine_raises_value_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            resolve_segmenter("botok")
        self.assertIn("botok", str(ctx.exception))
        self.assertIn("dandas", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
