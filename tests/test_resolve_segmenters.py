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
        mock_cls.assert_called_once_with(split_on_single_danda=False)

    def test_dandas_engine_passes_split_flag(self) -> None:
        with patch("sanskrit_pipeline.pipeline.DandasSegmenter", return_value="dandas_pada") as mock_cls:
            resolved = resolve_segmenter("dandas", split_on_single_danda=True)
        self.assertEqual(resolved, "dandas_pada")
        mock_cls.assert_called_once_with(split_on_single_danda=True)

    def test_stanza_engine_raises_when_not_installed(self) -> None:
        with patch("sanskrit_pipeline.segmenters.stanza_segmenter._STANZA_AVAILABLE", False):
            with self.assertRaises(ImportError):
                resolve_segmenter("stanza")

    def test_unknown_engine_raises_value_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            resolve_segmenter("botok")
        self.assertIn("botok", str(ctx.exception))
        self.assertIn("dandas", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
