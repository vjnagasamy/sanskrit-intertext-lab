"""Normalization and I/O tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sanskrit_pipeline.io import load_records
from sanskrit_pipeline.normalization import normalize_text

try:
    import indic_transliteration as _it
    _INDIC_AVAILABLE = True
except ImportError:
    _INDIC_AVAILABLE = False


class NormalizationTests(unittest.TestCase):
    def test_devanagari_passthrough_strips_whitespace(self) -> None:
        result = normalize_text("  धर्मो रक्षति  ")
        self.assertEqual(result, "धर्मो रक्षति")

    @unittest.skipUnless(_INDIC_AVAILABLE, "indic-transliteration not installed")
    def test_iast_to_devanagari(self) -> None:
        result = normalize_text("dharmo rakṣati", source_format="iast")
        self.assertIn("ध", result)

    def test_collapses_newlines_by_default(self) -> None:
        result = normalize_text("धर्मो\nरक्षति")
        self.assertEqual(result, "धर्मो रक्षति")

    def test_preserve_lines_keeps_newline_boundaries(self) -> None:
        result = normalize_text("  धर्मो  \n\n  रक्षति  ", preserve_lines=True)
        self.assertEqual(result, "धर्मो\nरक्षति")

    def test_transliterate_false_keeps_romanized_iast(self) -> None:
        result = normalize_text("dharmo rakṣati", source_format="iast", transliterate=False)
        self.assertEqual(result, "dharmo rakṣati")

    def test_transliterate_false_preserves_iast_lines(self) -> None:
        result = normalize_text(
            "1.1ab: athato\n1.1cd: sriheruka",
            source_format="iast",
            preserve_lines=True,
            transliterate=False,
        )
        self.assertEqual(result, "1.1ab: athato\n1.1cd: sriheruka")

    def test_unicode_format_raises_helpful_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            normalize_text("धर्मो", source_format="unicode")
        self.assertIn("'unicode'", str(ctx.exception))
        self.assertIn("Tibetan", str(ctx.exception))

    def test_wylie_format_raises_helpful_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            normalize_text("dharmo", source_format="wylie")
        self.assertIn("'wylie'", str(ctx.exception))
        self.assertIn("Tibetan", str(ctx.exception))

    def test_unknown_format_raises(self) -> None:
        with self.assertRaises(ValueError):
            normalize_text("धर्मो", source_format="latin")

    def test_load_records_uses_first_column_when_needed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "input.csv"
            path.write_text("text\nधर्मो रक्षति रक्षितः\n", encoding="utf-8")
            records = load_records(path)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].text, "धर्मो रक्षति रक्षितः")


if __name__ == "__main__":
    unittest.main()
