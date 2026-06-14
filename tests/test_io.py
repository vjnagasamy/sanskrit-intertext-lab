"""Tests for input record loading, especially .txt whole-file mode."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sanskrit_pipeline.io import load_records


class TxtLoadingTests(unittest.TestCase):
    def _write(self, temp_dir: str, name: str, text: str) -> Path:
        path = Path(temp_dir) / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_line_mode_is_one_record_per_line(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write(temp_dir, "verse.txt", "alpha\nbeta\n\ngamma\n")
            records = load_records(path)
        self.assertEqual([r.text for r in records], ["alpha", "beta", "gamma"])

    def test_whole_file_mode_is_single_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write(temp_dir, "verse.txt", "alpha\nbeta\ngamma\n")
            records = load_records(path, whole_file=True)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].text, "alpha\nbeta\ngamma")
        self.assertEqual(records[0].record_id, "verse")

    def test_whole_file_empty_returns_no_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write(temp_dir, "blank.txt", "   \n  \n")
            self.assertEqual(load_records(path, whole_file=True), [])


if __name__ == "__main__":
    unittest.main()
