"""I/O helpers for loading Sanskrit text sources."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class InputRecord:
    """A raw input text record loaded from disk."""

    record_id: str
    text: str


def load_records(
    path: str | Path,
    text_column: str = "input_text",
    limit: int | None = None,
) -> list[InputRecord]:
    """Load input records from CSV, TSV, JSONL, or TXT."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".csv":
        records = _load_delimited_records(path, ",", text_column)
    elif suffix == ".tsv":
        records = _load_delimited_records(path, "\t", text_column)
    elif suffix == ".jsonl":
        records = _load_jsonl_records(path, text_column)
    elif suffix == ".txt":
        records = _load_text_records(path)
    else:
        raise ValueError(f"Unsupported input format for {path}")

    if limit is not None:
        return records[:limit]
    return records


def _load_delimited_records(path: Path, delimiter: str, text_column: str) -> list[InputRecord]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames:
            return []
        if text_column not in reader.fieldnames:
            text_column = reader.fieldnames[0]

        records: list[InputRecord] = []
        for index, row in enumerate(reader):
            text = (row.get(text_column) or "").strip()
            if not text:
                continue
            record_id = str(row.get("id") or row.get("record_id") or index)
            records.append(InputRecord(record_id=record_id, text=text))
        return records


def _load_jsonl_records(path: Path, text_column: str) -> list[InputRecord]:
    records: list[InputRecord] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = str(item.get(text_column) or item.get("text") or "").strip()
            if not text:
                continue
            record_id = str(item.get("id") or item.get("record_id") or index)
            records.append(InputRecord(record_id=record_id, text=text))
    return records


def _load_text_records(path: Path) -> list[InputRecord]:
    records: list[InputRecord] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            text = line.strip()
            if not text:
                continue
            records.append(InputRecord(record_id=str(index), text=text))
    return records
