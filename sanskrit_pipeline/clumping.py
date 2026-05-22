"""Utilities for constructing longer passage clumps from sentence rows."""

from __future__ import annotations

import json
from dataclasses import dataclass

from .io import InputRecord


@dataclass(slots=True)
class ClumpedRecord(InputRecord):
    """A synthetic passage assembled from neighboring sentence rows."""

    source_sentences: list[str]


def build_clumped_records(
    records: list[InputRecord],
    clump_size: int = 5,
    stride: int | None = None,
    joiner: str = " ",
) -> list[ClumpedRecord]:
    """Concatenate neighboring sentence rows into larger passage records."""
    if clump_size <= 0:
        raise ValueError("clump_size must be positive")

    stride = stride or clump_size
    if stride <= 0:
        raise ValueError("stride must be positive")

    clumps: list[ClumpedRecord] = []
    for start in range(0, len(records), stride):
        window = records[start : start + clump_size]
        if not window:
            continue
        source_sentences = [record.text.strip() for record in window if record.text.strip()]
        if not source_sentences:
            continue
        clumps.append(
            ClumpedRecord(
                record_id=f"clump-{start:05d}",
                text=joiner.join(source_sentences).strip(),
                source_sentences=source_sentences,
            )
        )
    return clumps


def source_sentences_to_json(sentences: list[str]) -> str:
    """Serialize source sentence rows for review artifacts."""
    return json.dumps(sentences, ensure_ascii=False)
