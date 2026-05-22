"""Qualitative review artifact generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .pipeline import PipelineResult


def write_review_artifact(results: list[PipelineResult], output_path: str | Path) -> Path:
    """Write a human-reviewable CSV artifact for segmentation output."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "record_id",
        "engine",
        "source_format",
        "original_text",
        "normalized_text",
        "segment_count",
        "segments_json",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "record_id": result.record_id,
                    "engine": result.engine_name,
                    "source_format": result.source_format,
                    "original_text": result.original_text,
                    "normalized_text": result.normalized_text,
                    "segment_count": len(result.segments),
                    "segments_json": json.dumps(result.segments, ensure_ascii=False),
                }
            )

    return output_path
