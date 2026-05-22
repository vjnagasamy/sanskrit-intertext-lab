"""Pseudo-evaluation helpers for clumped segmentation review."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from .clumping import ClumpedRecord
from .pipeline import PipelineResult


@dataclass(slots=True)
class PseudoEvalRow:
    """Comparison between upstream sentence rows and predicted segmentation."""

    record_id: str
    engine: str
    source_sentence_count: int
    predicted_sentence_count: int
    exact_match_count: int
    source_recall: float
    predicted_precision: float
    boundary_true_positive: int
    boundary_false_positive: int
    boundary_false_negative: int
    boundary_precision: float
    boundary_recall: float
    boundary_f1: float
    clump_text: str
    source_sentences: list[str]
    predicted_sentences: list[str]


def compare_clump_to_prediction(clump: ClumpedRecord, result: PipelineResult) -> PseudoEvalRow:
    """Compute a lightweight overlap score against upstream sentence rows."""
    source = [_normalize_segment(text) for text in clump.source_sentences if _normalize_segment(text)]
    predicted = [_normalize_segment(text) for text in result.segments if _normalize_segment(text)]
    source_set = set(source)
    predicted_set = set(predicted)
    exact_matches = sorted(source_set & predicted_set)

    source_recall = len(exact_matches) / len(source_set) if source_set else 0.0
    predicted_precision = len(exact_matches) / len(predicted_set) if predicted_set else 0.0
    source_boundaries = _source_boundary_positions(clump.source_sentences)
    predicted_boundaries = _predicted_boundary_positions(result.segment_spans, result.normalized_text)
    boundary_tp = len(source_boundaries & predicted_boundaries)
    boundary_fp = len(predicted_boundaries - source_boundaries)
    boundary_fn = len(source_boundaries - predicted_boundaries)
    boundary_precision = boundary_tp / len(predicted_boundaries) if predicted_boundaries else 0.0
    boundary_recall = boundary_tp / len(source_boundaries) if source_boundaries else 0.0
    if boundary_precision + boundary_recall == 0:
        boundary_f1 = 0.0
    else:
        boundary_f1 = (2 * boundary_precision * boundary_recall) / (boundary_precision + boundary_recall)

    return PseudoEvalRow(
        record_id=result.record_id,
        engine=result.engine_name,
        source_sentence_count=len(source),
        predicted_sentence_count=len(predicted),
        exact_match_count=len(exact_matches),
        source_recall=source_recall,
        predicted_precision=predicted_precision,
        boundary_true_positive=boundary_tp,
        boundary_false_positive=boundary_fp,
        boundary_false_negative=boundary_fn,
        boundary_precision=boundary_precision,
        boundary_recall=boundary_recall,
        boundary_f1=boundary_f1,
        clump_text=result.normalized_text,
        source_sentences=source,
        predicted_sentences=predicted,
    )


def write_pseudo_eval_csv(rows: list[PseudoEvalRow], output_path: str | Path) -> Path:
    """Write clump-level pseudo-evaluation rows to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "record_id",
        "engine",
        "source_sentence_count",
        "predicted_sentence_count",
        "exact_match_count",
        "source_recall",
        "predicted_precision",
        "boundary_true_positive",
        "boundary_false_positive",
        "boundary_false_negative",
        "boundary_precision",
        "boundary_recall",
        "boundary_f1",
        "clump_text",
        "source_sentences_json",
        "predicted_sentences_json",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "record_id": row.record_id,
                    "engine": row.engine,
                    "source_sentence_count": row.source_sentence_count,
                    "predicted_sentence_count": row.predicted_sentence_count,
                    "exact_match_count": row.exact_match_count,
                    "source_recall": f"{row.source_recall:.4f}",
                    "predicted_precision": f"{row.predicted_precision:.4f}",
                    "boundary_true_positive": row.boundary_true_positive,
                    "boundary_false_positive": row.boundary_false_positive,
                    "boundary_false_negative": row.boundary_false_negative,
                    "boundary_precision": f"{row.boundary_precision:.4f}",
                    "boundary_recall": f"{row.boundary_recall:.4f}",
                    "boundary_f1": f"{row.boundary_f1:.4f}",
                    "clump_text": row.clump_text,
                    "source_sentences_json": json.dumps(row.source_sentences, ensure_ascii=False),
                    "predicted_sentences_json": json.dumps(row.predicted_sentences, ensure_ascii=False),
                }
            )
    return output_path


def summarize_pseudo_eval(rows: list[PseudoEvalRow]) -> dict[str, float]:
    """Aggregate high-level pseudo-evaluation metrics across clumps."""
    if not rows:
        return {
            "clump_count": 0.0,
            "mean_source_recall": 0.0,
            "mean_predicted_precision": 0.0,
            "mean_boundary_precision": 0.0,
            "mean_boundary_recall": 0.0,
            "mean_boundary_f1": 0.0,
        }
    return {
        "clump_count": float(len(rows)),
        "mean_source_recall": sum(row.source_recall for row in rows) / len(rows),
        "mean_predicted_precision": sum(row.predicted_precision for row in rows) / len(rows),
        "mean_boundary_precision": sum(row.boundary_precision for row in rows) / len(rows),
        "mean_boundary_recall": sum(row.boundary_recall for row in rows) / len(rows),
        "mean_boundary_f1": sum(row.boundary_f1 for row in rows) / len(rows),
    }


def _normalize_segment(text: str) -> str:
    return " ".join(text.strip().split())


def _source_boundary_positions(source_sentences: list[str]) -> set[int]:
    """Boundary positions after each source sentence, excluding final document end."""
    boundaries: set[int] = set()
    cursor = 0
    cleaned = [_normalize_segment(text) for text in source_sentences if _normalize_segment(text)]
    for index, sentence in enumerate(cleaned):
        cursor += len(sentence)
        is_last = index == len(cleaned) - 1
        if not is_last:
            boundaries.add(cursor)
            cursor += 1  # clump joiner is a single space
    return boundaries


def _predicted_boundary_positions(segment_spans: list[tuple[int, int]], full_text: str) -> set[int]:
    """Boundary positions after each predicted segment, excluding final document end."""
    boundaries: set[int] = set()
    for start, end in segment_spans:
        boundary = end
        while boundary > start and full_text[boundary - 1].isspace():
            boundary -= 1
        if boundary < len(full_text):
            boundaries.add(boundary)
    return boundaries
