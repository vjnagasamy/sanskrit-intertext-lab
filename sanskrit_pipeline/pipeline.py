"""Pipeline orchestration for Sanskrit sentence segmentation and embeddings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embeddings import DEFAULT_MODEL_ID, TextEmbedder
from .io import InputRecord
from .normalization import normalize_text
from .segmenters import BaseSegmenter, DandasSegmenter, LineSegmenter, StanzaSegmenter


@dataclass(slots=True)
class PipelineResult:
    """Single-record segmentation output."""

    record_id: str
    original_text: str
    normalized_text: str
    segments: list[str]
    segment_spans: list[tuple[int, int]]
    source_format: str
    engine_name: str


@dataclass(slots=True)
class PipelineArtifacts:
    """Generated artifact paths."""

    review_csv: Path
    embeddings_npy: Path | None = None
    embeddings_metadata_json: Path | None = None


class SanskritPipeline:
    """High-level orchestration around normalization, segmentation, and embedding."""

    def __init__(self, segmenter: BaseSegmenter) -> None:
        self.segmenter = segmenter

    def run_segmentation(
        self,
        records: list[InputRecord],
        source_format: str = "devanagari",
    ) -> list[PipelineResult]:
        results: list[PipelineResult] = []
        preserve_lines = self.segmenter.requires_line_structure
        transliterate = not self.segmenter.keep_source_script
        for record in records:
            normalized = normalize_text(
                record.text,
                source_format=source_format,
                preserve_lines=preserve_lines,
                transliterate=transliterate,
            )
            segmented = self.segmenter.segment(normalized)
            segments = [segment.text for segment in segmented]
            segment_spans = [(segment.start, segment.end) for segment in segmented]
            results.append(
                PipelineResult(
                    record_id=record.record_id,
                    original_text=record.text,
                    normalized_text=normalized,
                    segments=segments,
                    segment_spans=segment_spans,
                    source_format=source_format,
                    engine_name=self.segmenter.engine_name,
                )
            )
        return results

    def write_embeddings(
        self,
        results: list[PipelineResult],
        output_dir: str | Path,
        model_id: str = DEFAULT_MODEL_ID,
    ) -> tuple[Path, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        sentences: list[str] = []
        metadata: list[dict[str, str | int]] = []
        for result in results:
            for segment_index, segment in enumerate(result.segments):
                sentences.append(segment)
                metadata.append(
                    {
                        "record_id": result.record_id,
                        "segment_index": segment_index,
                        "segment_text": segment,
                    }
                )

        embedder = TextEmbedder(model_id=model_id)
        encoded = embedder.encode(sentences)
        embeddings_path = output_dir / "embeddings.npy"
        metadata_path = output_dir / "embeddings_metadata.json"
        np.save(embeddings_path, encoded.embeddings)
        metadata_path.write_text(
            json.dumps(
                {
                    "model_id": encoded.model_id,
                    "rows": metadata,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return embeddings_path, metadata_path


def resolve_segmenter(
    engine: str,
    split_on_single_danda: bool = False,
    strip_dandas: bool = False,
) -> BaseSegmenter:
    """Resolve the requested segmenter backend.

    Args:
        engine: ``"dandas"`` (default, fast regex), ``"stanza"`` (ML-based, runs
            on transliterated Devanagari), ``"stanza_iast"`` (ML-based, runs on
            the original romanized IAST text), or ``"lines"`` (deterministic
            one-segment-per-line).
        split_on_single_danda: Passed to DandasSegmenter; ignored otherwise.
        strip_dandas: Passed to DandasSegmenter; ignored otherwise. Removes daṇḍa
            punctuation from emitted segments.

    Raises:
        ValueError: If the engine name is not recognized.
        ImportError: If a stanza engine is requested but stanza is not installed.
    """
    engine = engine.lower().strip()

    if engine == "dandas":
        return DandasSegmenter(
            split_on_single_danda=split_on_single_danda,
            strip_dandas=strip_dandas,
        )
    if engine == "stanza":
        return StanzaSegmenter()
    if engine == "stanza_iast":
        return StanzaSegmenter(keep_source_script=True)
    if engine == "lines":
        return LineSegmenter()

    raise ValueError(
        f"Unsupported engine: {engine!r}. "
        "Valid options are 'dandas', 'stanza', 'stanza_iast', and 'lines'."
    )
