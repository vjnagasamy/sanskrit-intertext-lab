"""Jupyter-friendly SDK surface for modular Sanskrit pipeline experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .embeddings import DEFAULT_MODEL_ID, TextEmbedder, TorchDTypeName
from .normalization import normalize_text
from .pairwise import PairMatch
from .pairwise_run import PairwiseMetrics, PairwiseSegment, make_segments, run_pairwise_similarity_core
from .pipeline import resolve_segmenter


@dataclass(slots=True)
class SegmentationView:
    """Notebook-friendly segmentation view for one source text."""

    original_text: str
    normalized_text: str
    source_format: str
    engine_name: str
    segments: list[str]
    spans: list[tuple[int, int]]

    def to_dataframe(self) -> pd.DataFrame:
        rows = [
            {
                "segment_index": index,
                "start": start,
                "end": end,
                "segment_text": text,
            }
            for index, (text, (start, end)) in enumerate(zip(self.segments, self.spans))
        ]
        return pd.DataFrame(rows)


@dataclass(slots=True)
class EmbeddingView:
    """Notebook-friendly embedding view for sentence lists."""

    model_id: str
    device: str
    sentences: list[str]
    embeddings: np.ndarray

    def to_dataframe(self, include_vectors: bool = False) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for index, sentence in enumerate(self.sentences):
            row: dict[str, object] = {
                "sentence_index": index,
                "sentence_text": sentence,
            }
            if self.embeddings.size:
                row["vector_norm"] = float(np.linalg.norm(self.embeddings[index]))
                if include_vectors:
                    row["embedding"] = self.embeddings[index].tolist()
            rows.append(row)
        return pd.DataFrame(rows)


@dataclass(slots=True)
class PairwiseView:
    """Notebook-friendly pairwise similarity outputs."""

    model_id: str
    device: str
    segments_a: list[str]
    segments_b: list[str]
    segment_records_a: list[PairwiseSegment]
    segment_records_b: list[PairwiseSegment]
    similarity_matrix: np.ndarray
    matches: list[PairMatch]
    metrics: PairwiseMetrics

    def topk_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(match) for match in self.matches])


class SanskritResearchSDK:
    """High-level SDK for segmentation, embeddings, and pairwise notebook workflows."""

    def __init__(
        self,
        *,
        engine: str = "dandas",
        source_format: str = "devanagari",
        split_on_single_danda: bool = False,
        model_id: str = DEFAULT_MODEL_ID,
        batch_size: int = 8,
        device: Literal["auto", "cpu", "mps", "cuda"] = "auto",
        embedding_progress: Literal["off", "batch", "sentence"] = "off",
        torch_dtype: TorchDTypeName | None = None,
        device_map: str | dict[str, int | str] | None = None,
        load_in_8bit: bool = False,
        low_cpu_mem_usage: bool | None = None,
    ) -> None:
        self.engine = engine
        self.source_format = source_format
        self.split_on_single_danda = split_on_single_danda
        self.model_id = model_id
        self.batch_size = batch_size
        self.device = device
        self.embedding_progress = embedding_progress
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.load_in_8bit = load_in_8bit
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self._segmenter = resolve_segmenter(
            engine=engine,
            split_on_single_danda=split_on_single_danda,
        )
        self._embedders: dict[tuple, TextEmbedder] = {}

    def segment_text(self, text: str, source_format: str | None = None) -> SegmentationView:
        """Normalize and segment one source text."""
        source_format = source_format or self.source_format
        normalized = normalize_text(
            text,
            source_format=source_format,
            preserve_lines=self._segmenter.requires_line_structure,
            transliterate=not self._segmenter.keep_source_script,
        )
        segmented = self._segmenter.segment(normalized)
        segments = [segment.text for segment in segmented]
        spans = [(segment.start, segment.end) for segment in segmented]
        return SegmentationView(
            original_text=text,
            normalized_text=normalized,
            source_format=source_format,
            engine_name=self._segmenter.engine_name,
            segments=segments,
            spans=spans,
        )

    def embed_sentences(
        self,
        sentences: list[str],
        *,
        model_id: str | None = None,
        batch_size: int | None = None,
        device: Literal["auto", "cpu", "mps", "cuda"] | None = None,
        embedding_progress: Literal["off", "batch", "sentence"] | None = None,
        torch_dtype: TorchDTypeName | None = None,
        device_map: str | dict[str, int | str] | None = None,
        load_in_8bit: bool | None = None,
        low_cpu_mem_usage: bool | None = None,
        is_query: bool = False,
    ) -> EmbeddingView:
        """Embed a list of sentences using the configured model."""
        model_id = model_id or self.model_id
        batch_size = batch_size or self.batch_size
        device = device or self.device
        embedding_progress = embedding_progress or self.embedding_progress
        load_in_8bit = load_in_8bit if load_in_8bit is not None else self.load_in_8bit
        torch_dtype = torch_dtype if torch_dtype is not None else self.torch_dtype
        device_map = device_map if device_map is not None else self.device_map
        low_cpu_mem_usage = (
            low_cpu_mem_usage if low_cpu_mem_usage is not None else self.low_cpu_mem_usage
        )

        embedder = self._get_embedder(
            model_id=model_id,
            batch_size=batch_size,
            device=device,
            embedding_progress=embedding_progress,
            torch_dtype=torch_dtype,
            device_map=device_map,
            load_in_8bit=load_in_8bit,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
        if is_query:
            result = embedder.encode_queries(sentences)
        else:
            result = embedder.encode_corpus(sentences)

        return EmbeddingView(
            model_id=result.model_id,
            device=embedder._device,
            sentences=sentences,
            embeddings=result.embeddings,
        )

    def pairwise(
        self,
        text_a: str,
        text_b: str,
        *,
        top_k: int = 100,
        model_id: str | None = None,
        batch_size: int | None = None,
        device: Literal["auto", "cpu", "mps", "cuda"] | None = None,
        embedding_progress: Literal["off", "batch", "sentence"] | None = None,
        torch_dtype: TorchDTypeName | None = None,
        device_map: str | dict[str, int | str] | None = None,
        load_in_8bit: bool | None = None,
        low_cpu_mem_usage: bool | None = None,
    ) -> PairwiseView:
        sentences_a = self._segment_text_to_sentences(text_a)
        sentences_b = self._segment_text_to_sentences(text_b)
        return self.pairwise_from_sentences(
            sentences_a,
            sentences_b,
            top_k=top_k,
            model_id=model_id,
            batch_size=batch_size,
            device=device,
            embedding_progress=embedding_progress,
            torch_dtype=torch_dtype,
            device_map=device_map,
            load_in_8bit=load_in_8bit,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )

    def pairwise_from_sentences(
        self,
        sentences_a: list[str],
        sentences_b: list[str],
        *,
        top_k: int = 100,
        model_id: str | None = None,
        batch_size: int | None = None,
        device: Literal["auto", "cpu", "mps", "cuda"] | None = None,
        embedding_progress: Literal["off", "batch", "sentence"] | None = None,
        torch_dtype: TorchDTypeName | None = None,
        device_map: str | dict[str, int | str] | None = None,
        load_in_8bit: bool | None = None,
        low_cpu_mem_usage: bool | None = None,
    ) -> PairwiseView:
        embedding_a = self.embed_sentences(
            sentences_a,
            model_id=model_id,
            batch_size=batch_size,
            device=device,
            embedding_progress=embedding_progress,
            torch_dtype=torch_dtype,
            device_map=device_map,
            load_in_8bit=load_in_8bit,
            low_cpu_mem_usage=low_cpu_mem_usage,
            is_query=True,
        )
        embedding_b = self.embed_sentences(
            sentences_b,
            model_id=model_id,
            batch_size=batch_size,
            device=device,
            embedding_progress=embedding_progress,
            torch_dtype=torch_dtype,
            device_map=device_map,
            load_in_8bit=load_in_8bit,
            low_cpu_mem_usage=low_cpu_mem_usage,
            is_query=False,
        )
        result = run_pairwise_similarity_core(
            make_segments(sentences_a),
            embedding_a.embeddings,
            make_segments(sentences_b),
            embedding_b.embeddings,
            top_k=top_k,
        )
        return PairwiseView(
            model_id=embedding_a.model_id,
            device=embedding_a.device,
            segments_a=sentences_a,
            segments_b=sentences_b,
            segment_records_a=result.segments_a,
            segment_records_b=result.segments_b,
            similarity_matrix=result.similarity_matrix,
            matches=_compatibility_matches(result),
            metrics=result.metrics,
        )

    def pairwise_from_embedding_views(
        self,
        embedding_a: EmbeddingView,
        embedding_b: EmbeddingView,
        *,
        top_k: int = 100,
    ) -> PairwiseView:
        if embedding_a.model_id != embedding_b.model_id:
            raise ValueError("Embedding views must use the same model_id.")
        if embedding_a.embeddings.ndim != 2 or embedding_b.embeddings.ndim != 2:
            raise ValueError("Embedding views must contain rank-2 embedding arrays.")
        if embedding_a.embeddings.shape[0] != len(embedding_a.sentences):
            raise ValueError("Embedding view A row count must match its sentence count.")
        if embedding_b.embeddings.shape[0] != len(embedding_b.sentences):
            raise ValueError("Embedding view B row count must match its sentence count.")

        result = run_pairwise_similarity_core(
            make_segments(embedding_a.sentences),
            embedding_a.embeddings,
            make_segments(embedding_b.sentences),
            embedding_b.embeddings,
            top_k=top_k,
        )
        return PairwiseView(
            model_id=embedding_a.model_id,
            device=embedding_a.device,
            segments_a=embedding_a.sentences,
            segments_b=embedding_b.sentences,
            segment_records_a=result.segments_a,
            segment_records_b=result.segments_b,
            similarity_matrix=result.similarity_matrix,
            matches=_compatibility_matches(result),
            metrics=result.metrics,
        )

    def bidirectional_corpus_pairwise(
        self,
        dir_a: str | Path,
        dir_b: str | Path,
        output_dir: str | Path,
        **kwargs: object,
    ) -> object:
        """Run corpus pairwise analysis both directions using this SDK's defaults."""
        from .corpus_bidirectional import run_bidirectional_corpus_pairwise

        defaults = {
            "engine": self.engine,
            "source_format": self.source_format,
            "split_on_single_danda": self.split_on_single_danda,
            "model_id": self.model_id,
            "batch_size": self.batch_size,
            "device": self.device,
            "embedding_progress": self.embedding_progress,
            "torch_dtype": self.torch_dtype,
            "device_map": self.device_map,
            "load_in_8bit": self.load_in_8bit,
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
        }
        defaults.update(kwargs)
        return run_bidirectional_corpus_pairwise(dir_a, dir_b, output_dir, **defaults)

    def _segment_text_to_sentences(self, text: str) -> list[str]:
        view = self.segment_text(text)
        return [segment for segment in view.segments if segment.strip()]

    def _get_embedder(
        self,
        *,
        model_id: str,
        batch_size: int,
        device: Literal["auto", "cpu", "mps", "cuda"],
        embedding_progress: Literal["off", "batch", "sentence"],
        torch_dtype: TorchDTypeName | None,
        device_map: str | dict[str, int | str] | None,
        load_in_8bit: bool,
        low_cpu_mem_usage: bool | None,
    ) -> TextEmbedder:
        cache_key = (
            model_id,
            device,
            torch_dtype,
            repr(device_map),
            load_in_8bit,
            low_cpu_mem_usage,
        )
        embedder = self._embedders.get(cache_key)
        if embedder is None:
            embedder = TextEmbedder(
                model_id=model_id,
                batch_size=batch_size,
                normalize_embeddings=True,
                device=device,
                embedding_progress=embedding_progress,
                torch_dtype=torch_dtype,
                device_map=device_map,
                load_in_8bit=load_in_8bit,
                low_cpu_mem_usage=low_cpu_mem_usage,
            )
            self._embedders[cache_key] = embedder
        else:
            embedder.batch_size = batch_size
            embedder.embedding_progress = embedding_progress
        return embedder


def _compatibility_matches(result: object) -> list[PairMatch]:
    return [
        PairMatch(
            rank=match.rank,
            score=match.score,
            i=match.segment_a.index,
            j=match.segment_b.index,
            sentence_a=match.segment_a.text,
            sentence_b=match.segment_b.text,
        )
        for match in result.matches
    ]
