"""Corpus-level pairwise similarity workflows for Sanskrit text folders."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .embeddings import DEFAULT_MODEL_ID, TorchDTypeName
from .embedding_cache import EmbeddingCache
from .pairwise import PairMatch, write_topk_csv, write_topk_jsonl
from .pairwise_run import PairwiseSegment, TopKMode, make_segments, top_k_match_records
from .sdk import EmbeddingView, SanskritResearchSDK


@dataclass(slots=True)
class CorpusDocument:
    """One segmented and embedded source document."""

    doc_id: str
    relative_path: str
    sentence_count: int
    sentences_csv: Path
    embeddings_npy: Path
    segments: list[PairwiseSegment]
    embedding_view: EmbeddingView


@dataclass(slots=True)
class CorpusPairArtifacts:
    """Artifacts for one document-pair comparison."""

    pair_id: str
    doc_a_id: str
    doc_b_id: str
    pair_dir: Path
    sentences_a_csv: Path
    sentences_b_csv: Path
    topk_csv: Path
    topk_jsonl: Path
    similarity_npy: Path
    manifest_json: Path


@dataclass(slots=True)
class TopKExportArtifacts:
    """Artifacts regenerated from a saved pairwise matrix and sentence indexes."""

    topk_csv: Path
    topk_jsonl: Path
    k: int
    match_count: int
    mode: TopKMode


def run_corpus_pairwise_similarity(
    dir_a: str | Path,
    dir_b: str | Path,
    output_dir: str | Path,
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
    top_k: int = 100,
    glob_pattern: str = "*.txt",
    limit_a: int | None = None,
    limit_b: int | None = None,
    embedding_cache: EmbeddingCache | None = None,
) -> dict[str, Path]:
    """Run all cross-folder document-pair comparisons with reusable embeddings."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sdk = SanskritResearchSDK(
        engine=engine,
        source_format=source_format,
        split_on_single_danda=split_on_single_danda,
        model_id=model_id,
        batch_size=batch_size,
        device=device,
        embedding_progress=embedding_progress,
        torch_dtype=torch_dtype,
        device_map=device_map,
        load_in_8bit=load_in_8bit,
        low_cpu_mem_usage=low_cpu_mem_usage,
        cache=embedding_cache,
    )

    docs_a = _prepare_documents(
        root_dir=Path(dir_a),
        output_dir=output_dir / "documents_a",
        side_prefix="A",
        sdk=sdk,
        glob_pattern=glob_pattern,
        limit=limit_a,
        is_query=True,
    )
    docs_b = _prepare_documents(
        root_dir=Path(dir_b),
        output_dir=output_dir / "documents_b",
        side_prefix="B",
        sdk=sdk,
        glob_pattern=glob_pattern,
        limit=limit_b,
        is_query=False,
    )

    documents_a_csv = _write_document_index(docs_a, output_dir / "documents_a.csv")
    documents_b_csv = _write_document_index(docs_b, output_dir / "documents_b.csv")

    pairs_dir = output_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []

    for doc_a in docs_a:
        for doc_b in docs_b:
            pair_id = f"{doc_a.doc_id}__{doc_b.doc_id}"
            artifacts = _write_pair_artifacts(
                pair_id=pair_id,
                doc_a=doc_a,
                doc_b=doc_b,
                output_dir=pairs_dir / pair_id,
                sdk=sdk,
                top_k=top_k,
            )
            manifest = json.loads(artifacts.manifest_json.read_text(encoding="utf-8"))
            summary_rows.append(manifest)

    summary_csv = _write_summary_csv(summary_rows, output_dir / "document_pair_summary.csv")
    manifest_json = _write_corpus_manifest(
        output_dir=output_dir / "corpus_manifest.json",
        dir_a=Path(dir_a),
        dir_b=Path(dir_b),
        engine=engine,
        source_format=source_format,
        model_id=model_id,
        device=device,
        batch_size=batch_size,
        top_k=top_k,
        glob_pattern=glob_pattern,
        limit_a=limit_a,
        limit_b=limit_b,
        doc_count_a=len(docs_a),
        doc_count_b=len(docs_b),
        pair_count=len(summary_rows),
        documents_a_csv=documents_a_csv,
        documents_b_csv=documents_b_csv,
        summary_csv=summary_csv,
    )

    return {
        "documents_a_csv": documents_a_csv,
        "documents_b_csv": documents_b_csv,
        "summary_csv": summary_csv,
        "manifest_json": manifest_json,
    }


def _prepare_documents(
    *,
    root_dir: Path,
    output_dir: Path,
    side_prefix: str,
    sdk: SanskritResearchSDK,
    glob_pattern: str,
    limit: int | None,
    is_query: bool,
) -> list[CorpusDocument]:
    if not root_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root_dir}")
    if not root_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {root_dir}")

    files = sorted(path for path in root_dir.rglob(glob_pattern) if path.is_file())
    if limit is not None:
        if limit < 0:
            raise ValueError("Document limit must be non-negative.")
        files = files[:limit]
    if not files:
        raise ValueError(f"No files matched pattern {glob_pattern!r} under {root_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    documents: list[CorpusDocument] = []
    for index, path in enumerate(files, start=1):
        relative_path = path.relative_to(root_dir).as_posix()
        doc_id = f"{side_prefix}{index:03d}"
        if sdk.cache is not None:
            cached = sdk.cached_embed_file(path, is_query=is_query)
            sentences = cached.segments
            segments = make_segments(sentences)
            embedding_view = EmbeddingView(
                model_id=sdk.model_id,
                device=sdk.device,
                sentences=sentences,
                embeddings=np.asarray(cached.embeddings),
            )
        else:
            text = path.read_text(encoding="utf-8")
            seg_view = sdk.segment_text(text)
            sentences = []
            spans = []
            for segment_text, span in zip(seg_view.segments, seg_view.spans):
                if not segment_text.strip():
                    continue
                sentences.append(segment_text)
                spans.append(span)
            segments = make_segments(sentences, spans=spans)
            embedding_view = sdk.embed_sentences(sentences, is_query=is_query)
        sentences_csv = _write_sentence_index_csv(
            segments,
            output_dir / f"{doc_id}_sentences.csv",
        )
        embeddings_npy = output_dir / f"{doc_id}_embeddings.npy"
        np.save(embeddings_npy, embedding_view.embeddings.astype(np.float32, copy=False))
        documents.append(
            CorpusDocument(
                doc_id=doc_id,
                relative_path=relative_path,
                sentence_count=len(sentences),
                sentences_csv=sentences_csv,
                embeddings_npy=embeddings_npy,
                segments=segments,
                embedding_view=embedding_view,
            )
        )
    return documents


def _write_pair_artifacts(
    *,
    pair_id: str,
    doc_a: CorpusDocument,
    doc_b: CorpusDocument,
    output_dir: Path,
    sdk: SanskritResearchSDK,
    top_k: int,
) -> CorpusPairArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    pairwise_view = sdk.pairwise_from_embedding_views(
        doc_a.embedding_view,
        doc_b.embedding_view,
        top_k=top_k,
    )
    matrix = pairwise_view.similarity_matrix

    similarity_npy = output_dir / "similarity_matrix.npy"
    np.save(similarity_npy, matrix)

    topk_csv = write_topk_csv(pairwise_view.matches, output_dir / "topk_pairs.csv")
    topk_jsonl = write_topk_jsonl(pairwise_view.matches, output_dir / "topk_pairs.jsonl")
    sentences_a_csv = _write_sentence_index_csv(pairwise_view.segment_records_a, output_dir / "sentences_a.csv")
    sentences_b_csv = _write_sentence_index_csv(pairwise_view.segment_records_b, output_dir / "sentences_b.csv")

    manifest = {
        "pair_id": pair_id,
        "doc_a_id": doc_a.doc_id,
        "doc_a_relative_path": doc_a.relative_path,
        "doc_b_id": doc_b.doc_id,
        "doc_b_relative_path": doc_b.relative_path,
        "sentence_count_a": len(pairwise_view.segments_a),
        "sentence_count_b": len(pairwise_view.segments_b),
        "matrix_rows": int(matrix.shape[0]),
        "matrix_cols": int(matrix.shape[1]),
        "matrix_score_count": int(matrix.size),
        "max_score": pairwise_view.metrics.max_score,
        "mean_score": pairwise_view.metrics.mean_score,
        "median_score": pairwise_view.metrics.median_score,
        "p95_score": pairwise_view.metrics.p95_score,
        "mean_best_a_to_b": pairwise_view.metrics.mean_best_a_to_b,
        "mean_best_b_to_a": pairwise_view.metrics.mean_best_b_to_a,
        "top_k_requested": top_k,
        "top_k_returned": len(pairwise_view.matches),
        "top_k_note": "top-k files are a generated view; regenerate any k from similarity_npy plus sentence indexes",
        "sentences_a_csv": str(sentences_a_csv),
        "sentences_b_csv": str(sentences_b_csv),
        "topk_csv": str(topk_csv),
        "topk_jsonl": str(topk_jsonl),
        "similarity_npy": str(similarity_npy),
    }
    manifest_json = output_dir / "pair_manifest.json"
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return CorpusPairArtifacts(
        pair_id=pair_id,
        doc_a_id=doc_a.doc_id,
        doc_b_id=doc_b.doc_id,
        pair_dir=output_dir,
        sentences_a_csv=sentences_a_csv,
        sentences_b_csv=sentences_b_csv,
        topk_csv=topk_csv,
        topk_jsonl=topk_jsonl,
        similarity_npy=similarity_npy,
        manifest_json=manifest_json,
    )


def regenerate_topk_for_pair_dir(
    pair_dir: str | Path,
    *,
    k: int,
    mode: TopKMode = "raw",
    diversity_radius: int = 2,
    output_stem: str | None = None,
) -> TopKExportArtifacts:
    """Regenerate top-k match tables for any k from a saved corpus pair directory.

    The full `similarity_matrix.npy` is the durable evidence artifact. The CSV/JSONL
    top-k files are convenience views over that matrix and can be regenerated later
    without re-segmenting or re-embedding.
    """
    pair_dir = Path(pair_dir)
    manifest_path = pair_dir / "pair_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing pair manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matrix_path = _resolve_artifact_path(pair_dir, manifest["similarity_npy"])
    sentences_a_path = _resolve_artifact_path(pair_dir, manifest["sentences_a_csv"])
    sentences_b_path = _resolve_artifact_path(pair_dir, manifest["sentences_b_csv"])

    matrix = np.load(matrix_path)
    segments_a = read_sentence_index_csv(sentences_a_path)
    segments_b = read_sentence_index_csv(sentences_b_path)
    match_records = top_k_match_records(
        matrix,
        segments_a,
        segments_b,
        k,
        mode=mode,
        diversity_radius=diversity_radius,
    )
    matches = [
        PairMatch(
            rank=match.rank,
            score=match.score,
            i=match.segment_a.index,
            j=match.segment_b.index,
            sentence_a=match.segment_a.text,
            sentence_b=match.segment_b.text,
        )
        for match in match_records
    ]

    stem = output_stem or (f"topk_{k}" if mode == "raw" else f"topk_{mode}_{k}")
    topk_csv = write_topk_csv(matches, pair_dir / f"{stem}.csv")
    topk_jsonl = write_topk_jsonl(matches, pair_dir / f"{stem}.jsonl")
    return TopKExportArtifacts(
        topk_csv=topk_csv,
        topk_jsonl=topk_jsonl,
        k=k,
        match_count=len(matches),
        mode=mode,
    )


def read_sentence_index_csv(path: str | Path) -> list[PairwiseSegment]:
    """Read sentence-index artifacts back into pairwise segment records."""
    segments: list[PairwiseSegment] = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            segments.append(
                PairwiseSegment(
                    index=int(row["sentence_index"]),
                    text=row["sentence_text"],
                    start=_optional_int(row.get("start")),
                    end=_optional_int(row.get("end")),
                )
            )
    return segments


def _write_sentence_index_csv(segments: list[PairwiseSegment], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sentence_index", "sentence_text", "start", "end"])
        writer.writeheader()
        for segment in segments:
            writer.writerow(
                {
                    "sentence_index": segment.index,
                    "sentence_text": segment.text,
                    "start": segment.start,
                    "end": segment.end,
                }
            )
    return output_path


def _write_document_index(documents: list[CorpusDocument], output_path: Path) -> Path:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["doc_id", "relative_path", "sentence_count", "sentences_csv", "embeddings_npy"],
        )
        writer.writeheader()
        for document in documents:
            writer.writerow(
                {
                    "doc_id": document.doc_id,
                    "relative_path": document.relative_path,
                    "sentence_count": document.sentence_count,
                    "sentences_csv": str(document.sentences_csv),
                    "embeddings_npy": str(document.embeddings_npy),
                }
            )
    return output_path


def _write_summary_csv(summary_rows: list[dict[str, object]], output_path: Path) -> Path:
    fieldnames = [
        "pair_id",
        "doc_a_id",
        "doc_a_relative_path",
        "doc_b_id",
        "doc_b_relative_path",
        "sentence_count_a",
        "sentence_count_b",
        "matrix_rows",
        "matrix_cols",
        "matrix_score_count",
        "max_score",
        "mean_score",
        "median_score",
        "p95_score",
        "mean_best_a_to_b",
        "mean_best_b_to_a",
        "top_k_requested",
        "top_k_returned",
        "top_k_note",
        "sentences_a_csv",
        "sentences_b_csv",
        "topk_csv",
        "topk_jsonl",
        "similarity_npy",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    return output_path


def _write_corpus_manifest(
    *,
    output_dir: Path,
    dir_a: Path,
    dir_b: Path,
    engine: str,
    source_format: str,
    model_id: str,
    device: str,
    batch_size: int,
    top_k: int,
    glob_pattern: str,
    limit_a: int | None,
    limit_b: int | None,
    doc_count_a: int,
    doc_count_b: int,
    pair_count: int,
    documents_a_csv: Path,
    documents_b_csv: Path,
    summary_csv: Path,
) -> Path:
    manifest = {
        "dir_a": str(dir_a),
        "dir_b": str(dir_b),
        "engine": engine,
        "source_format": source_format,
        "model_id": model_id,
        "device": device,
        "batch_size": batch_size,
        "top_k": top_k,
        "top_k_note": "initial top-k files are generated views; each pair directory can regenerate arbitrary k from matrix and sentence indexes",
        "glob_pattern": glob_pattern,
        "limit_a": limit_a,
        "limit_b": limit_b,
        "doc_count_a": doc_count_a,
        "doc_count_b": doc_count_b,
        "pair_count": pair_count,
        "documents_a_csv": str(documents_a_csv),
        "documents_b_csv": str(documents_b_csv),
        "summary_csv": str(summary_csv),
    }
    output_dir.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_dir


def _resolve_artifact_path(pair_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return pair_dir / path.name


def _optional_int(value: str | None) -> int | None:
    if value in {None, ""}:
        return None
    return int(value)


def safe_slug(value: str) -> str:
    """Return a filesystem-friendly slug."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return slug or "document"
