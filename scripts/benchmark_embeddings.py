#!/usr/bin/env python3
"""Benchmark embedding models and segmenters on hand-labelled Sanskrit pairs."""

from __future__ import annotations

import argparse
import csv
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sanskrit_pipeline.embeddings import DEFAULT_MODEL_ID, TextEmbedder
from sanskrit_pipeline.pairwise import segment_text_to_sentences
from sanskrit_pipeline.pairwise_run import cosine_similarity_matrix


class EmbedderProtocol(Protocol):
    model_id: str

    def encode(self, sentences: list[str]) -> np.ndarray: ...


@dataclass(slots=True)
class BenchmarkResult:
    model_id: str
    segmenter: str
    recall_at: dict[int, float]
    mrr: float
    runtime_seconds: float
    peak_ram_mb: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare embedding models on labelled Sanskrit retrieval pairs."
    )
    parser.add_argument("--text-a", required=True, help="Path to text A (.txt).")
    parser.add_argument("--text-b", required=True, help="Path to text B (.txt).")
    parser.add_argument(
        "--gold-pairs",
        required=True,
        help="CSV with columns seg_a_idx, seg_b_idx, is_match.",
    )
    parser.add_argument(
        "--segmenter",
        default="dandas",
        choices=["dandas", "prose", "hybrid", "stanza"],
    )
    parser.add_argument("--input-format", default="devanagari", choices=["devanagari", "iast"])
    parser.add_argument("--split-on-single-danda", action="store_true")
    parser.add_argument("--extra-model", action="append", default=[], help="Additional model id/path.")
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", help="Optional CSV path for benchmark results.")
    parser.add_argument("--recall-k", type=int, nargs="+", default=[1, 5, 10, 20])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    text_a = Path(args.text_a).read_text(encoding="utf-8")
    text_b = Path(args.text_b).read_text(encoding="utf-8")
    gold_pairs = _load_gold_pairs(Path(args.gold_pairs))

    sentences_a = segment_text_to_sentences(
        text_a,
        engine=args.segmenter,
        source_format=args.input_format,
        split_on_single_danda=args.split_on_single_danda,
    )
    sentences_b = segment_text_to_sentences(
        text_b,
        engine=args.segmenter,
        source_format=args.input_format,
        split_on_single_danda=args.split_on_single_danda,
    )

    model_ids = [DEFAULT_MODEL_ID, "sentence-transformers/LaBSE", *args.extra_model]
    results: list[BenchmarkResult] = []
    for model_id in model_ids:
        embedder = _build_embedder(model_id, device=args.device, batch_size=args.batch_size)
        if embedder is None:
            continue
        results.append(
            _evaluate_model(
                embedder=embedder,
                sentences_a=sentences_a,
                sentences_b=sentences_b,
                gold_pairs=gold_pairs,
                segmenter=args.segmenter,
                recall_k=args.recall_k,
            )
        )

    _print_markdown_table(results, recall_k=args.recall_k)
    if args.output:
        _write_csv(Path(args.output), results, recall_k=args.recall_k)
    return 0


def _build_embedder(model_id: str, *, device: str, batch_size: int) -> EmbedderProtocol | None:
    if model_id == DEFAULT_MODEL_ID:
        return _GemmaEmbedder(model_id=model_id, device=device, batch_size=batch_size)
    if model_id == "sentence-transformers/LaBSE":
        return _LabSEEmbedder(model_id=model_id, device=device, batch_size=batch_size)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(f"Skipping {model_id}: install sentence-transformers to use extra models.", file=sys.stderr)
        return None
    return _SentenceTransformerEmbedder(
        model_id=model_id,
        device=device,
        batch_size=batch_size,
        st_model=SentenceTransformer(model_id, device=device if device != "auto" else None),
    )


class _GemmaEmbedder:
    def __init__(self, *, model_id: str, device: str, batch_size: int) -> None:
        self.model_id = model_id
        self._embedder = TextEmbedder(model_id=model_id, device=device, batch_size=batch_size)

    def encode(self, sentences: list[str]) -> np.ndarray:
        return self._embedder.encode_queries(sentences).embeddings


class _LabSEEmbedder:
    def __init__(self, *, model_id: str, device: str, batch_size: int) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Install sentence-transformers to benchmark LaBSE."
            ) from exc
        self.model_id = model_id
        self._batch_size = batch_size
        self._model = SentenceTransformer("LaBSE", device=device if device != "auto" else None)

    def encode(self, sentences: list[str]) -> np.ndarray:
        return self._model.encode(
            sentences,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )


class _SentenceTransformerEmbedder:
    def __init__(self, *, model_id: str, device: str, batch_size: int, st_model: object) -> None:
        self.model_id = model_id
        self._batch_size = batch_size
        self._model = st_model

    def encode(self, sentences: list[str]) -> np.ndarray:
        return self._model.encode(
            sentences,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )


def _evaluate_model(
    *,
    embedder: EmbedderProtocol,
    sentences_a: list[str],
    sentences_b: list[str],
    gold_pairs: list[tuple[int, int]],
    segmenter: str,
    recall_k: list[int],
) -> BenchmarkResult:
    tracemalloc.start()
    started = time.perf_counter()
    embeddings_a = embedder.encode(sentences_a)
    embeddings_b = embedder.encode(sentences_b)
    matrix = cosine_similarity_matrix(embeddings_a, embeddings_b)
    runtime_seconds = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    ranks: list[int] = []
    for a_idx, b_idx in gold_pairs:
        row = matrix[a_idx]
        order = np.argsort(row)[::-1]
        rank = int(np.where(order == b_idx)[0][0]) + 1
        ranks.append(rank)

    recall_at = {
        k: float(np.mean([rank <= k for rank in ranks])) if ranks else 0.0
        for k in recall_k
    }
    mrr = float(np.mean([1.0 / rank for rank in ranks])) if ranks else 0.0
    return BenchmarkResult(
        model_id=embedder.model_id,
        segmenter=segmenter,
        recall_at=recall_at,
        mrr=mrr,
        runtime_seconds=runtime_seconds,
        peak_ram_mb=peak / (1024 * 1024),
    )


def _load_gold_pairs(path: Path) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("is_match", "").strip().lower() in {"false", "0", "no"}:
                continue
            pairs.append((int(row["seg_a_idx"]), int(row["seg_b_idx"])))
    if not pairs:
        raise ValueError(f"No positive gold pairs found in {path}")
    return pairs


def _print_markdown_table(results: list[BenchmarkResult], *, recall_k: list[int]) -> None:
    headers = ["model", "segmenter", "MRR", "runtime_s", "peak_ram_mb"] + [f"R@{k}" for k in recall_k]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for result in results:
        cells = [
            result.model_id,
            result.segmenter,
            f"{result.mrr:.4f}",
            f"{result.runtime_seconds:.3f}",
            f"{result.peak_ram_mb:.2f}",
        ]
        cells.extend(f"{result.recall_at[k]:.4f}" for k in recall_k)
        print("| " + " | ".join(cells) + " |")


def _write_csv(path: Path, results: list[BenchmarkResult], *, recall_k: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model_id", "segmenter", "mrr", "runtime_seconds", "peak_ram_mb"] + [
        f"recall_at_{k}" for k in recall_k
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {
                "model_id": result.model_id,
                "segmenter": result.segmenter,
                "mrr": result.mrr,
                "runtime_seconds": result.runtime_seconds,
                "peak_ram_mb": result.peak_ram_mb,
            }
            for k in recall_k:
                row[f"recall_at_{k}"] = result.recall_at[k]
            writer.writerow(row)


if __name__ == "__main__":
    raise SystemExit(main())
