"""Bidirectional corpus pairwise orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .corpus_pairwise import run_corpus_pairwise_similarity
from .embeddings import DEFAULT_MODEL_ID, TorchDTypeName
from .embedding_cache import EmbeddingCache
from .reports import generate_bidirectional_synthesis_report, generate_corpus_pairwise_report


@dataclass(slots=True)
class BidirectionalCorpusArtifacts:
    """Artifacts produced by a bidirectional corpus pairwise run."""

    output_dir: Path
    forward: dict[str, Path]
    reverse: dict[str, Path]
    manifest_json: Path
    forward_report_html: Path | None = None
    reverse_report_html: Path | None = None
    synthesis_report_html: Path | None = None
    synthesis_csv: Path | None = None
    synthesis_json: Path | None = None


def run_bidirectional_corpus_pairwise(
    dir_a: str | Path,
    dir_b: str | Path,
    output_dir: str | Path,
    *,
    label_a: str = "Corpus A",
    label_b: str = "Corpus B",
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
    run_forward: bool = True,
    run_reverse: bool = True,
    reuse_existing: bool = False,
    generate_reports: bool = True,
    report_heatmap_size: int = 56,
    report_max_topk: int = 100,
    embedding_cache: EmbeddingCache | None = None,
) -> BidirectionalCorpusArtifacts:
    """Run A->B and B->A corpus comparisons, then generate three reports.

    `reuse_existing=True` lets callers regenerate reports/synthesis from existing
    forward and reverse run dirs without spending embedding/GPU time again.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    forward_dir = output_dir / "forward"
    reverse_dir = output_dir / "reverse"

    if run_forward and not (reuse_existing and _run_exists(forward_dir)):
        run_corpus_pairwise_similarity(
            dir_a=dir_a,
            dir_b=dir_b,
            output_dir=forward_dir,
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
            top_k=top_k,
            glob_pattern=glob_pattern,
            limit_a=limit_a,
            limit_b=limit_b,
            embedding_cache=embedding_cache,
        )

    if run_reverse and not (reuse_existing and _run_exists(reverse_dir)):
        run_corpus_pairwise_similarity(
            dir_a=dir_b,
            dir_b=dir_a,
            output_dir=reverse_dir,
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
            top_k=top_k,
            glob_pattern=glob_pattern,
            limit_a=limit_b,
            limit_b=limit_a,
            embedding_cache=embedding_cache,
        )

    forward = _existing_run_artifacts(forward_dir)
    reverse = _existing_run_artifacts(reverse_dir)

    forward_report_html: Path | None = None
    reverse_report_html: Path | None = None
    synthesis_report_html: Path | None = None
    synthesis_csv: Path | None = None
    synthesis_json: Path | None = None
    if generate_reports:
        forward_report_html = generate_corpus_pairwise_report(
            forward_dir,
            heatmap_size=report_heatmap_size,
            max_topk=report_max_topk,
        )
        reverse_report_html = generate_corpus_pairwise_report(
            reverse_dir,
            heatmap_size=report_heatmap_size,
            max_topk=report_max_topk,
        )
        synthesis_report_html = generate_bidirectional_synthesis_report(
            forward_dir,
            reverse_dir,
            output_dir / "synthesis" / "report",
        )
        synthesis_csv = output_dir / "synthesis" / "report" / "synthesis.csv"
        synthesis_json = output_dir / "synthesis" / "report" / "synthesis.json"

    manifest_json = _write_bidirectional_manifest(
        output_dir=output_dir,
        dir_a=Path(dir_a),
        dir_b=Path(dir_b),
        label_a=label_a,
        label_b=label_b,
        forward_dir=forward_dir,
        reverse_dir=reverse_dir,
        generate_reports=generate_reports,
        forward_report_html=forward_report_html,
        reverse_report_html=reverse_report_html,
        synthesis_report_html=synthesis_report_html,
    )

    return BidirectionalCorpusArtifacts(
        output_dir=output_dir,
        forward=forward,
        reverse=reverse,
        manifest_json=manifest_json,
        forward_report_html=forward_report_html,
        reverse_report_html=reverse_report_html,
        synthesis_report_html=synthesis_report_html,
        synthesis_csv=synthesis_csv,
        synthesis_json=synthesis_json,
    )


def _run_exists(run_dir: Path) -> bool:
    return (
        (run_dir / "corpus_manifest.json").exists()
        and (run_dir / "documents_a.csv").exists()
        and (run_dir / "documents_b.csv").exists()
        and (run_dir / "document_pair_summary.csv").exists()
    )


def _existing_run_artifacts(run_dir: Path) -> dict[str, Path]:
    if not _run_exists(run_dir):
        raise FileNotFoundError(f"Missing completed corpus run artifacts under: {run_dir}")
    return {
        "documents_a_csv": run_dir / "documents_a.csv",
        "documents_b_csv": run_dir / "documents_b.csv",
        "summary_csv": run_dir / "document_pair_summary.csv",
        "manifest_json": run_dir / "corpus_manifest.json",
    }


def _write_bidirectional_manifest(
    *,
    output_dir: Path,
    dir_a: Path,
    dir_b: Path,
    label_a: str,
    label_b: str,
    forward_dir: Path,
    reverse_dir: Path,
    generate_reports: bool,
    forward_report_html: Path | None,
    reverse_report_html: Path | None,
    synthesis_report_html: Path | None,
) -> Path:
    manifest = {
        "dir_a": str(dir_a),
        "dir_b": str(dir_b),
        "label_a": label_a,
        "label_b": label_b,
        "forward_run_dir": str(forward_dir),
        "reverse_run_dir": str(reverse_dir),
        "generate_reports": generate_reports,
        "forward_report_html": str(forward_report_html) if forward_report_html else None,
        "reverse_report_html": str(reverse_report_html) if reverse_report_html else None,
        "synthesis_report_html": str(synthesis_report_html) if synthesis_report_html else None,
    }
    manifest_json = output_dir / "bidirectional_manifest.json"
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_json
