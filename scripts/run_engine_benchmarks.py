#!/usr/bin/env python3
"""Run clumped pseudo-evaluation for multiple Sanskrit engines and compare summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sanskrit_pipeline.clumping import build_clumped_records
from sanskrit_pipeline.io import load_records
from sanskrit_pipeline.pipeline import SanskritPipeline, resolve_segmenter
from sanskrit_pipeline.pseudo_eval import compare_clump_to_prediction, summarize_pseudo_eval, write_pseudo_eval_csv
from sanskrit_pipeline.review import write_review_artifact

ENGINE_METADATA = {
    "dandas": {"engine_family": "regex", "engine_source": "sanskrit_pipeline"},
    "dandas_pada": {"engine_family": "regex", "engine_source": "sanskrit_pipeline"},
    "stanza": {"engine_family": "ml", "engine_source": "stanza"},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multi-engine clumped segmentation benchmarks for Sanskrit.")
    parser.add_argument("--input", required=True, help="Input file of upstream segmented sentence rows.")
    parser.add_argument("--output-dir", default="output/benchmarks", help="Benchmark output directory.")
    parser.add_argument("--text-column", default="input_text", help="Column containing Sanskrit text.")
    parser.add_argument("--input-format", default="devanagari", choices=["devanagari", "iast"])
    parser.add_argument("--clump-size", type=int, default=6)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--limit", type=int, default=12000)
    parser.add_argument(
        "--engines",
        nargs="+",
        default=["dandas", "dandas_pada"],
        choices=["dandas", "dandas_pada", "stanza"],
        help="dandas=double-daṇḍa only; dandas_pada=both daṇḍas; stanza=ML segmenter.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = load_records(args.input, text_column=args.text_column, limit=args.limit)
    clumps = build_clumped_records(records, clump_size=args.clump_size, stride=args.stride)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    comparison_rows: list[dict[str, str]] = []

    for engine_key in args.engines:
        engine_out = output_root / engine_key
        engine_out.mkdir(parents=True, exist_ok=True)

        split_on_single_danda = engine_key == "dandas_pada"
        engine_name = "dandas" if engine_key in {"dandas", "dandas_pada"} else engine_key
        segmenter = resolve_segmenter(engine=engine_name, split_on_single_danda=split_on_single_danda)
        pipeline = SanskritPipeline(segmenter)
        results = pipeline.run_segmentation(clumps, source_format=args.input_format)
        review_csv = write_review_artifact(results, engine_out / "clumped_segmentation_review.csv")
        pseudo_rows = [
            compare_clump_to_prediction(clump, result)
            for clump, result in zip(clumps, results, strict=False)
        ]
        pseudo_csv = write_pseudo_eval_csv(pseudo_rows, engine_out / "clumped_pseudo_eval.csv")
        summary = summarize_pseudo_eval(pseudo_rows)
        summary_path = engine_out / "clumped_pseudo_eval_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        metadata = ENGINE_METADATA[engine_key]
        comparison_rows.append(
            {
                "engine": engine_key,
                "engine_family": metadata["engine_family"],
                "engine_source": metadata["engine_source"],
                "clump_count": f"{summary['clump_count']:.0f}",
                "mean_source_recall": f"{summary['mean_source_recall']:.4f}",
                "mean_predicted_precision": f"{summary['mean_predicted_precision']:.4f}",
                "mean_boundary_precision": f"{summary['mean_boundary_precision']:.4f}",
                "mean_boundary_recall": f"{summary['mean_boundary_recall']:.4f}",
                "mean_boundary_f1": f"{summary['mean_boundary_f1']:.4f}",
                "review_csv": str(review_csv),
                "pseudo_eval_csv": str(pseudo_csv),
                "summary_json": str(summary_path),
            }
        )

    comparison_path = output_root / "comparison_summary.csv"
    with comparison_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "engine",
            "engine_family",
            "engine_source",
            "clump_count",
            "mean_source_recall",
            "mean_predicted_precision",
            "mean_boundary_precision",
            "mean_boundary_recall",
            "mean_boundary_f1",
            "review_csv",
            "pseudo_eval_csv",
            "summary_json",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)

    print(f"comparison_summary_csv={comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
