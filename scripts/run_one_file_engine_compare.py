#!/usr/bin/env python3
"""Run multi-engine segmentation comparison on one raw Sanskrit text file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sanskrit_pipeline.clumping import build_clumped_records, source_sentences_to_json
from sanskrit_pipeline.io import InputRecord
from sanskrit_pipeline.pipeline import SanskritPipeline, resolve_segmenter
from sanskrit_pipeline.review import write_review_artifact
from sanskrit_pipeline.segmenters.base import DANDA, DOUBLE_DANDA

DEFAULT_ENGINES = ["dandas", "dandas_pada"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Split one raw Sanskrit file into daṇḍa-based units, clump units, "
            "run multiple engines, and write side-by-side review outputs."
        )
    )
    parser.add_argument("--input-file", required=True, help="Path to one raw Sanskrit .txt file.")
    parser.add_argument("--output-dir", default="output/one_file_compare", help="Output directory.")
    parser.add_argument("--source-format", default="devanagari", choices=["devanagari", "iast"])
    parser.add_argument("--clump-size", type=int, default=6)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--unit-offset", type=int, default=0)
    parser.add_argument("--unit-limit", type=int, default=1200)
    parser.add_argument(
        "--engines",
        nargs="+",
        default=DEFAULT_ENGINES,
        choices=["dandas", "dandas_pada", "stanza"],
        help="dandas=double-daṇḍa only; dandas_pada=both daṇḍas; stanza=ML segmenter.",
    )
    return parser


def split_units_from_raw_text(text: str) -> list[str]:
    """Split raw Sanskrit text into candidate sentence units on daṇḍa punctuation."""
    collapsed = re.sub(r"\s+", " ", text).strip()
    if not collapsed:
        return []
    pattern = re.compile(rf"[^{re.escape(DANDA)}{re.escape(DOUBLE_DANDA)}]+[{re.escape(DANDA)}{re.escape(DOUBLE_DANDA)}]*")
    units = [match.strip() for match in pattern.findall(collapsed) if match.strip()]
    return units if units else [collapsed]


def build_side_by_side_rows(
    clumps: list[InputRecord],
    results_by_engine: dict[str, list],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for clump_index, clump in enumerate(clumps):
        row: dict[str, str] = {
            "record_id": clump.record_id,
            "source_format": "",
            "normalized_text": "",
            "source_units_json": source_sentences_to_json(getattr(clump, "source_sentences", [])),
        }
        for engine, engine_results in results_by_engine.items():
            result = engine_results[clump_index]
            if not row["normalized_text"]:
                row["normalized_text"] = result.normalized_text
            if not row["source_format"]:
                row["source_format"] = result.source_format
            row[f"{engine}_segment_count"] = str(len(result.segments))
            row[f"{engine}_segments_json"] = json.dumps(result.segments, ensure_ascii=False)
        rows.append(row)
    return rows


def write_side_by_side_csv(
    output_path: Path,
    rows: list[dict[str, str]],
    engines: list[str],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["record_id", "source_format", "normalized_text", "source_units_json"]
    for engine in engines:
        fieldnames.append(f"{engine}_segment_count")
    for engine in engines:
        fieldnames.append(f"{engine}_segments_json")

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    input_file = Path(args.input_file)
    raw_text = input_file.read_text(encoding="utf-8")
    units = split_units_from_raw_text(raw_text)
    selected_units = units[args.unit_offset : args.unit_offset + args.unit_limit]
    records = [InputRecord(record_id=f"unit-{idx:06d}", text=text) for idx, text in enumerate(selected_units)]
    clumps = build_clumped_records(records, clump_size=args.clump_size, stride=args.stride)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_by_engine: dict[str, list] = {}
    review_paths: dict[str, str] = {}

    for engine_key in args.engines:
        engine_dir = output_dir / engine_key
        engine_dir.mkdir(parents=True, exist_ok=True)

        split_on_single_danda = engine_key == "dandas_pada"
        engine_name = "dandas" if engine_key in {"dandas", "dandas_pada"} else engine_key
        segmenter = resolve_segmenter(engine=engine_name, split_on_single_danda=split_on_single_danda)
        pipeline = SanskritPipeline(segmenter)
        results = pipeline.run_segmentation(clumps, source_format=args.source_format)
        review_path = write_review_artifact(results, engine_dir / "one_file_review.csv")

        results_by_engine[engine_key] = results
        review_paths[engine_key] = str(review_path)

    side_by_side_rows = build_side_by_side_rows(clumps, results_by_engine)
    side_by_side_path = write_side_by_side_csv(
        output_path=output_dir / "manual_review_side_by_side.csv",
        rows=side_by_side_rows,
        engines=args.engines,
    )

    manifest = {
        "input_file": str(input_file),
        "unit_count_raw": len(units),
        "unit_offset": args.unit_offset,
        "unit_limit": args.unit_limit,
        "unit_count_selected": len(selected_units),
        "clump_size": args.clump_size,
        "stride": args.stride,
        "clump_count": len(clumps),
        "engines": args.engines,
        "review_csv_by_engine": review_paths,
        "side_by_side_csv": str(side_by_side_path),
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"side_by_side_csv={side_by_side_path}")
    print(f"run_manifest_json={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
