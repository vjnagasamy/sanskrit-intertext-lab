#!/usr/bin/env python3
"""Run corpus pairwise similarity in both directions and generate reports."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_corpus_pairwise_similarity import _selected_files
from sanskrit_pipeline.corpus_bidirectional import run_bidirectional_corpus_pairwise
from sanskrit_pipeline.embeddings import DEFAULT_MODEL_ID


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run forward and reverse corpus pairwise similarity, then generate forward, reverse, and synthesis reports."
    )
    parser.add_argument("--dir-a", required=True, help="First corpus directory.")
    parser.add_argument("--dir-b", required=True, help="Second corpus directory.")
    parser.add_argument("--output-dir", required=True, help="Output directory for bidirectional artifacts.")
    parser.add_argument("--label-a", default="Corpus A", help="Human-readable label for dir A.")
    parser.add_argument("--label-b", default="Corpus B", help="Human-readable label for dir B.")
    parser.add_argument("--engine", default="dandas", choices=["dandas", "stanza"])
    parser.add_argument("--input-format", default="devanagari", choices=["devanagari", "iast"])
    parser.add_argument(
        "--split-on-single-danda",
        action="store_true",
        help="Also split on single daṇḍa (।) in addition to double daṇḍa (॥). Produces pāda-level segments.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", help="Transformers device_map value, for example 'auto'.")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--low-cpu-mem-usage", action="store_true")
    parser.add_argument("--embedding-progress", default="off", choices=["off", "batch", "sentence"])
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--glob-pattern", default="*.txt", help="File glob under each corpus directory.")
    parser.add_argument("--limit-a", type=int, help="Use only first N files from dir A for smoke runs.")
    parser.add_argument("--limit-b", type=int, help="Use only first N files from dir B for smoke runs.")
    parser.add_argument("--skip-forward", action="store_true", help="Do not run forward direction; requires existing forward artifacts.")
    parser.add_argument("--skip-reverse", action="store_true", help="Do not run reverse direction; requires existing reverse artifacts.")
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse completed forward/reverse artifacts when present.")
    parser.add_argument("--reports-only", action="store_true", help="Only regenerate reports from existing forward/reverse artifacts.")
    parser.add_argument("--no-reports", action="store_true", help="Run forward/reverse artifacts without generating reports.")
    parser.add_argument("--report-heatmap-size", type=int, default=56)
    parser.add_argument("--report-max-topk", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true", help="Print selected file counts and pair counts without embedding.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        files_a = _selected_files(Path(args.dir_a), args.glob_pattern, args.limit_a)
        files_b = _selected_files(Path(args.dir_b), args.glob_pattern, args.limit_b)
        print(f"forward={args.label_a} -> {args.label_b}")
        print(f"reverse={args.label_b} -> {args.label_a}")
        print(f"dir_a={Path(args.dir_a)}")
        print(f"dir_b={Path(args.dir_b)}")
        print(f"glob_pattern={args.glob_pattern}")
        print(f"doc_count_a={len(files_a)}")
        print(f"doc_count_b={len(files_b)}")
        print(f"pair_count_per_direction={len(files_a) * len(files_b)}")
        print(f"total_pair_matrices={2 * len(files_a) * len(files_b)}")
        return 0

    reports_only = args.reports_only
    artifacts = run_bidirectional_corpus_pairwise(
        dir_a=args.dir_a,
        dir_b=args.dir_b,
        output_dir=args.output_dir,
        label_a=args.label_a,
        label_b=args.label_b,
        engine=args.engine,
        source_format=args.input_format,
        split_on_single_danda=args.split_on_single_danda,
        model_id=args.model_id,
        batch_size=args.batch_size,
        device=args.device,
        embedding_progress=args.embedding_progress,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        load_in_8bit=args.load_in_8bit,
        low_cpu_mem_usage=args.low_cpu_mem_usage or None,
        top_k=args.top_k,
        glob_pattern=args.glob_pattern,
        limit_a=args.limit_a,
        limit_b=args.limit_b,
        run_forward=not args.skip_forward and not reports_only,
        run_reverse=not args.skip_reverse and not reports_only,
        reuse_existing=args.reuse_existing or reports_only,
        generate_reports=not args.no_reports,
        report_heatmap_size=args.report_heatmap_size,
        report_max_topk=args.report_max_topk,
    )
    print(f"forward_manifest={artifacts.forward['manifest_json']}")
    print(f"reverse_manifest={artifacts.reverse['manifest_json']}")
    print(f"bidirectional_manifest={artifacts.manifest_json}")
    if artifacts.forward_report_html:
        print(f"forward_report={artifacts.forward_report_html}")
    if artifacts.reverse_report_html:
        print(f"reverse_report={artifacts.reverse_report_html}")
    if artifacts.synthesis_report_html:
        print(f"synthesis_report={artifacts.synthesis_report_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
