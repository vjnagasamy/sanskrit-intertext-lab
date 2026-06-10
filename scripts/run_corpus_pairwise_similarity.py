#!/usr/bin/env python3
"""Run folder-to-folder corpus pairwise sentence similarity."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sanskrit_pipeline.corpus_pairwise import run_corpus_pairwise_similarity
from sanskrit_pipeline.embeddings import DEFAULT_MODEL_ID


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Segment and embed every Sanskrit text file in two corpus folders once, then "
            "write all cross-folder document-pair similarity artifacts."
        )
    )
    parser.add_argument("--dir-a", required=True, help="First corpus directory.")
    parser.add_argument("--dir-b", required=True, help="Second corpus directory.")
    parser.add_argument("--output-dir", required=True, help="Output directory for corpus artifacts.")
    parser.add_argument("--engine", default="dandas", choices=["dandas", "stanza", "stanza_iast", "lines"])
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
    parser.add_argument(
        "--embedding-progress",
        default="off",
        choices=["off", "batch", "sentence"],
        help="Embedding progress logging granularity.",
    )
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--glob-pattern", default="*.txt", help="File glob under each corpus directory.")
    parser.add_argument("--limit-a", type=int, help="Use only the first N files from dir A for smoke runs.")
    parser.add_argument("--limit-b", type=int, help="Use only the first N files from dir B for smoke runs.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected file counts and pair count without embedding.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        files_a = _selected_files(Path(args.dir_a), args.glob_pattern, args.limit_a)
        files_b = _selected_files(Path(args.dir_b), args.glob_pattern, args.limit_b)
        print(f"dir_a={Path(args.dir_a)}")
        print(f"dir_b={Path(args.dir_b)}")
        print(f"glob_pattern={args.glob_pattern}")
        print(f"doc_count_a={len(files_a)}")
        print(f"doc_count_b={len(files_b)}")
        print(f"pair_count={len(files_a) * len(files_b)}")
        return 0

    artifacts = run_corpus_pairwise_similarity(
        dir_a=args.dir_a,
        dir_b=args.dir_b,
        output_dir=args.output_dir,
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
    )
    print(f"documents_a_csv={artifacts['documents_a_csv']}")
    print(f"documents_b_csv={artifacts['documents_b_csv']}")
    print(f"summary_csv={artifacts['summary_csv']}")
    print(f"manifest_json={artifacts['manifest_json']}")
    return 0


def _selected_files(root_dir: Path, glob_pattern: str, limit: int | None) -> list[Path]:
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
    return files


if __name__ == "__main__":
    raise SystemExit(main())
