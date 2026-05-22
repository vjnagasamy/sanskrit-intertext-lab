#!/usr/bin/env python3
"""Run pairwise sentence similarity for two input Sanskrit text files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sanskrit_pipeline.embeddings import DEFAULT_MODEL_ID
from sanskrit_pipeline.pairwise import run_pairwise_similarity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Segment two Sanskrit texts, embed sentence lists, compute cross-text "
            "cosine similarities, and export global top-k sentence pairs."
        )
    )
    parser.add_argument("--text-a", required=True, help="Path to text A (.txt).")
    parser.add_argument("--text-b", required=True, help="Path to text B (.txt).")
    parser.add_argument("--output-dir", required=True, help="Output directory for top-k artifacts.")
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
    parser.add_argument(
        "--embedding-progress",
        default="off",
        choices=["off", "batch", "sentence"],
        help="Embedding progress logging granularity.",
    )
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--save-similarity-npy", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    text_a = Path(args.text_a).read_text(encoding="utf-8")
    text_b = Path(args.text_b).read_text(encoding="utf-8")

    artifacts = run_pairwise_similarity(
        text_a=text_a,
        text_b=text_b,
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
        save_similarity_npy=args.save_similarity_npy,
    )
    print(f"topk_csv={artifacts.topk_csv}")
    print(f"topk_jsonl={artifacts.topk_jsonl}")
    print(f"manifest_json={artifacts.manifest_json}")
    if artifacts.similarity_npy:
        print(f"similarity_npy={artifacts.similarity_npy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
