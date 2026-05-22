#!/usr/bin/env python3
"""Regenerate top-k match tables from an existing corpus pair directory."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sanskrit_pipeline.corpus_pairwise import regenerate_topk_for_pair_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate top-k CSV/JSONL match tables from a saved pair directory. "
            "Requires pair_manifest.json, similarity_matrix.npy, and sentence indexes."
        )
    )
    parser.add_argument("--pair-dir", required=True, help="Path to one pairs/<pair_id>/ directory.")
    parser.add_argument("--k", required=True, type=int, help="Number of highest-scoring matches to export.")
    parser.add_argument(
        "--mode",
        default="raw",
        choices=["raw", "unique_a", "unique_b", "unique_both", "diverse_both"],
        help="Ranking mode. Defaults to raw global top-k.",
    )
    parser.add_argument(
        "--diversity-radius",
        type=int,
        default=2,
        help="Sentence-index exclusion radius for diverse_both mode.",
    )
    parser.add_argument(
        "--output-stem",
        help="Output basename without extension. Defaults to topk_<k>.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts = regenerate_topk_for_pair_dir(
        args.pair_dir,
        k=args.k,
        mode=args.mode,
        diversity_radius=args.diversity_radius,
        output_stem=args.output_stem,
    )
    print(f"topk_csv={artifacts.topk_csv}")
    print(f"topk_jsonl={artifacts.topk_jsonl}")
    print(f"k={artifacts.k}")
    print(f"mode={artifacts.mode}")
    print(f"match_count={artifacts.match_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
