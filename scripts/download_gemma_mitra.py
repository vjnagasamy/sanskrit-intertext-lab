#!/usr/bin/env python3
"""Download and verify buddhist-nlp/gemma-2-mitra-e into local HF cache."""

from __future__ import annotations

import argparse
import os
import time

from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "buddhist-nlp/gemma-2-mitra-e"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prefetch Gemma Mitra model to local Hugging Face cache.")
    parser.add_argument("--model-id", default=MODEL_ID, help="Hugging Face model id.")
    parser.add_argument("--cache-dir", default=None, help="Optional HF cache directory override.")
    parser.add_argument("--max-attempts", type=int, default=8, help="Retry attempts for flaky networks.")
    parser.add_argument("--sleep-seconds", type=int, default=8, help="Wait time between attempts.")
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable name containing Hugging Face token (optional but recommended).",
    )
    return parser


def verify_local(model_id: str, cache_dir: str | None) -> None:
    AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        local_files_only=True,
        trust_remote_code=True,
    )
    AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        local_files_only=True,
        trust_remote_code=True,
    )


def main() -> int:
    args = build_parser().parse_args()
    token = os.getenv(args.token_env)

    for attempt in range(1, args.max_attempts + 1):
        print(f"[attempt {attempt}/{args.max_attempts}] downloading {args.model_id}...")
        try:
            snapshot_download(
                repo_id=args.model_id,
                cache_dir=args.cache_dir,
                token=token,
                resume_download=True,
            )
            print("[attempt] download step completed, verifying local files...")
            verify_local(args.model_id, args.cache_dir)
            print("[ok] local cache is complete and loadable with local_files_only=True")
            return 0
        except Exception as exc:
            print(f"[attempt] failed: {type(exc).__name__}: {exc}")
            if attempt == args.max_attempts:
                print("[error] reached max attempts; rerun to continue from partial cache.")
                return 1
            time.sleep(args.sleep_seconds)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
