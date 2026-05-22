#!/usr/bin/env python3
"""Download the Stanza Sanskrit model into local cache."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download the Stanza Sanskrit language model.")
    parser.add_argument(
        "--lang",
        default="sa",
        help="Stanza language code for Sanskrit (default: 'sa').",
    )
    parser.add_argument(
        "--dir",
        default=None,
        help="Optional Stanza model directory override.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        import stanza
    except ImportError:
        print("[error] stanza is not installed. Run: pip install stanza")
        return 1

    kwargs: dict[str, object] = {"lang": args.lang}
    if args.dir:
        kwargs["dir"] = args.dir

    print(f"[downloading] stanza model for language '{args.lang}'...")
    stanza.download(**kwargs)
    print("[ok] stanza Sanskrit model downloaded successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
