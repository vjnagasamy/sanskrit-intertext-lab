# Repository Guidelines

## Agent skills

### Issue tracker

Issues tracked in GitHub Issues for this repo.

### Domain docs

Domain docs use single-context layout with root `CONTEXT.md`. See `CONTEXT.md` for term definitions.

## Project Structure & Module Organization
- `sanskrit_pipeline/`: main Python package (normalization, segmentation, clumping, pseudo-eval, embeddings, pairwise core, CLI, SDK).
- `sanskrit_pipeline/segmenters/`: pluggable segmenter interface — `DandasSegmenter` (regex, always available) and `StanzaSegmenter` (optional ML, guarded import).
- `sanskrit_pipeline/reports/`: HTML report generators for corpus pairwise and bidirectional synthesis workflows.
- `scripts/`: runnable entry points for end-to-end workflows (pipeline run, pairwise text, corpus pairwise, bidirectional, reports, benchmarks).
- `tests/`: unit tests (`test_*.py`) covering CLI, normalization, pairwise core, clumping, segmenters, SDK, and device selection.
- `tasks/`: lightweight planning and lessons docs.
- `notebooks/`: Jupyter starter notebooks for segmentation, pairwise, and corpus workflows.

## Build, Test, and Development Commands
- Install deps: `python -m pip install -r requirements.txt`
- Run all tests: `python -m unittest discover -s tests -v`
- Run segmentation pipeline: `python scripts/run_sanskrit_pipeline.py --input <file> --output-dir output/<run_name> --engine dandas`
- Run multi-engine benchmark: `python scripts/run_engine_benchmarks.py --input <file> --output-dir output/benchmarks --engines dandas dandas_pada`
- Run two-text pairwise similarity: `python scripts/run_pairwise_text_similarity.py --text-a <file_a> --text-b <file_b> --output-dir output/<run_name>`
- Run corpus pairwise: `python scripts/run_corpus_pairwise_similarity.py --dir-a <corpus_a/> --dir-b <corpus_b/> --output-dir output/<run_name>`
- Run bidirectional corpus pairwise: `python scripts/run_bidirectional_corpus_pairwise.py --dir-a <corpus_a/> --dir-b <corpus_b/> --output-dir output/<run_name>`
- Dry run (file counts only): add `--dry-run` to corpus scripts
- Download model: `python scripts/download_gemma_mitra.py`
- Download Stanza Sanskrit model: `python scripts/download_stanza_sanskrit.py`

## Sanskrit-specific Notes
- **Default engine**: `dandas` — splits on double daṇḍa (॥) only. Use `--split-on-single-danda` for pāda-level segments.
- **Default format**: `devanagari` — Devanagari Unicode text. Use `--input-format iast` for IAST romanized input.
- **StanzaSegmenter**: optional; install with `pip install stanza` then `python scripts/download_stanza_sanskrit.py`. Tests that require it use `@unittest.skipUnless(stanza_available(), ...)`.
- **Embedding model**: `buddhist-nlp/gemma-2-mitra-e` — a Gemma-2-2B model fine-tuned on Buddhist texts (Pali, Sanskrit, Tibetan). Requires ~5GB VRAM for float16 inference.
- **Format errors**: passing `source_format="unicode"` or `"wylie"` raises an explicit error directing users to use `"devanagari"` or `"iast"` instead.

## Coding Style & Naming Conventions
- Use Python 3 with 4-space indentation, type hints, and `from __future__ import annotations` in new modules when appropriate.
- Prefer `snake_case` for functions/variables, `PascalCase` for classes, and descriptive module names.
- Keep functions focused and side effects explicit; prefer `pathlib.Path` for filesystem logic.
- Match existing docstring style: short, purpose-first module/class docstrings.
- No comments unless the WHY is non-obvious. No multi-line comment blocks.

## Testing Guidelines
- Framework: built-in `unittest` (no pytest dependency required).
- Place tests in `tests/` and name files `test_<feature>.py`; test methods should start with `test_`.
- Sanskrit sample text for tests: use short Devanagari strings such as `"धर्मो रक्षति रक्षितः॥"` or pipe-delimited fake segments like `"alpha|shared"`.
- Patch targets: use `sanskrit_pipeline.corpus_pairwise.SanskritResearchSDK` (not the Tibetan equivalent).
- Add or update tests for behavior changes, especially CLI argument wiring, pairwise metrics/ranking semantics, and segmentation boundaries.
- Run `python -m unittest discover -s tests -v` before opening a PR.

## Commit & Pull Request Guidelines
- Follow Conventional Commit prefixes: `feat:`, `docs:`, `fix:`, `test:`, `refactor:`.
- Keep commits scoped to one logical change and include tests/docs in the same PR when relevant.
- PRs should include: concise summary, why the change is needed, validation steps/commands run, and sample output paths (for script changes).
- For workflow changes, include a small reproducible command example in the PR description.

## Security & Configuration Tips
- Do not commit large input datasets or generated outputs (`data/`, `output/` are typically local artifacts).
- Avoid hardcoded absolute machine-specific paths in scripts or config.
