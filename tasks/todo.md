# Task Plan

## 2026-05-20 Sanskrit Intertext Lab — Initial Build

- [x] Port segmenters: DandasSegmenter (regex, double/single daṇḍa), StanzaSegmenter (optional ML), BaseSegmenter ABC.
- [x] Port normalization with IAST→Devanagari via indic-transliteration; add explicit rejection of Tibetan format names.
- [x] Port embeddings with Sanskrit query instruction.
- [x] Port core modules: io, clumping, review, pseudo_eval, pipeline, pairwise, pairwise_run.
- [x] Port SDK (SanskritResearchSDK) with split_on_single_danda replacing botok params throughout the 6-site param chain.
- [x] Port corpus_pairwise and corpus_bidirectional with Sanskrit defaults.
- [x] Port reports: corpus_pairwise_report, bidirectional_synthesis_report — update all Tibetan/SMDG/Txt-18 strings.
- [x] Port CLI with dandas/stanza engine choices, devanagari/iast format choices, split_on_single_danda flag.
- [x] Create scripts/ directory with all 11 scripts ported to Sanskrit defaults.
- [x] Create tests/ directory with all 11 test files ported to Sanskrit sample text and patch targets.
- [x] Create pyproject.toml (drop botok/pyewts/[tool.uv], add indic-transliteration, stanza as optional).
- [x] Create requirements.txt, environment.yml, .python-version.
- [x] Create CONTEXT.md, AGENTS.md docs.
- [ ] Create notebooks/: 01_segmentation_starter.ipynb, 02_pairwise_starter.ipynb, 03_corpus_pairwise_starter.ipynb.
- [ ] Run full test suite and fix any import or wiring issues.

## Next Steps

- Run smoke test: `python -m unittest discover -s tests -v` to validate all test wiring.
- Run a real segmentation smoke test with a small Devanagari file.
- Download gemma-2-mitra-e and validate end-to-end embedding with Sanskrit text.
- Run `scripts/run_bidirectional_corpus_pairwise.py --dry-run` on sample corpus folders.
