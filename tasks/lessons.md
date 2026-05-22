# Lessons

- Sanskrit daṇḍa segmentation is simpler than Tibetan botok but requires verse-number filtering (`_is_verse_number`) and a Devanagari character guard (`_has_devanagari`) to avoid polluting segments with digit-only verse markers or Latin text.
- The `split_on_single_danda` flag must thread through 6 sites: SDK constructor, SDK defaults dict in `bidirectional_corpus_pairwise`, `run_corpus_pairwise_similarity` signature, `SanskritResearchSDK` constructor call inside `corpus_pairwise.py`, and both `run_corpus_pairwise_similarity` calls in `corpus_bidirectional.py`. Missing any one site breaks the param silently.
- `indic-transliteration` is not available on conda-forge as a package; install it via the `pip:` block in `environment.yml` or directly with pip.
- `stanza` is optional and its import is guarded; tests that require it should be decorated with `@unittest.skipUnless(stanza_available(), "stanza not installed")`.
- Always check corpus size before launching a non-trivial pipeline job; choose an explicit sample limit for interactive evaluation and reserve full-corpus runs for deliberate batch jobs.
- When caching expensive inference objects, key the cache by heavyweight load-time settings only; mutable runtime knobs like batch size and progress logging should update the existing instance instead of forcing a model reload.
- Do not assume corpus filenames are readable evidence labels. Reports should show a human-readable corpus/title label first, then expose exact filenames and paths as provenance.
- The `readable_label()` function in the synthesis report should be generic (not corpus-specific regex) so it works for any file naming convention.
