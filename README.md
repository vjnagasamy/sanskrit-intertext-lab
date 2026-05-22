# Sanskrit Intertext Lab — Architecture & Usage Guide

> Reference notes for `sanskrit-intertext-lab/` — a Sanskrit-native port of the Tibetan Intertext Lab pipeline. Tibetan source: https://github.com/ten-jampa/tibetan-intertext-lab. Adapted for Sanskrit using Claude Code
---

## What This Repo Is

A **pure-inference NLP pipeline** for cross-textual Sanskrit research. It takes Sanskrit text (Devanagari or IAST romanization), segments it into sentence units on daṇḍa punctuation, embeds those sentences using `buddhist-nlp/gemma-2-mitra-e`, and runs pairwise cosine similarity across corpora.

The use case is **intertextuality detection**: finding semantically similar passages across Sanskrit Buddhist (and broader Sanskrit) texts — quotations, formulaic phrases, shared doctrinal vocabulary.

**Key difference from the Tibetan repo**: segmentation is far simpler (regex on daṇḍa marks rather than morphosyntactic word lists), but the embedding backend and all pairwise/corpus workflows are identical in structure.

---

## Architecture Overview

```
Raw Text (CSV/TSV/JSONL/TXT or .txt corpus files)
    ↓
normalize_text()            ← Unicode NFC; optional IAST→Devanagari via indic-transliteration
    ↓
DandasSegmenter.segment()   ← Regex split on ॥ (and optionally ।); verse-number filter
    ↓
TextEmbedder.encode*()      ← Dense vectors via Gemma-2 last-token pooling
    ↓
cosine_similarity_matrix()  ← A×B similarity matrix
    ↓
top_k_match_records()       ← Ranked sentence pairs (raw / unique / diverse modes)
    ↓
topk_pairs.csv / topk_pairs.jsonl / document_pair_summary.csv
    ↓
HTML reports (corpus_pairwise_report, bidirectional_synthesis_report)
```

---

## The Embedding Model

**Model**: `buddhist-nlp/gemma-2-mitra-e` (Gemma-2-2B fine-tuned on Pali, Sanskrit, Tibetan Buddhist corpus)

**How embeddings work** (`sanskrit_pipeline/embeddings.py`):
- Loaded via `AutoModelForCausalLM.from_pretrained()`
- Forward pass → **last hidden state of the last non-padding token** → L2-normalized vector
- Cosine similarity reduces to dot product on normalized vectors
- **Asymmetric retrieval**: query sentences get an instruction prefix:
  ```
  <instruct>Please find the semantically most similar text in Sanskrit.\n<query>{text}
  ```
  Corpus sentences are encoded raw (no prefix).
- **Fallback chain** if not Gemma: `sentence-transformers` → generic mean pooling

**Device/precision support**: auto-detects CUDA > MPS > CPU; `float16`/`bfloat16`/`float32`; 8-bit quantization via BitsAndBytes; `device_map` for multi-GPU.

---

## Segmentation

This is the main language-specific component. Sanskrit segmentation is **much simpler** than Tibetan because Sanskrit uses explicit daṇḍa punctuation with no morphosyntactic ambiguity.

### DandasSegmenter (`sanskrit_pipeline/segmenters/dandas.py`)

| Mode | How it splits |
|---|---|
| Default (`split_on_single_danda=False`) | On `॥` (double daṇḍa, U+0965) only — full verses |
| Pāda-level (`split_on_single_danda=True`) | On both `।` (U+0964) and `॥` — half-verses |

**Post-split filters** (both applied after every segment boundary):
- `_has_devanagari()` — drops segments with no Devanagari letter/digit content (excludes daṇḍas themselves: U+0964–U+0965 are not counted)
- `_is_verse_number()` — drops digit/space/punctuation-only tokens like `॥ १ ॥` or `॥ 1.2 ॥`

These two filters prevent verse-number markers and stray Latin metadata from polluting the sentence index.

### StanzaSegmenter (`sanskrit_pipeline/segmenters/stanza_segmenter.py`)

Optional ML-based segmenter. Import-guarded — the module loads cleanly even if `stanza` is not installed:

```python
from sanskrit_pipeline.segmenters import stanza_available
if stanza_available():
    from sanskrit_pipeline.segmenters import StanzaSegmenter
```

Activate with `--engine stanza`. Download the model first:
```bash
python scripts/download_stanza_sanskrit.py
```

---

## Source Formats

| Format | Value | Description |
|---|---|---|
| Devanagari | `"devanagari"` | Default. Unicode NFC normalization only. |
| IAST romanization | `"iast"` | Transliterated via `indic-transliteration` → Devanagari, then NFC. |

Passing `"unicode"` or `"wylie"` raises an explicit `ValueError` pointing to the correct format names (those are Tibetan pipeline format names and would otherwise silently produce bad output).

---

## How to Run It

**Setup**:
```bash
conda env create -f environment.yml
conda activate sanskrit-intertext-env
pip install indic-transliteration   # not on conda-forge; needed for IAST input
python scripts/download_gemma_mitra.py   # pre-cache model (~5 GB)
```

**Segmentation only**:
```bash
python scripts/run_sanskrit_pipeline.py \
  --input data/input.csv \
  --output-dir output/seg \
  --engine dandas
```

**Two-text pairwise similarity**:
```bash
python scripts/run_pairwise_text_similarity.py \
  --text-a data/text_a.txt \
  --text-b data/text_b.txt \
  --output-dir output/pair_run \
  --device auto \
  --top-k 100
```

**Corpus pairwise (all pairs across two folders)**:
```bash
python scripts/run_corpus_pairwise_similarity.py \
  --dir-a data/corpus_a/ \
  --dir-b data/corpus_b/ \
  --output-dir output/corpus_run \
  --device auto
```

**Bidirectional (A→B and B→A + synthesis report)**:
```bash
python scripts/run_bidirectional_corpus_pairwise.py \
  --dir-a data/corpus_a/ \
  --dir-b data/corpus_b/ \
  --output-dir output/bidir_run \
  --label-a "Mūlamadhyamakakārikā" \
  --label-b "Bodhicaryāvatāra" \
  --device auto
```

**Dry run** (count files/pairs, no embedding):
```bash
python scripts/run_corpus_pairwise_similarity.py \
  --dir-a data/corpus_a/ --dir-b data/corpus_b/ \
  --output-dir output/tmp --dry-run
```

**Notebook SDK** (clean Python API):
```python
from sanskrit_pipeline.sdk import SanskritResearchSDK

sdk = SanskritResearchSDK(engine="dandas", device="auto")
seg = sdk.segment_text(text)
emb_a = sdk.embed_sentences(seg.segments, is_query=True)
emb_b = sdk.embed_sentences(other_segments, is_query=False)
view = sdk.pairwise_from_embedding_views(emb_a, emb_b, top_k=20)
view.topk_dataframe()
```

---

## File Structure

```
sanskrit-intertext-lab/
├── sanskrit_pipeline/
│   ├── __init__.py                    — Package exports
│   ├── cli.py                         — CLI entry point (argparse)
│   ├── embeddings.py                  — TextEmbedder; Gemma + fallback backends
│   ├── io.py                          — InputRecord; CSV/TSV/JSONL/TXT loaders
│   ├── normalization.py               — normalize_text(); NFC + IAST→Devanagari
│   ├── pipeline.py                    — SanskritPipeline orchestrator
│   ├── pairwise.py                    — cosine_similarity_matrix, global_top_k_matches
│   ├── pairwise_run.py                — run_pairwise_similarity_core; top_k_match_records
│   ├── corpus_pairwise.py             — Folder-to-folder corpus workflow
│   ├── corpus_bidirectional.py        — A→B and B→A orchestration + synthesis
│   ├── sdk.py                         — SanskritResearchSDK (notebook-friendly)
│   ├── clumping.py                    — Passage clumps for pseudo-evaluation
│   ├── pseudo_eval.py                 — Boundary metrics for segmentation eval
│   ├── review.py                      — CSV generation for segmentation output
│   ├── segmenters/
│   │   ├── base.py                    — BaseSegmenter ABC + Sanskrit Unicode constants
│   │   ├── dandas.py                  — DandasSegmenter (regex, always available)
│   │   └── stanza_segmenter.py        — StanzaSegmenter (optional ML, guarded import)
│   └── reports/
│       ├── corpus_pairwise_report.py  — Per-run interactive HTML report
│       └── bidirectional_synthesis_report.py — Forward+reverse synthesis HTML report
├── scripts/                           — Standalone CLI runners
├── tests/                             — unittest suite (53 tests)
├── notebooks/                         — Jupyter starter notebooks (3)
├── tasks/                             — Planning and lessons docs
├── pyproject.toml
├── requirements.txt
├── environment.yml
├── CONTEXT.md                         — Domain term glossary
└── AGENTS.md                          — Coding and agent guidelines
```

---

## Top-K Ranking Modes

The pairwise core supports five modes for selecting sentence-pair matches from the similarity matrix:

| Mode | Behavior |
|---|---|
| `raw` | Global top-k by score; a sentence can appear multiple times |
| `unique_a` | Each sentence in A appears at most once (best match per A sentence) |
| `unique_b` | Each sentence in B appears at most once |
| `unique_both` | Each sentence in A and B appears at most once (bipartite matching) |
| `diverse_both` | Like `unique_both` but also excludes index neighbors within `diversity_radius` |

The default corpus run uses `raw`. Use `scripts/export_pair_topk.py` to regenerate any mode from saved artifacts without re-embedding.

---

## Corpus Workflow Artifacts

After a corpus pairwise run, each `pairs/<pair_id>/` directory contains:

```
pair_manifest.json          — metadata, top_k params, paths
similarity_matrix.npy       — full A×B cosine similarity matrix
sentences_a.csv             — sentence index for text A
sentences_b.csv             — sentence index for text B
topk_raw_100.csv            — top-100 global matches
topk_raw_100.jsonl          — same, JSON lines format
```

The `document_pair_summary.csv` at the run root aggregates per-pair metrics (`max_score`, `p95_score`, `mean_best_a_to_b`, `mean_best_b_to_a`) for all pairs — this is what the HTML reports and synthesis workflow read.

---

## Bidirectional Synthesis Report

`run_bidirectional_corpus_pairwise` runs both A→B and B→A, then merges their summaries into a **survival-ranked** pair list. A pair ranks high only if its signal is strong in **both** model prompt directions — this filters out direction-specific artifacts.

Four survival scores contribute to the composite:

| Score | Based on |
|---|---|
| `broad_affinity_survival` | p95 of the similarity matrix in both runs (35% weight) |
| `corpus_a_to_b_survival` | Mean best A→B score in both runs (25%) |
| `corpus_b_to_a_survival` | Mean best B→A score in both runs (25%) |
| `local_spike_survival` | Max single sentence-pair score in both runs (15%) |

The HTML report at `synthesis/report/index.html` is self-contained (data embedded in `report_data.js`) and opens locally in any browser.

---

## The `split_on_single_danda` Param Chain

This flag must be threaded through **6 sites** — missing any one silently breaks pāda-level segmentation for corpus runs:

1. `SanskritResearchSDK.__init__()` — stored as `self.split_on_single_danda`
2. `SanskritResearchSDK.bidirectional_corpus_pairwise()` — included in `defaults` dict
3. `run_corpus_pairwise_similarity()` function signature
4. `SanskritResearchSDK(...)` constructor call inside `corpus_pairwise.py`
5. Forward `run_corpus_pairwise_similarity()` call in `corpus_bidirectional.py`
6. Reverse `run_corpus_pairwise_similarity()` call in `corpus_bidirectional.py`

All 6 sites are wired correctly. If you add a new corpus entry point, replicate this pattern.

---

## Testing

```bash
python -m unittest discover -s tests -v
```

53 tests; 1 skips if `indic-transliteration` is not installed (IAST normalization test). All others run with no real model or GPU — mocks cover the embedding layer.

Key test targets:
- `sanskrit_pipeline.corpus_pairwise.SanskritResearchSDK` — patch this to inject `FakeSDK` in corpus tests
- `sanskrit_pipeline.pairwise.segment_text_to_sentences` — patch for pairwise unit tests
- `sanskrit_pipeline.embeddings.TextEmbedder` — patch for SDK and device tests

---

## Comparison to Tibetan Intertext Lab

| Aspect | Tibetan | Sanskrit |
|---|---|---|
| Segmenter | Botok tokenizer + morphosyntactic heuristics | DandasSegmenter (regex on ॥/।) |
| Segmentation complexity | High (40+ particle/terminator word lists) | Low (two Unicode codepoints + two filters) |
| Normalization dep | `pyewts` (Wylie→Unicode; needs `--no-build-isolation`) | `indic-transliteration` (IAST→Devanagari; pip install) |
| Source formats | `unicode`, `wylie` | `devanagari`, `iast` |
| Engine choices | `botok`, `botok_ours`, `botok_intellexus`, `regex_intellexus` | `dandas`, `dandas_pada` (benchmark alias), `stanza` (optional) |
| Embedding model | `buddhist-nlp/gemma-2-mitra-e` | Same |
| Query instruction | "…in Tibetan." | "…in Sanskrit." |
| `[tool.uv]` in pyproject | Yes (needed for pyewts) | Removed |
| Corpus/pairwise workflows | Identical structure | Identical structure |
| HTML reports | Same codebase, Tibetan labels | Same codebase, generic Corpus A/B labels |

---

## Strengths

- **Simple, robust segmentation**: daṇḍa marks are unambiguous; the regex never misses a boundary
- **Verse-number filtering**: `_is_verse_number` and `_has_devanagari` prevent digit tokens from entering the embedding index
- **Full corpus workflow**: embed once per document, reuse across all pairs — efficient at scale
- **Bidirectional synthesis**: direction-invariant ranking catches real parallelism, not model-direction artifacts
- **Modular**: segmenters swappable via ABC interface; `StanzaSegmenter` is opt-in
- **No heavy deps at import time**: `stanza` import is guarded; the core pipeline works with just `torch` + `transformers`

## Limitations

- **Regex segmentation is naive for prose**: compounds and sandhi mean double-daṇḍa segments vary enormously in length; prose texts (not verse) may need the Stanza ML segmenter
- **IAST transliteration requires `indic-transliteration`**: not on conda-forge, must be pip-installed separately
- **Pseudo-eval only**: no gold-standard Sanskrit sentence segmentation benchmark integrated
- **Gemma Mitra is Buddhist-corpus fine-tuned**: may underperform on Vedic, technical grammar (Pāṇini), or secular Sanskrit texts that differ in vocabulary from the training distribution
- **MPS (Apple Silicon) can be fragile**: reduce batch_size to 1 or fall back to CPU if you hit MPS memory errors
