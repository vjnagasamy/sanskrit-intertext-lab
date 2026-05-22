# Sanskrit Research Pipeline

This context covers a research workflow for Sanskrit text segmentation, embedding, and cross-text similarity analysis. The repo exists to make those workflows reproducible across scripts, notebooks, and audit-friendly artifacts.

## Language

**Source Text**:
One Sanskrit text submitted to the pipeline for normalization, segmentation, or similarity analysis.
_Avoid_: raw blob, payload

**Source Record**:
One row-level input unit loaded from a tabular dataset for segmentation or evaluation.
_Avoid_: document, sample row

**Source Format**:
The textual representation of Sanskrit input, currently `devanagari` or `iast`.
_Avoid_: encoding, dialect

**Segmentation Engine**:
A sentence-segmentation strategy used to split Sanskrit text into sentence segments. Current engines: `dandas` (regex on ॥/।) and `stanza` (ML-based, optional).
_Avoid_: tokenizer, parser

**Sentence Segment**:
One sentence-like unit emitted by a segmentation engine, with text and character-span boundaries.
_Avoid_: chunk, token

**Segment Boundary**:
The character position where one sentence segment ends and the next begins, typically marked by a daṇḍa (। or ॥).
_Avoid_: split point, delimiter

**Daṇḍa**:
Sanskrit punctuation mark: single daṇḍa (।, `।`) marks a half-verse or clause boundary; double daṇḍa (॥, `॥`) marks a verse or sentence boundary.
_Avoid_: period, full stop

**Pāda-level Segmentation**:
Segmentation that also splits on single daṇḍas, producing half-verse units rather than full verses.
_Avoid_: fine-grained split

**Segmentation Review Artifact**:
Human-reviewable output that records normalized text, engine choice, and emitted sentence segments.
_Avoid_: debug dump, log

**Passage Clump**:
A synthetic passage formed by joining neighboring source sentences for segmentation pseudo-evaluation.
_Avoid_: batch, window

**Pseudo-evaluation**:
A lightweight evaluation that compares predicted sentence segments against upstream sentence rows using exact-match and boundary metrics.
_Avoid_: gold evaluation, benchmark

**Embedding Backend**:
The model-backed adapter that turns sentence segments into vector embeddings. Default: `buddhist-nlp/gemma-2-mitra-e` (Gemma-2-2B fine-tuned on Buddhist corpus including Sanskrit).
_Avoid_: encoder stack, model wrapper

**Pairwise Similarity Run**:
One comparison between two source texts that produces an `A x B` similarity matrix and ranked sentence-pair matches.
_Avoid_: search job, diff

**Top-k Match Table**:
The ranked list of highest-scoring sentence pairs from a pairwise similarity run.
_Avoid_: leaderboard, hit list

**Pairwise Segment Record**:
One sentence segment carried through a pairwise workflow with stable index and optional source spans.
_Avoid_: row wrapper, match input

**Corpus Document**:
One file-backed text in a corpus-level pairwise workflow, prepared once with sentence segments and embeddings for reuse across many comparisons.
_Avoid_: record, row

**Document Pair Summary**:
An aggregate row that describes one corpus-document comparison using matrix-level similarity signals and artifact paths.
_Avoid_: scorecard, report card

**Sentence Index Artifact**:
A tabular artifact that maps sentence indices back to sentence text and source spans for later audit.
_Avoid_: lookup dump, side table

## Relationships

- A **Source Text** has one **Source Format** before normalization.
- A **Segmentation Engine** turns one **Source Text** into many **Sentence Segments** and **Segment Boundaries**.
- A **Segmentation Review Artifact** records the output of one segmentation pass over many **Source Records** or **Passage Clumps**.
- A **Passage Clump** is built from neighboring sentences taken from many **Source Records**.
- A **Pseudo-evaluation** compares predicted **Sentence Segments** against sentence rows implied by a **Passage Clump**.
- An **Embedding Backend** turns many **Sentence Segments** into vectors for a **Pairwise Similarity Run**.
- A **Pairwise Similarity Run** uses **Pairwise Segment Records** and produces one similarity matrix, one **Top-k Match Table**, and aggregate metrics.
- A **Corpus Document** participates in many **Pairwise Similarity Runs** inside one corpus-level workflow.
- A **Sentence Index Artifact** records the source mapping for **Pairwise Segment Records**.
- A **Document Pair Summary** summarizes one **Pairwise Similarity Run** between two **Corpus Documents**.

## Example dialogue

> **Dev:** "For this corpus comparison, should each file be embedded again for every document pair?"
> **Domain expert:** "No. Prepare each **Corpus Document** once, then reuse those sentence embeddings across each **Pairwise Similarity Run**."
>
> **Dev:** "When we check segmentation quality, do we treat a joined passage like a normal source row?"
> **Domain expert:** "No. That joined passage is a **Passage Clump**, and we score it with **Pseudo-evaluation** rather than treating it like a normal **Source Record**."
>
> **Dev:** "Should verse numbers like ॥१॥ appear as segments in the pairwise matrix?"
> **Domain expert:** "No. Those digit-only tokens after the double daṇḍa are **verse number** markers, not content segments. The **DandasSegmenter** filters them out."
