from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from sanskrit_pipeline.pairwise_run import PairwiseMatchRecord, PairwiseSegment, top_k_match_records

TOPK_MODES = ["raw", "unique_a", "unique_b", "unique_both", "diverse_both"]


def generate_corpus_pairwise_report(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    heatmap_size: int = 56,
    max_topk: int = 100,
) -> Path:
    """Generate an interactive HTML report for one completed corpus pairwise run."""
    run_dir = Path(run_dir)
    report_dir = Path(output_dir) if output_dir is not None else run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    data = build_report_data(run_dir, heatmap_size, max_topk)
    (report_dir / "report_data.js").write_text(
        "window.CORPUS_REPORT_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    (report_dir / "index.html").write_text(build_html(), encoding="utf-8")
    return report_dir / "index.html"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a local interactive HTML report for a corpus pairwise run."
    )
    parser.add_argument("--run-dir", required=True, help="Corpus pairwise output directory.")
    parser.add_argument("--output-dir", help="Report directory. Defaults to <run-dir>/report.")
    parser.add_argument("--heatmap-size", type=int, default=56, help="Downsampled pair heatmap size.")
    parser.add_argument("--max-topk", type=int, default=100, help="Top-k rows to embed per pair.")
    args = parser.parse_args()

    print(
        generate_corpus_pairwise_report(
            args.run_dir,
            args.output_dir,
            heatmap_size=args.heatmap_size,
            max_topk=args.max_topk,
        )
    )


def build_report_data(run_dir: Path, heatmap_size: int, max_topk: int) -> dict[str, Any]:
    manifest = json.loads((run_dir / "corpus_manifest.json").read_text(encoding="utf-8"))
    docs_a = read_csv(run_dir / "documents_a.csv")
    docs_b = read_csv(run_dir / "documents_b.csv")
    pairs = read_csv(run_dir / "document_pair_summary.csv")

    for rows in (docs_a, docs_b):
        for row in rows:
            row["sentence_count"] = int(row["sentence_count"])
            row["short_name"] = shorten_path(row["relative_path"])
            row["embedding_shape"] = list(np.load(resolve_run_path(run_dir, row["embeddings_npy"]), mmap_mode="r").shape)
            row["sentences"] = read_doc_sentences(resolve_run_path(run_dir, row["sentences_csv"]))

    pair_payloads: list[dict[str, Any]] = []
    for row in pairs:
        pair_dir = run_dir / "pairs" / row["pair_id"]
        matrix = np.load(pair_dir / "similarity_matrix.npy", mmap_mode="r")
        segments_a = read_segments(resolve_run_path(run_dir, row["sentences_a_csv"]))
        segments_b = read_segments(resolve_run_path(run_dir, row["sentences_b_csv"]))
        topk_by_mode = {
            mode: match_records_to_topk(
                top_k_match_records(
                    matrix,
                    segments_a,
                    segments_b,
                    max_topk,
                    mode=mode,  # type: ignore[arg-type]
                    diversity_radius=2,
                )
            )
            for mode in TOPK_MODES
        }
        pair_payloads.append(
            {
                "pair_id": row["pair_id"],
                "doc_a_id": row["doc_a_id"],
                "doc_b_id": row["doc_b_id"],
                "doc_a_name": shorten_path(row["doc_a_relative_path"]),
                "doc_b_name": shorten_path(row["doc_b_relative_path"]),
                "doc_a_relative_path": row["doc_a_relative_path"],
                "doc_b_relative_path": row["doc_b_relative_path"],
                "sentence_count_a": int(row["sentence_count_a"]),
                "sentence_count_b": int(row["sentence_count_b"]),
                "score_count": int(row["matrix_score_count"]),
                "max_score": round_float(row["max_score"]),
                "mean_score": round_float(row["mean_score"]),
                "median_score": round_float(row["median_score"]),
                "p95_score": round_float(row["p95_score"]),
                "mean_best_a_to_b": round_float(row["mean_best_a_to_b"]),
                "mean_best_b_to_a": round_float(row["mean_best_b_to_a"]),
                "top_k_returned": int(row["top_k_returned"]),
                "similarity_npy": row["similarity_npy"],
                "topk_csv": row["topk_csv"],
                "topk": topk_by_mode["raw"],
                "topk_by_mode": topk_by_mode,
                "heatmap": downsample_matrix(matrix, heatmap_size),
                "stream_profiles": build_stream_profiles(matrix, heatmap_size * 2),
            }
        )

    gpu = read_gpu_monitor(run_dir / "logs" / "gpu_monitor.csv")
    stderr_path = run_dir / "logs" / "full_run.stderr.log"
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
    timing = parse_time_log(stderr)

    return {
        "run": {
            "title": "Full Corpus Pairwise Run",
            "model_id": manifest.get("model_id"),
            "device": manifest.get("device"),
            "batch_size": manifest.get("batch_size"),
            "pair_count": manifest.get("pair_count"),
            "doc_count_a": len(docs_a),
            "doc_count_b": len(docs_b),
            "total_sentences": sum(row["sentence_count"] for row in docs_a + docs_b),
            "embedding_dim": docs_a[0]["embedding_shape"][1] if docs_a else None,
            "wall_time": timing.get("wall_time"),
            "max_resident_set_kb": timing.get("max_resident_set_kb"),
            "max_gpu_mem_mib": max((row["mem_used_mib"] for row in gpu), default=None),
            "max_gpu_util_pct": max((row["gpu_util_pct"] for row in gpu), default=None),
            "heatmap_size": heatmap_size,
        },
        "documentsA": docs_a,
        "documentsB": docs_b,
        "pairs": pair_payloads,
        "gpu": gpu,
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_doc_sentences(path: Path) -> list[str]:
    return [segment.text for segment in read_segments(path)]


def resolve_run_path(run_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    try:
        return run_dir / path.relative_to(run_dir.as_posix())
    except ValueError:
        return run_dir / path.name


def shorten_path(path: str) -> str:
    name = Path(path).name
    return re.sub(r"\.txt$", "", name)


def round_float(value: str | float) -> float:
    return round(float(value), 6)


def normalize_topk(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        normalized.append(
            {
                "rank": int(row["rank"]),
                "score": round_float(row["score"]),
                "i": int(row["i"]),
                "j": int(row["j"]),
                "sentence_a": row["sentence_a"],
                "sentence_b": row["sentence_b"],
            }
        )
    return normalized


def read_segments(path: Path) -> list[PairwiseSegment]:
    segments: list[PairwiseSegment] = []
    for row in read_csv(path):
        segments.append(
            PairwiseSegment(
                index=int(row["sentence_index"]),
                text=row["sentence_text"],
                start=optional_int(row.get("start")),
                end=optional_int(row.get("end")),
            )
        )
    return segments


def optional_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def match_records_to_topk(records: list[PairwiseMatchRecord]) -> list[dict[str, Any]]:
    return [
        {
            "rank": match.rank,
            "score": round(float(match.score), 6),
            "i": match.segment_a.index,
            "j": match.segment_b.index,
            "sentence_a": match.segment_a.text,
            "sentence_b": match.segment_b.text,
        }
        for match in records
    ]


def downsample_matrix(matrix: np.ndarray, size: int) -> list[list[float]]:
    rows, cols = matrix.shape
    row_edges = np.linspace(0, rows, min(size, rows) + 1, dtype=int)
    col_edges = np.linspace(0, cols, min(size, cols) + 1, dtype=int)
    sampled: list[list[float]] = []
    for r0, r1 in zip(row_edges[:-1], row_edges[1:]):
        out_row = []
        for c0, c1 in zip(col_edges[:-1], col_edges[1:]):
            block = matrix[r0:max(r1, r0 + 1), c0:max(c1, c0 + 1)]
            out_row.append(round(float(np.mean(block)), 4))
        sampled.append(out_row)
    return sampled


def downsample_vector(vector: np.ndarray, size: int) -> list[float]:
    edges = np.linspace(0, len(vector), min(size, len(vector)) + 1, dtype=int)
    sampled = []
    for start, end in zip(edges[:-1], edges[1:]):
        block = vector[start:max(end, start + 1)]
        sampled.append(round(float(np.mean(block)), 4))
    return sampled


def build_stream_profiles(matrix: np.ndarray, size: int) -> dict[str, dict[str, list[float]]]:
    modes = {
        "best_1": 1,
        "top_3_avg": 3,
        "top_5_avg": 5,
    }
    profiles: dict[str, dict[str, list[float]]] = {}
    for mode, k in modes.items():
        profiles[mode] = {
            "a_to_b": downsample_vector(top_k_axis_mean(matrix, axis=1, k=k), size),
            "b_to_a": downsample_vector(top_k_axis_mean(matrix, axis=0, k=k), size),
        }
    return profiles


def top_k_axis_mean(matrix: np.ndarray, *, axis: int, k: int) -> np.ndarray:
    axis_len = matrix.shape[axis]
    effective_k = min(k, axis_len)
    if effective_k == 1:
        return np.max(matrix, axis=axis)
    partition_index = axis_len - effective_k
    top_values = np.partition(matrix, partition_index, axis=axis)
    if axis == 1:
        return np.mean(top_values[:, partition_index:], axis=1)
    if axis == 0:
        return np.mean(top_values[partition_index:, :], axis=0)
    raise ValueError(f"Unsupported axis: {axis}")


def read_gpu_monitor(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for row in read_csv(path):
        try:
            rows.append(
                {
                    "timestamp": row["timestamp"],
                    "gpu_util_pct": int(row["gpu_util_pct"].strip()),
                    "mem_used_mib": int(row["mem_used_mib"].strip()),
                    "mem_total_mib": int(row["mem_total_mib"].strip()),
                }
            )
        except (KeyError, ValueError):
            continue
    return rows


def parse_time_log(text: str) -> dict[str, Any]:
    wall = re.search(r"Elapsed \\(wall clock\\) time.*: ([^\\n]+)", text)
    rss = re.search(r"Maximum resident set size \\(kbytes\\): ([0-9]+)", text)
    return {
        "wall_time": wall.group(1).strip() if wall else None,
        "max_resident_set_kb": int(rss.group(1)) if rss else None,
    }


def build_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sanskrit Corpus Pairwise Report</title>
  <script src="report_data.js"></script>
  <style>
    :root {
      --ink: #171717;
      --muted: #666;
      --line: #d8d8d8;
      --panel: #f7f7f5;
      --accent: #245c63;
      --accent-2: #a6422a;
      --gold: #b3852f;
      --bg: #fffffc;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); }
    header { padding: 26px 32px 18px; border-bottom: 1px solid var(--line); background: #fff; }
    .top-link { display: inline-block; margin-bottom: 10px; color: var(--accent); font-size: 13px; font-weight: 700; text-decoration: none; }
    h1 { margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }
    h2 { margin: 0 0 12px; font-size: 18px; letter-spacing: 0; }
    h3 { margin: 0 0 8px; font-size: 15px; letter-spacing: 0; }
    .sub { color: var(--muted); font-size: 14px; }
    main { display: grid; grid-template-columns: minmax(320px, 420px) 1fr; min-height: calc(100vh - 93px); }
    aside { border-right: 1px solid var(--line); padding: 18px; background: var(--panel); overflow: auto; max-height: calc(100vh - 93px); }
    section { padding: 22px 26px 36px; overflow: auto; max-height: calc(100vh - 93px); }
    .stats { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-bottom: 18px; }
    .stat { border: 1px solid var(--line); background: #fff; padding: 10px; border-radius: 6px; min-height: 64px; }
    .stat b { display: block; font-size: 20px; margin-bottom: 2px; }
    .stat span { color: var(--muted); font-size: 12px; }
    .controls { display: grid; gap: 10px; margin-bottom: 14px; }
    label { display: grid; gap: 4px; font-size: 12px; color: var(--muted); }
    input, select { width: 100%; border: 1px solid var(--line); border-radius: 6px; background: #fff; color: var(--ink); padding: 8px 10px; font-size: 14px; }
    button { border: 1px solid var(--line); border-radius: 6px; background: #fff; color: var(--ink); padding: 8px 10px; font-size: 13px; cursor: pointer; }
    button.active { background: var(--accent); border-color: var(--accent); color: #fff; }
    .label-row { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
    .info-btn { width: 22px; height: 22px; border-radius: 999px; padding: 0; font-size: 12px; font-weight: 700; line-height: 1; color: var(--accent); }
    .help-popover { display: none; border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: #fff; color: var(--ink); font-size: 12px; line-height: 1.45; box-shadow: 0 12px 28px rgba(0,0,0,.08); }
    .help-popover.open { display: block; }
    .metric-buttons { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px; }
    .file-browser { border: 1px solid var(--line); border-radius: 8px; background: #fff; padding: 10px; display: grid; gap: 8px; }
    .file-browser-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
    .file-browser-title { font-size: 12px; color: var(--muted); font-weight: 700; }
    .file-browser-list { display: grid; gap: 5px; max-height: 190px; overflow: auto; }
    .file-option { text-align: left; font-size: 12px; line-height: 1.25; padding: 7px; }
    .file-option b { display: block; font-size: 12px; margin-bottom: 2px; }
    .file-option span { color: var(--muted); }
    .pair-list { display: grid; gap: 6px; }
    .pair-row { border: 1px solid var(--line); background: #fff; border-radius: 6px; padding: 9px; cursor: pointer; }
    .pair-row.active { border-color: var(--accent); box-shadow: inset 4px 0 0 var(--accent); }
    .pair-title { font-weight: 700; font-size: 13px; margin-bottom: 4px; }
    .pair-meta { color: var(--muted); font-size: 12px; display: flex; gap: 8px; flex-wrap: wrap; }
    .grid { display: grid; gap: 18px; }
    .panel { border: 1px solid var(--line); border-radius: 8px; background: #fff; padding: 16px; }
    .two { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 18px; }
    .canvas-wrap { width: 100%; overflow: auto; }
    canvas { display: block; max-width: 100%; border: 1px solid var(--line); background: #fff; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { border-bottom: 1px solid var(--line); padding: 8px; text-align: left; vertical-align: top; }
    th { color: var(--muted); font-size: 12px; font-weight: 700; background: #fafafa; position: sticky; top: 0; }
    .matches { display: grid; gap: 10px; }
    .match { border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: #fff; cursor: pointer; }
    .match.active { border-color: var(--accent); box-shadow: 0 0 0 2px rgba(36, 92, 99, .12); }
    .match-head { display: flex; justify-content: space-between; gap: 12px; color: var(--muted); font-size: 12px; margin-bottom: 8px; }
    .sentences { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .sentence { font-size: 15px; line-height: 1.6; padding: 10px; border-radius: 6px; background: var(--panel); }
    .context-panel { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--line); }
    .context-col { border: 1px solid var(--line); border-radius: 8px; overflow: hidden; background: #fbfbf7; }
    .context-title { padding: 8px 10px; background: #ece8d9; font-weight: 700; font-size: 13px; }
    .context-row { display: grid; grid-template-columns: 48px 1fr; gap: 8px; padding: 8px 10px; border-top: 1px solid var(--line); font-size: 14px; line-height: 1.55; }
    .context-row.hit { background: #fff7d6; }
    .context-index { color: var(--muted); font-variant-numeric: tabular-nums; }
    .note { color: var(--muted); font-size: 12px; line-height: 1.45; }
    .method-note { border-left: 4px solid var(--accent); background: var(--panel); padding: 12px 14px; margin: 0 0 14px; font-size: 13px; line-height: 1.5; }
    .method-note p { margin: 0 0 8px; }
    .method-note p:last-child { margin-bottom: 0; }
    .bar { height: 8px; background: #e9e4d9; border-radius: 999px; overflow: hidden; }
    .bar > span { display: block; height: 100%; background: var(--accent); }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
      aside, section { max-height: none; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .two, .sentences, .context-panel { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <a class="top-link" href="../">Reports index</a>
    <h1>Sanskrit Corpus Pairwise Report</h1>
    <div class="sub" id="runSubtitle"></div>
  </header>
  <main>
    <aside>
      <div class="stats" id="stats"></div>
      <div class="controls">
        <label><span class="label-row"><span>Search pair or file</span><button class="info-btn" type="button" data-help="helpSearch" aria-label="Explain search">i</button></span>
          <input id="search" list="fileOptions" placeholder="A001, file-name, Corpus A...">
          <datalist id="fileOptions"></datalist>
          <div class="help-popover" id="helpSearch">Filters the pair list by pair id, file id, or filename. Clicking an available file inserts its id so you can see all pairs involving that file.</div>
        </label>
        <div class="file-browser">
          <div class="file-browser-head">
            <div class="file-browser-title">Available Files</div>
            <button id="clearSearch" type="button">Clear</button>
          </div>
          <div class="file-browser-list" id="fileBrowser"></div>
        </div>
        <label><span class="label-row"><span>Sort pairs by</span><button class="info-btn" type="button" data-help="helpSort" aria-label="Explain sort metrics">i</button></span>
          <select id="sortMetric">
            <option value="max_score">Max score</option>
            <option value="mean_best_a_to_b">Mean best A to B</option>
            <option value="mean_best_b_to_a">Mean best B to A</option>
            <option value="p95_score">P95 score</option>
            <option value="mean_score">Mean score</option>
            <option value="score_count">Matrix size</option>
          </select>
          <div class="help-popover" id="helpSort">Max is the strongest single sentence hit. P95 is the broad upper tail. Mean best A to B asks how well each Corpus A sentence finds a match in Corpus B; B to A asks the reverse. Matrix size is sentence count A times sentence count B.</div>
        </label>
        <label><span class="label-row"><span>Top K shown for selected pair</span><button class="info-btn" type="button" data-help="helpTopK" aria-label="Explain top K">i</button></span>
          <input id="topK" type="number" min="1" max="100" value="20">
          <div class="help-popover" id="helpTopK">Controls how many sentence-pair matches are displayed for the selected document pair. The saved matrix supports regenerating other K values later.</div>
        </label>
        <label><span class="label-row"><span>Match view</span><button class="info-btn" type="button" data-help="helpMatchView" aria-label="Explain match views">i</button></span>
          <select id="matchMode">
            <option value="raw">Raw top-k</option>
            <option value="unique_a">Unique A</option>
            <option value="unique_b">Unique B</option>
            <option value="unique_both">Unique both</option>
            <option value="diverse_both">Diverse both</option>
          </select>
          <div class="help-popover" id="helpMatchView">Raw can repeat the same sentence many times. Unique A lets each Corpus A sentence appear once. Unique B lets each Corpus B sentence appear once. Unique both is one-to-one. Diverse both also suppresses nearby sentence clumps on both sides.</div>
        </label>
        <div class="metric-buttons">
          <button class="active" data-overview="max_score">Overview: max</button>
          <button data-overview="mean_best_a_to_b">Overview: best A→B</button>
          <button data-overview="mean_best_b_to_a">Overview: best B→A</button>
          <button data-overview="p95_score">Overview: p95</button>
        </div>
      </div>
      <h2>Corpus Heatmap <button class="info-btn" type="button" data-help="helpOverviewHeatmap" aria-label="Explain corpus heatmap">i</button></h2>
      <div class="help-popover" id="helpOverviewHeatmap">The overview heatmap recolors the same 16 x 22 document grid using the selected metric. It does not change the underlying texts or matrices; it changes which summary statistic is used to color each document-pair cell.</div>
      <div class="method-note" id="overviewMetricNote"></div>
      <div class="canvas-wrap"><canvas id="overviewCanvas" width="360" height="260"></canvas></div>
      <p class="note">Rows are Corpus A docs, columns are Corpus B docs. Click a cell or a pair row to inspect details.</p>
      <h2>Pairs</h2>
      <div class="pair-list" id="pairList"></div>
    </aside>
    <section>
      <div class="grid">
        <div class="panel" id="pairHeader"></div>
        <div class="two">
          <div class="panel">
            <h2>Selected Pair Heatmap</h2>
            <div class="canvas-wrap"><canvas id="pairCanvas" width="620" height="420"></canvas></div>
            <p class="note">This is a downsampled view of the full sentence-by-sentence matrix. Full `.npy` paths remain in the pair artifacts.</p>
          </div>
          <div class="panel">
            <h2>Selected Pair Metrics <button class="info-btn" type="button" data-help="helpPairMetrics" aria-label="Explain selected pair metrics">i</button></h2>
            <div class="help-popover" id="helpPairMetrics">These summarize the entire sentence-by-sentence matrix for the selected pair. Directional metrics are not claims about historical influence; they show coverage asymmetry between the two texts.</div>
            <div id="pairMetrics"></div>
            <p class="note">Use these as triage signals, not as final philological claims. Strong candidates still need close reading.</p>
          </div>
        </div>
        <div class="panel">
          <h2>Whole-Text Stream Profile <button class="info-btn" type="button" data-help="helpStream" aria-label="Explain whole-text stream profile">i</button></h2>
          <div class="help-popover" id="helpStream">This treats each text as an ordered stream. Best 1 is sensitive to the strongest local contact. Top 3 and Top 5 average ask whether each sentence has several good counterparts, making the profile more conservative.</div>
          <div class="method-note">
            <p><b>Method:</b> for every sentence in one text, the report finds its strongest sentence-level embedding matches anywhere in the other text, then plots the selected aggregation in the original sentence order. The upper line is Corpus A to Corpus B coverage; the lower line is Corpus B to Corpus A coverage.</p>
            <p><b>What to read from it:</b> Best 1 highlights the strongest local contact. Top 3 average and Top 5 average are more conservative because they require several good counterparts, not just one spike. Sustained elevated regions suggest broader passage-level coverage. This is a directional nearest-neighbor coverage profile derived from the full similarity matrix, related to accepted similarity-matrix, alignment, and text-reuse workflows. It is exploratory evidence for close reading, not proof of borrowing or historical direction by itself.</p>
          </div>
          <label><span class="label-row"><span>Stream aggregation</span><button class="info-btn" type="button" data-help="helpStreamMode" aria-label="Explain stream aggregation">i</button></span>
            <select id="streamMode">
              <option value="best_1">Best 1</option>
              <option value="top_3_avg">Top 3 average</option>
              <option value="top_5_avg">Top 5 average</option>
            </select>
            <div class="help-popover" id="helpStreamMode">Best 1 asks whether each sentence has at least one strong counterpart. Top 3/5 average asks whether the sentence sits in a broader shared semantic neighborhood.</div>
          </label>
          <div class="canvas-wrap"><canvas id="streamCanvas" width="620" height="220"></canvas></div>
        </div>
        <div class="panel">
          <h2>Top Matches</h2>
          <div class="matches" id="matches"></div>
        </div>
        <div class="panel">
          <h2>Documents</h2>
          <div class="two">
            <div><h3>Corpus A</h3><div id="docsA"></div></div>
            <div><h3>Corpus B</h3><div id="docsB"></div></div>
          </div>
        </div>
        <div class="panel">
          <h2>Colophon</h2>
          <div id="colophon"></div>
          <canvas id="gpuCanvas" width="620" height="180"></canvas>
        </div>
      </div>
    </section>
  </main>
  <script>
    const DATA = window.CORPUS_REPORT_DATA;
    let selectedPair = DATA.pairs[0];
    let overviewMetric = 'max_score';
    let matchMode = 'raw';
    let streamMode = 'best_1';
    let selectedMatchKey = null;
    const contextRadius = 3;

    const fmt = new Intl.NumberFormat('en-US');
    const score = (x) => Number(x).toFixed(3);
    const $ = (id) => document.getElementById(id);
    const overviewNotes = {
      max_score: {
        title: 'Overview: max',
        body: 'Each cell is colored by the single strongest sentence-pair score between the two documents. This is sensitive to local reuse, formulaic overlap, or one striking passage, but it can overstate a relationship if the rest of the matrix is weak.'
      },
      mean_best_a_to_b: {
        title: 'Overview: best A to B',
        body: 'For each Corpus A sentence, find its best match anywhere in the Corpus B document, then average those best scores. High values mean much of Corpus A finds plausible coverage in Corpus B. This is directional and should not be read as historical influence by itself.'
      },
      mean_best_b_to_a: {
        title: 'Overview: best B to A',
        body: 'For each Corpus B sentence, find its best match anywhere in the Corpus A document, then average those best scores. High values often show that a shorter Corpus B text is well covered by Corpus A, or that the two share a concentrated semantic field.'
      },
      p95_score: {
        title: 'Overview: p95',
        body: 'Each cell is colored by the 95th percentile of all sentence-pair scores in the matrix. This ignores most ordinary cells and asks whether the upper tail is broadly elevated. It is usually better than max for finding sustained document-pair similarity.'
      }
    };

    function colorFor(v, min, max) {
      const t = max === min ? 0 : Math.max(0, Math.min(1, (v - min) / (max - min)));
      const stops = [
        [248, 248, 240],
        [223, 184, 91],
        [164, 66, 42],
        [36, 92, 99]
      ];
      const p = t * (stops.length - 1);
      const i = Math.min(stops.length - 2, Math.floor(p));
      const f = p - i;
      const c = stops[i].map((x, k) => Math.round(x + (stops[i + 1][k] - x) * f));
      return `rgb(${c[0]},${c[1]},${c[2]})`;
    }

    function init() {
      $('runSubtitle').textContent = `${DATA.run.doc_count_a} Corpus A files x ${DATA.run.doc_count_b} Corpus B files · ${fmt.format(DATA.run.pair_count)} pairwise matrices · ${fmt.format(DATA.run.total_sentences)} sentence embeddings`;
      renderStats();
      renderColophon();
      renderDocs();
      renderFileBrowser();
      bindControls();
      renderPairList();
      renderOverview();
      selectPair(DATA.pairs[0].pair_id);
    }

    function renderStats() {
      const stats = [
        [DATA.run.pair_count, 'document pairs'],
        [`${DATA.run.doc_count_a} x ${DATA.run.doc_count_b}`, 'corpus shape'],
        [DATA.run.total_sentences, 'sentence embeddings'],
        [DATA.run.embedding_dim, 'embedding dimensions'],
        [DATA.pairs.length, 'saved matrices'],
        ['5', 'match views']
      ];
      $('stats').innerHTML = stats.map(([v, label]) => `<div class="stat"><b>${formatValue(v)}</b><span>${label}</span></div>`).join('');
    }

    function renderColophon() {
      const rssGb = DATA.run.max_resident_set_kb ? (DATA.run.max_resident_set_kb / 1024 / 1024).toFixed(1) + ' GB' : 'n/a';
      $('colophon').innerHTML = `
        <table>
          <tr><th>Model</th><td>${escapeHtml(DATA.run.model_id)}</td></tr>
          <tr><th>Device</th><td>${DATA.run.device}, batch ${DATA.run.batch_size}</td></tr>
          <tr><th>Wall time</th><td>${DATA.run.wall_time || 'n/a'}</td></tr>
          <tr><th>Peak GPU util</th><td>${DATA.run.max_gpu_util_pct}%</td></tr>
          <tr><th>Peak GPU memory</th><td>${DATA.run.max_gpu_mem_mib} MiB</td></tr>
          <tr><th>Peak host RSS</th><td>${rssGb}</td></tr>
        </table>`;
      drawGpu();
    }

    function bindControls() {
      $('search').addEventListener('input', () => {
        renderFileBrowser();
        renderPairList();
      });
      $('clearSearch').addEventListener('click', () => {
        $('search').value = '';
        renderFileBrowser();
        renderPairList();
      });
      $('sortMetric').addEventListener('change', renderPairList);
      $('topK').addEventListener('input', renderMatches);
      $('matchMode').addEventListener('change', () => {
        matchMode = $('matchMode').value;
        renderMatches();
      });
      $('streamMode').addEventListener('change', () => {
        streamMode = $('streamMode').value;
        renderStreamProfile();
      });
      document.querySelectorAll('[data-overview]').forEach(btn => {
        btn.addEventListener('click', () => {
          document.querySelectorAll('[data-overview]').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          overviewMetric = btn.dataset.overview;
          renderOverview();
        });
      });
      document.querySelectorAll('[data-help]').forEach(btn => {
        btn.addEventListener('click', () => {
          $(btn.dataset.help).classList.toggle('open');
        });
      });
    }

    function filteredPairs() {
      const q = $('search').value.trim().toLowerCase();
      const metric = $('sortMetric').value;
      return DATA.pairs
        .filter(p => !q || [p.pair_id, p.doc_a_name, p.doc_b_name, p.doc_a_relative_path, p.doc_b_relative_path].join(' ').toLowerCase().includes(q))
        .sort((a, b) => Number(b[metric]) - Number(a[metric]));
    }

    function allDocs() {
      return [
        ...DATA.documentsA.map(d => ({...d, corpus: 'Corpus A'})),
        ...DATA.documentsB.map(d => ({...d, corpus: 'Corpus B'}))
      ];
    }

    function renderFileBrowser() {
      const docs = allDocs();
      $('fileOptions').innerHTML = docs.map(d => `<option value="${escapeHtml(d.doc_id)}">${escapeHtml(d.corpus + ' · ' + d.relative_path)}</option>`).join('');
      const q = $('search').value.trim().toLowerCase();
      const rows = docs
        .filter(d => !q || [d.doc_id, d.relative_path, d.short_name, d.corpus].join(' ').toLowerCase().includes(q))
        .slice(0, 38);
      $('fileBrowser').innerHTML = rows.map(d => `
        <button class="file-option" type="button" data-file-query="${escapeHtml(d.doc_id)}">
          <b>${escapeHtml(d.doc_id)} · ${escapeHtml(d.corpus)}</b>
          <span>${escapeHtml(d.relative_path)} · ${fmt.format(d.sentence_count)} sentences</span>
        </button>`).join('');
      document.querySelectorAll('[data-file-query]').forEach(el => {
        el.addEventListener('click', () => {
          $('search').value = el.dataset.fileQuery;
          renderFileBrowser();
          renderPairList();
        });
      });
    }

    function renderPairList() {
      const rows = filteredPairs().slice(0, 80);
      $('pairList').innerHTML = rows.map(p => `
        <div class="pair-row ${p.pair_id === selectedPair?.pair_id ? 'active' : ''}" data-pair="${p.pair_id}">
          <div class="pair-title">${p.pair_id}: ${escapeHtml(p.doc_a_name)} ↔ ${escapeHtml(p.doc_b_name)}</div>
          <div class="pair-meta"><span>max ${score(p.max_score)}</span><span>p95 ${score(p.p95_score)}</span><span>${fmt.format(p.score_count)} cells</span></div>
        </div>`).join('');
      document.querySelectorAll('[data-pair]').forEach(el => el.addEventListener('click', () => selectPair(el.dataset.pair)));
    }

    function selectPair(pairId) {
      selectedPair = DATA.pairs.find(p => p.pair_id === pairId) || DATA.pairs[0];
      selectedMatchKey = null;
      renderPairList();
      renderPairHeader();
      renderPairMetrics();
      renderPairHeatmap();
      renderStreamProfile();
      renderMatches();
    }

    function renderPairHeader() {
      const p = selectedPair;
      $('pairHeader').innerHTML = `
        <h2>${p.pair_id}</h2>
        <div class="two">
          <div><h3>${p.doc_a_id}</h3><div>${escapeHtml(p.doc_a_relative_path)}</div><p class="note">${fmt.format(p.sentence_count_a)} sentences</p></div>
          <div><h3>${p.doc_b_id}</h3><div>${escapeHtml(p.doc_b_relative_path)}</div><p class="note">${fmt.format(p.sentence_count_b)} sentences</p></div>
        </div>
        <table>
          <tr><th>Max</th><th>P95</th><th>Mean</th><th>Mean best A→B</th><th>Mean best B→A</th><th>Matrix cells</th></tr>
          <tr><td>${score(p.max_score)}</td><td>${score(p.p95_score)}</td><td>${score(p.mean_score)}</td><td>${score(p.mean_best_a_to_b)}</td><td>${score(p.mean_best_b_to_a)}</td><td>${fmt.format(p.score_count)}</td></tr>
        </table>`;
    }

    function renderPairMetrics() {
      const p = selectedPair;
      const asymmetry = Number(p.mean_best_b_to_a) - Number(p.mean_best_a_to_b);
      const direction = Math.abs(asymmetry) < 0.025
        ? 'balanced coverage'
        : asymmetry > 0
          ? 'Corpus B sentences find stronger homes in Corpus A'
          : 'Corpus A sentences find stronger homes in Corpus B';
      $('pairMetrics').innerHTML = `
        <table>
          <tr><th>Max score</th><td>${score(p.max_score)}<br><span class="note">strongest single sentence-pair hit</span></td></tr>
          <tr><th>P95 score</th><td>${score(p.p95_score)}<br><span class="note">upper-tail breadth across the matrix</span></td></tr>
          <tr><th>Mean best A to B</th><td>${score(p.mean_best_a_to_b)}</td></tr>
          <tr><th>Mean best B to A</th><td>${score(p.mean_best_b_to_a)}</td></tr>
          <tr><th>Coverage pattern</th><td>${escapeHtml(direction)} (${asymmetry >= 0 ? '+' : ''}${asymmetry.toFixed(3)})</td></tr>
          <tr><th>Matrix cells</th><td>${fmt.format(p.score_count)}</td></tr>
        </table>`;
    }

    function renderOverview() {
      renderOverviewMetricNote();
      const canvas = $('overviewCanvas');
      const ctx = canvas.getContext('2d');
      const margin = { left: 48, right: 12, top: 18, bottom: 42 };
      const w = canvas.width - margin.left - margin.right;
      const h = canvas.height - margin.top - margin.bottom;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const values = DATA.pairs.map(p => Number(p[overviewMetric]));
      const min = Math.min(...values), max = Math.max(...values);
      const cw = w / DATA.documentsB.length;
      const ch = h / DATA.documentsA.length;
      DATA.pairs.forEach(p => {
        const ai = DATA.documentsA.findIndex(d => d.doc_id === p.doc_a_id);
        const bi = DATA.documentsB.findIndex(d => d.doc_id === p.doc_b_id);
        ctx.fillStyle = colorFor(Number(p[overviewMetric]), min, max);
        ctx.fillRect(margin.left + bi * cw, margin.top + ai * ch, Math.ceil(cw), Math.ceil(ch));
        if (p.pair_id === selectedPair?.pair_id) {
          ctx.strokeStyle = '#111';
          ctx.lineWidth = 2;
          ctx.strokeRect(margin.left + bi * cw + 1, margin.top + ai * ch + 1, cw - 2, ch - 2);
        }
      });
      ctx.fillStyle = '#333';
      ctx.font = '11px system-ui';
      DATA.documentsA.forEach((d, i) => ctx.fillText(d.doc_id, 8, margin.top + i * ch + ch * .68));
      DATA.documentsB.forEach((d, i) => {
        ctx.save();
        ctx.translate(margin.left + i * cw + cw * .55, canvas.height - 8);
        ctx.rotate(-Math.PI / 3);
        ctx.fillText(d.doc_id, 0, 0);
        ctx.restore();
      });
      canvas.onclick = (event) => {
        const rect = canvas.getBoundingClientRect();
        const x = (event.clientX - rect.left) * (canvas.width / rect.width) - margin.left;
        const y = (event.clientY - rect.top) * (canvas.height / rect.height) - margin.top;
        const bi = Math.floor(x / cw), ai = Math.floor(y / ch);
        if (ai >= 0 && ai < DATA.documentsA.length && bi >= 0 && bi < DATA.documentsB.length) {
          selectPair(`${DATA.documentsA[ai].doc_id}__${DATA.documentsB[bi].doc_id}`);
        }
      };
    }

    function renderOverviewMetricNote() {
      const note = overviewNotes[overviewMetric];
      $('overviewMetricNote').innerHTML = `<p><b>${note.title}:</b> ${escapeHtml(note.body)}</p>`;
    }

    function renderPairHeatmap() {
      const canvas = $('pairCanvas');
      const ctx = canvas.getContext('2d');
      const hm = selectedPair.heatmap;
      const rows = hm.length, cols = hm[0].length;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const flat = hm.flat();
      const min = Math.min(...flat), max = Math.max(...flat);
      const margin = { left: 58, right: 18, top: 18, bottom: 42 };
      const w = canvas.width - margin.left - margin.right;
      const h = canvas.height - margin.top - margin.bottom;
      const cw = w / cols, ch = h / rows;
      hm.forEach((row, r) => row.forEach((v, c) => {
        ctx.fillStyle = colorFor(v, min, max);
        ctx.fillRect(margin.left + c * cw, margin.top + r * ch, Math.ceil(cw), Math.ceil(ch));
      }));
      ctx.fillStyle = '#333';
      ctx.font = '12px system-ui';
      ctx.fillText(`A sentences (${selectedPair.sentence_count_a})`, margin.left, canvas.height - 12);
      ctx.save();
      ctx.translate(16, margin.top + h);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText(`B sentences (${selectedPair.sentence_count_b})`, 0, 0);
      ctx.restore();
      ctx.fillText(`mean ${score(selectedPair.mean_score)} · max ${score(selectedPair.max_score)}`, margin.left, 14);
    }

    function renderStreamProfile() {
      const canvas = $('streamCanvas');
      const ctx = canvas.getContext('2d');
      const profile = selectedPair.stream_profiles?.[streamMode] || selectedPair.stream_profiles.best_1;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const label = streamMode === 'best_1' ? 'Best 1' : streamMode === 'top_3_avg' ? 'Top 3 average' : 'Top 5 average';
      drawProfileLine(ctx, profile.a_to_b, 30, 70, '#245c63', `${label}: A to B coverage across Corpus A order`);
      drawProfileLine(ctx, profile.b_to_a, 132, 70, '#a6422a', `${label}: B to A coverage across Corpus B order`);
    }

    function drawProfileLine(ctx, values, top, height, color, label) {
      const left = 46, right = 18;
      const width = ctx.canvas.width - left - right;
      const min = Math.min(...values), max = Math.max(...values);
      ctx.strokeStyle = '#d8d8d8';
      ctx.lineWidth = 1;
      ctx.strokeRect(left, top, width, height);
      ctx.fillStyle = '#666';
      ctx.font = '12px system-ui';
      ctx.fillText(label, left, top - 9);
      ctx.fillText(max.toFixed(2), 8, top + 10);
      ctx.fillText(min.toFixed(2), 8, top + height);
      ctx.beginPath();
      values.forEach((value, index) => {
        const x = left + (values.length === 1 ? 0 : index * width / (values.length - 1));
        const t = max === min ? .5 : (value - min) / (max - min);
        const y = top + height - t * height;
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    function renderMatches() {
      const k = Math.max(1, Math.min(100, Number($('topK').value) || 20));
      const rows = (selectedPair.topk_by_mode?.[matchMode] || selectedPair.topk).slice(0, k);
      const modeNotes = {
        raw: 'Raw top-k allows repeated A and B sentence indices.',
        unique_a: 'Unique A keeps each A sentence at most once.',
        unique_b: 'Unique B keeps each B sentence at most once.',
        unique_both: 'Unique both is greedy one-to-one matching.',
        diverse_both: 'Diverse both is one-to-one and suppresses neighboring sentence clumps.'
      };
      $('matches').innerHTML = `<p class="note">${modeNotes[matchMode]} Showing ${rows.length} rows for ${selectedPair.pair_id}.</p>` + rows.map(m => `
        <div class="match ${matchKey(m) === selectedMatchKey ? 'active' : ''}" data-match-key="${escapeHtml(matchKey(m))}">
          <div class="match-head">
            <span>#${m.rank} · score ${score(m.score)} · A[${m.i}] ↔ B[${m.j}]</span>
            <span>click for ±3 context</span>
          </div>
          <div class="sentences">
            <div class="sentence">${escapeHtml(m.sentence_a)}</div>
            <div class="sentence">${escapeHtml(m.sentence_b)}</div>
          </div>
          ${matchKey(m) === selectedMatchKey ? renderMatchContext(m) : ''}
        </div>`).join('');
      document.querySelectorAll('[data-match-key]').forEach(el => {
        el.addEventListener('click', () => {
          selectedMatchKey = selectedMatchKey === el.dataset.matchKey ? null : el.dataset.matchKey;
          renderMatches();
        });
      });
    }

    function matchKey(match) {
      return `${matchMode}:${match.rank}:${match.i}:${match.j}`;
    }

    function renderMatchContext(match) {
      const docA = DATA.documentsA.find(doc => doc.doc_id === selectedPair.doc_a_id);
      const docB = DATA.documentsB.find(doc => doc.doc_id === selectedPair.doc_b_id);
      return `<div class="context-panel">
        ${renderContextColumn('Corpus A surroundings', selectedPair.doc_a_id, docA?.short_name || selectedPair.doc_a_name, docA?.sentences || [], match.i)}
        ${renderContextColumn('Corpus B surroundings', selectedPair.doc_b_id, docB?.short_name || selectedPair.doc_b_name, docB?.sentences || [], match.j)}
      </div>`;
    }

    function renderContextColumn(title, docId, docName, sentences, centerIndex) {
      const start = Math.max(0, centerIndex - contextRadius);
      const end = Math.min(sentences.length - 1, centerIndex + contextRadius);
      const rows = [];
      for (let index = start; index <= end; index++) {
        rows.push(`<div class="context-row ${index === centerIndex ? 'hit' : ''}">
          <div class="context-index">[${index}]</div>
          <div>${escapeHtml(sentences[index] || '')}</div>
        </div>`);
      }
      return `<div class="context-col">
        <div class="context-title">${escapeHtml(title)} · ${escapeHtml(docId)} · ${escapeHtml(docName)}</div>
        ${rows.join('')}
      </div>`;
    }

    function renderDocs() {
      $('docsA').innerHTML = docsTable(DATA.documentsA);
      $('docsB').innerHTML = docsTable(DATA.documentsB);
    }

    function docsTable(rows) {
      const maxSent = Math.max(...rows.map(r => r.sentence_count));
      return `<table><tr><th>ID</th><th>File</th><th>Sentences</th></tr>${rows.map(r => `
        <tr><td>${r.doc_id}</td><td>${escapeHtml(r.short_name)}<div class="note">${escapeHtml(r.embeddings_npy)}</div></td><td>${fmt.format(r.sentence_count)}<div class="bar"><span style="width:${100 * r.sentence_count / maxSent}%"></span></div></td></tr>
      `).join('')}</table>`;
    }

    function drawGpu() {
      const canvas = $('gpuCanvas');
      const ctx = canvas.getContext('2d');
      const rows = DATA.gpu;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (!rows.length) return;
      const pad = 28, w = canvas.width - 2 * pad, h = canvas.height - 2 * pad;
      ctx.strokeStyle = '#d8d8d8';
      ctx.strokeRect(pad, pad, w, h);
      drawLine(rows.map(r => r.gpu_util_pct), 100, '#a6422a');
      drawLine(rows.map(r => r.mem_used_mib), rows[0].mem_total_mib, '#245c63');
      ctx.fillStyle = '#333';
      ctx.font = '12px system-ui';
      ctx.fillText('GPU util', pad, 16);
      ctx.fillStyle = '#245c63';
      ctx.fillText('memory', pad + 70, 16);
      function drawLine(values, max, color) {
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        values.forEach((v, i) => {
          const x = pad + (i / Math.max(1, values.length - 1)) * w;
          const y = pad + h - (v / max) * h;
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();
      }
    }

    function escapeHtml(s) {
      return String(s).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[ch]));
    }

    function formatValue(v) {
      return typeof v === 'number' && Number.isFinite(v) ? fmt.format(v) : escapeHtml(v);
    }

    init();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
