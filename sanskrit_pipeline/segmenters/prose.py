"""Prose-aware Sanskrit segmentation with sliding-window chunking."""

from __future__ import annotations

import re

from ..normalization import normalize_text
from .base import DANDA, DOUBLE_DANDA, BaseSegmenter, Segment
from .dandas import DandasSegmenter, _has_devanagari, _is_verse_number

_HAS_DEVANAGARI_RE = re.compile(r"[ऀ-ॣ०-ॿ]")
_MIN_SEGMENT_CHARS = 10


class ProseSegmenter(BaseSegmenter):
    """Split long verse chunks into prose-friendly windows with overlap."""

    engine_name = "prose"

    def __init__(
        self,
        max_tokens: int = 400,
        overlap_tokens: int = 50,
        chars_per_token: float = 3.5,
        source_format: str = "devanagari",
    ) -> None:
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.chars_per_token = chars_per_token
        self.source_format = source_format
        self._max_chars = max(1, int(max_tokens * chars_per_token))
        self._overlap_chars = max(0, int(overlap_tokens * chars_per_token))
        self._dandas = DandasSegmenter(split_on_single_danda=False)

    def segment(self, text: str) -> list[Segment]:
        normalized = normalize_text(text, source_format=self.source_format)
        if not normalized.strip():
            return []

        verse_chunks = self._dandas.segment(normalized)
        if not verse_chunks:
            return _merge_short_segments(_window_segments(normalized, 0, self._max_chars, self._overlap_chars))

        segments: list[Segment] = []
        for chunk in verse_chunks:
            if self._estimated_tokens(chunk.text) <= self.max_tokens:
                segments.append(chunk)
            else:
                segments.extend(
                    _window_segments(chunk.text, chunk.start, self._max_chars, self._overlap_chars)
                )
        return _merge_short_segments(segments)

    def _estimated_tokens(self, text: str) -> float:
        return len(text) / self.chars_per_token


def _window_segments(text: str, base_offset: int, max_chars: int, overlap_chars: int) -> list[Segment]:
    clauses = _split_clauses(text)
    if not clauses:
        return _char_window_segments(text, base_offset, max_chars, overlap_chars)

    segments: list[Segment] = []
    current_parts: list[tuple[str, int, int]] = []
    current_len = 0

    def flush() -> None:
        nonlocal current_parts, current_len
        if not current_parts:
            return
        window_text = "".join(part[0] for part in current_parts).strip()
        if window_text and _has_devanagari(window_text) and not _is_verse_number(window_text):
            segments.append(
                Segment(
                    window_text,
                    base_offset + current_parts[0][1],
                    base_offset + current_parts[-1][2],
                )
            )
        current_parts = []
        current_len = 0

    for clause in clauses:
        clause_text, clause_start, clause_end = clause
        clause_len = len(clause_text)
        if clause_len > max_chars:
            flush()
            segments.extend(_char_window_segments(clause_text, base_offset + clause_start, max_chars, overlap_chars))
            continue
        if current_parts and current_len + clause_len > max_chars:
            previous_text = "".join(part[0] for part in current_parts)
            flush()
            if overlap_chars > 0 and previous_text:
                overlap_text = previous_text[-overlap_chars:]
                overlap_start = clause_start - len(overlap_text)
                current_parts = [(overlap_text, overlap_start, clause_start)]
                current_len = len(overlap_text)
        current_parts.append(clause)
        current_len += clause_len

    flush()
    return segments


def _split_clauses(text: str) -> list[tuple[str, int, int]]:
    pattern = re.compile(rf"([{DANDA}{DOUBLE_DANDA}]|\n+)")
    parts = pattern.split(text)
    clauses: list[tuple[str, int, int]] = []
    cursor = 0
    buffer: list[str] = []
    buffer_start = 0

    for part in parts:
        if not part:
            continue
        part_len = len(part)
        if part in {DANDA, DOUBLE_DANDA} or "\n" in part:
            buffer.append(part if part not in {DANDA, DOUBLE_DANDA} else part)
            clause_text = "".join(buffer).strip()
            if clause_text:
                clauses.append((clause_text, buffer_start, cursor + part_len))
            buffer = []
            buffer_start = cursor + part_len
        else:
            if not buffer:
                buffer_start = cursor
            buffer.append(part)
        cursor += part_len

    if buffer:
        tail = "".join(buffer).strip()
        if tail:
            clauses.append((tail, buffer_start, buffer_start + len(tail)))
    return clauses


def _char_window_segments(text: str, base_offset: int, max_chars: int, overlap_chars: int) -> list[Segment]:
    segments: list[Segment] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        piece = text[start:end].strip()
        if piece and _has_devanagari(piece) and not _is_verse_number(piece):
            segments.append(Segment(piece, base_offset + start, base_offset + end))
        if end >= len(text):
            break
        start = max(end - overlap_chars, start + 1)
    return segments


def _merge_short_segments(segments: list[Segment]) -> list[Segment]:
    if not segments:
        return []

    merged: list[Segment] = []
    for segment in segments:
        devanagari_count = len(_HAS_DEVANAGARI_RE.findall(segment.text))
        if devanagari_count < _MIN_SEGMENT_CHARS and merged:
            previous = merged[-1]
            merged[-1] = Segment(
                f"{previous.text} {segment.text}".strip(),
                previous.start,
                segment.end,
            )
        else:
            merged.append(segment)
    return merged
