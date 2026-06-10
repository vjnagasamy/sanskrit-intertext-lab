"""Regex-based Sanskrit sentence segmentation engine using daṇḍa punctuation."""

from __future__ import annotations

import re

from .base import (
    DANDA,
    DOUBLE_DANDA,
    BaseSegmenter,
    Segment,
    has_devanagari,
    is_verse_number,
)


class DandasSegmenter(BaseSegmenter):
    """Fast regex segmentation splitting on Sanskrit daṇḍa punctuation.

    Default: split only on ``॥`` (double daṇḍa), which marks full-verse boundaries
    in most Sanskrit texts. Set ``split_on_single_danda=True`` for half-verse
    (pāda-level) granularity.

    Verse-number segments (e.g. ``॥ १ ॥``, ``॥ 1.2 ॥``) are automatically dropped.
    Segments with no Devanagari characters are also dropped.
    """

    engine_name = "dandas"

    def __init__(self, split_on_single_danda: bool = False) -> None:
        self.split_on_single_danda = split_on_single_danda
        if split_on_single_danda:
            self._split_pattern = re.compile(rf"([{DANDA}{DOUBLE_DANDA}]+)")
        else:
            self._split_pattern = re.compile(rf"([{DOUBLE_DANDA}]+)")

    def segment(self, text: str) -> list[Segment]:
        if not text.strip():
            return []

        parts = self._split_pattern.split(text)
        segments: list[Segment] = []
        current_parts: list[str] = []
        cursor = 0
        buffer_start = 0

        i = 0
        while i < len(parts):
            part = parts[i]
            if not part:
                i += 1
                continue

            part_len = len(part)
            is_delimiter = DOUBLE_DANDA in part or (self.split_on_single_danda and DANDA in part)

            if is_delimiter:
                current_parts.append(part)
                segment_text = "".join(current_parts).strip()
                if segment_text and has_devanagari(segment_text) and not is_verse_number(segment_text):
                    segments.append(Segment(segment_text, buffer_start, cursor + part_len))
                current_parts = []
                buffer_start = cursor + part_len
            else:
                if not current_parts:
                    buffer_start = cursor
                current_parts.append(part)

            cursor += part_len
            i += 1

        # Trailing text without a final delimiter
        if current_parts:
            tail = "".join(current_parts).strip()
            if tail and has_devanagari(tail) and not is_verse_number(tail):
                segments.append(Segment(tail, buffer_start, len(text)))

        return segments
