"""Hybrid daṇḍa + prose segmentation for mixed verse/prose texts."""

from __future__ import annotations

from .base import BaseSegmenter, Segment
from .dandas import DandasSegmenter
from .prose import ProseSegmenter, _window_segments


class HybridSegmenter(BaseSegmenter):
    """Use daṇḍa segmentation, falling back to prose windows for long chunks."""

    engine_name = "hybrid"

    def __init__(
        self,
        max_tokens: int = 400,
        overlap_tokens: int = 50,
        chars_per_token: float = 3.5,
        split_on_single_danda: bool = False,
        source_format: str = "devanagari",
    ) -> None:
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.chars_per_token = chars_per_token
        self.split_on_single_danda = split_on_single_danda
        self.source_format = source_format
        self._dandas = DandasSegmenter(split_on_single_danda=split_on_single_danda)
        self._max_chars = max(1, int(max_tokens * chars_per_token))
        self._overlap_chars = max(0, int(overlap_tokens * chars_per_token))

    def segment(self, text: str) -> list[Segment]:
        if not text.strip():
            return []

        danda_segments = self._dandas.segment(text)
        if not danda_segments:
            return ProseSegmenter(
                max_tokens=self.max_tokens,
                overlap_tokens=self.overlap_tokens,
                chars_per_token=self.chars_per_token,
                source_format=self.source_format,
            ).segment(text)

        output: list[Segment] = []
        for segment in danda_segments:
            estimated_tokens = len(segment.text) / self.chars_per_token
            if estimated_tokens <= self.max_tokens:
                output.append(segment)
            else:
                output.extend(
                    _window_segments(segment.text, segment.start, self._max_chars, self._overlap_chars)
                )
        return output
