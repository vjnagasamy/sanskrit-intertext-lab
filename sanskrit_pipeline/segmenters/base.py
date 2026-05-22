"""Shared sentence segmenter interfaces for Sanskrit text."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

DANDA = "।"
DOUBLE_DANDA = "॥"

# Devanagari digit range U+0966–U+096F
DEVANAGARI_DIGITS = "".join(chr(c) for c in range(0x0966, 0x0970))


@dataclass(slots=True)
class Segment:
    """A segmented sentence span."""

    text: str
    start: int
    end: int


class BaseSegmenter(ABC):
    """Base contract for sentence segmentation engines."""

    engine_name = "base"

    @abstractmethod
    def segment(self, text: str) -> list[Segment]:
        """Return segmented spans for the provided text."""
