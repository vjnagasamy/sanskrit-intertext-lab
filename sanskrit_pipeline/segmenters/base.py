"""Shared sentence segmenter interfaces for Sanskrit text."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

DANDA = "।"
DOUBLE_DANDA = "॥"

# Devanagari digit range U+0966–U+096F
DEVANAGARI_DIGITS = "".join(chr(c) for c in range(0x0966, 0x0970))

# Segments containing only digits, spaces, and daṇḍa marks are verse numbers (e.g. ॥ १ ॥)
_VERSE_NUMBER_RE = re.compile(rf"^[\s{DEVANAGARI_DIGITS}0-9।॥.]+$")

# Matches Devanagari letters/digits/marks but NOT daṇḍa punctuation (U+0964–U+0965)
_HAS_DEVANAGARI_RE = re.compile(r"[ऀ-ॣ०-ॿ]")


@dataclass(slots=True)
class Segment:
    """A segmented sentence span."""

    text: str
    start: int
    end: int


class BaseSegmenter(ABC):
    """Base contract for sentence segmentation engines."""

    engine_name = "base"

    # When True, callers must normalize with ``preserve_lines=True`` so the
    # engine can see physical line breaks (newlines) in the input.
    requires_line_structure = False

    # When True, callers must skip IAST→Devanagari transliteration so the engine
    # segments the text in its original (romanized) script.
    keep_source_script = False

    @abstractmethod
    def segment(self, text: str) -> list[Segment]:
        """Return segmented spans for the provided text."""


def is_verse_number(text: str) -> bool:
    """Return True if the segment contains only digits, spaces, and punctuation."""
    return bool(_VERSE_NUMBER_RE.match(text))


def has_devanagari(text: str) -> bool:
    """Return True if the segment contains at least one Devanagari character."""
    return bool(_HAS_DEVANAGARI_RE.search(text))
