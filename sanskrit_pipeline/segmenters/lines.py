"""Deterministic line-based Sanskrit segmentation engine.

Designed for verse editions where each physical line is a pāda / half-verse,
including IAST sources that carry no daṇḍa punctuation. One non-empty line
becomes one segment. Blank lines and ``%`` comment lines are dropped, and
leading verse-number labels (``1.1ab:``, ``1.2``) are stripped.
"""

from __future__ import annotations

import re

from .base import BaseSegmenter, Segment, has_devanagari

_COMMENT_PREFIX = "%"

# Leading verse label terminated by a colon, e.g. "1.1ab:" or transliterated
# "१.१अब्:". The label must contain at least one digit and no spaces, and the
# colon must fall within a short prefix so mid-line colons are never stripped.
_LABEL_COLON_RE = re.compile(r"^(?=[^\s:]{1,16}:)[^\s:]*\d[^\s:]*:[ \t]*")

# Leading numeric label with no colon, e.g. "1.2 " or transliterated "१.२ ".
_LABEL_SPACE_RE = re.compile(r"^\d[\d.\u2013\u2014\-]{0,14}[ \t]+")


class LineSegmenter(BaseSegmenter):
    """Split normalized text into one segment per non-empty line."""

    engine_name = "lines"
    requires_line_structure = True

    def __init__(
        self,
        *,
        strip_labels: bool = True,
        drop_comments: bool = True,
        require_devanagari: bool = False,
    ) -> None:
        self.strip_labels = strip_labels
        self.drop_comments = drop_comments
        self.require_devanagari = require_devanagari

    def segment(self, text: str) -> list[Segment]:
        if not text.strip():
            return []

        segments: list[Segment] = []
        offset = 0
        for raw_line in text.split("\n"):
            line_start = offset
            offset += len(raw_line) + 1  # account for the split "\n"

            stripped = raw_line.strip()
            if not stripped:
                continue
            if self.drop_comments and stripped.startswith(_COMMENT_PREFIX):
                continue

            content = _strip_label(stripped) if self.strip_labels else stripped
            content = content.strip()
            if not content:
                continue
            if self.require_devanagari and not has_devanagari(content):
                continue

            content_offset = raw_line.find(content)
            if content_offset < 0:
                start, end = line_start, line_start + len(raw_line)
            else:
                start = line_start + content_offset
                end = start + len(content)
            segments.append(Segment(content, start, end))
        return segments


def _strip_label(text: str) -> str:
    """Remove a leading verse-number label if present (colon or space form)."""
    match = _LABEL_COLON_RE.match(text)
    if match:
        return text[match.end():]
    match = _LABEL_SPACE_RE.match(text)
    if match:
        return text[match.end():]
    return text
