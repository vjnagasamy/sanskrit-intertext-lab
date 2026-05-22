"""Stanza-backed Sanskrit sentence segmentation engine."""

from __future__ import annotations

try:
    import stanza as _stanza
    _STANZA_AVAILABLE = True
except ImportError:
    _stanza = None  # type: ignore[assignment]
    _STANZA_AVAILABLE = False

from .base import BaseSegmenter, Segment


def stanza_available() -> bool:
    """Return True if stanza is installed and importable."""
    return _STANZA_AVAILABLE


class StanzaSegmenter(BaseSegmenter):
    """ML-based Sanskrit segmentation using the Stanza Sanskrit model.

    Requires stanza and the Sanskrit model:

        pip install stanza
        python -c "import stanza; stanza.download('sa')"

    Or use the helper script:

        python scripts/download_stanza_sanskrit.py
    """

    engine_name = "stanza"

    def __init__(self) -> None:
        if not _STANZA_AVAILABLE:
            raise ImportError(
                "stanza is required for the 'stanza' segmentation engine.\n"
                "Install it with: pip install stanza\n"
                "Then download the Sanskrit model: python scripts/download_stanza_sanskrit.py"
            )
        self._pipeline = _stanza.Pipeline(lang="sa", processors="tokenize", verbose=False)

    def segment(self, text: str) -> list[Segment]:
        if not text.strip():
            return []

        doc = self._pipeline(text)
        segments: list[Segment] = []
        for sentence in doc.sentences:
            sentence_text = sentence.text.strip()
            if not sentence_text:
                continue
            start = sentence.tokens[0].start_char
            end = sentence.tokens[-1].end_char
            segments.append(Segment(sentence_text, start, end))
        return segments
