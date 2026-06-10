"""Stanza-backed Sanskrit sentence segmentation engine."""

from __future__ import annotations

try:
    import stanza as _stanza
    _STANZA_AVAILABLE = True
except ImportError:
    _stanza = None  # type: ignore[assignment]
    _STANZA_AVAILABLE = False

from .base import BaseSegmenter, Segment, has_devanagari, is_verse_number


def stanza_available() -> bool:
    """Return True if stanza is installed and importable."""
    return _STANZA_AVAILABLE


def _resolve_download_method(name: str | None):
    """Map a friendly name to a stanza DownloadMethod, tolerating old versions."""
    if name is None:
        return None
    download_method = getattr(_stanza, "DownloadMethod", None)
    if download_method is None:
        return None
    mapping = {
        "reuse": "REUSE_RESOURCES",
        "download": "DOWNLOAD_RESOURCES",
        "none": None,
    }
    member = mapping.get(name.lower(), "REUSE_RESOURCES")
    if member is None:
        return None
    return getattr(download_method, member, None)


class StanzaSegmenter(BaseSegmenter):
    """ML-based Sanskrit segmentation using the Stanza Sanskrit model.

    Requires stanza and the Sanskrit model:

        pip install stanza
        python -c "import stanza; stanza.download('sa')"

    Or use the helper script:

        python scripts/download_stanza_sanskrit.py

    By default the pipeline reuses already-downloaded resources (no network
    check on each construction) and drops verse-number-only segments so output
    is comparable to the daṇḍa engine.
    """

    engine_name = "stanza"

    def __init__(
        self,
        *,
        use_gpu: bool | None = None,
        download_method: str | None = "reuse",
        drop_verse_numbers: bool = True,
        require_devanagari: bool = True,
        keep_source_script: bool = False,
    ) -> None:
        if not _STANZA_AVAILABLE:
            raise ImportError(
                "stanza is required for the 'stanza' segmentation engine.\n"
                "Install it with: pip install stanza\n"
                "Then download the Sanskrit model: python scripts/download_stanza_sanskrit.py"
            )
        self.drop_verse_numbers = drop_verse_numbers
        self.keep_source_script = keep_source_script
        # Romanized (IAST) segments have no Devanagari, so that filter is disabled
        # whenever the source script is preserved.
        self.require_devanagari = require_devanagari and not keep_source_script

        pipeline_kwargs: dict[str, object] = {
            "lang": "sa",
            "processors": "tokenize",
            "verbose": False,
        }
        if use_gpu is not None:
            pipeline_kwargs["use_gpu"] = use_gpu
        resolved_download = _resolve_download_method(download_method)
        if resolved_download is not None:
            pipeline_kwargs["download_method"] = resolved_download

        self._pipeline = _stanza.Pipeline(**pipeline_kwargs)

    def segment(self, text: str) -> list[Segment]:
        if not text.strip():
            return []

        doc = self._pipeline(text)
        segments: list[Segment] = []
        for sentence in doc.sentences:
            sentence_text = sentence.text.strip()
            if not sentence_text:
                continue
            if self.drop_verse_numbers and is_verse_number(sentence_text):
                continue
            if self.require_devanagari and not has_devanagari(sentence_text):
                continue
            start = sentence.tokens[0].start_char
            end = sentence.tokens[-1].end_char
            segments.append(Segment(sentence_text, start, end))
        return segments
