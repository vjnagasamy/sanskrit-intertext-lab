"""Input normalization utilities for Sanskrit text."""

from __future__ import annotations

import re
import unicodedata

# indic_transliteration is optional at import time; only required when source_format="iast"
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate as _transliterate
    _INDIC_AVAILABLE = True
except ImportError:
    _INDIC_AVAILABLE = False

_MULTISPACE_RE = re.compile(r"\s+")
# Horizontal whitespace (spaces, tabs) but not newlines.
_HORIZONTAL_WS_RE = re.compile(r"[^\S\n]+")


def normalize_text(
    text: str,
    source_format: str = "devanagari",
    *,
    preserve_lines: bool = False,
    transliterate: bool = True,
) -> str:
    """Normalize incoming text to Unicode Devanagari.

    Args:
        text: Raw input text.
        source_format: ``"devanagari"`` (default) or ``"iast"``.
            NFC normalization is applied in both cases.
        preserve_lines: When True, keep newline boundaries (collapsing only
            horizontal whitespace and dropping blank lines). Required by
            line-based segmentation engines; the default collapses all
            whitespace to single spaces.
        transliterate: When False, IAST input is left in its romanized script
            instead of being converted to Devanagari (NFC, BOM, and whitespace
            handling still apply). Used by engines that segment in the source
            script.

    Returns:
        Normalized Devanagari string (or romanized text when
        ``transliterate=False`` and ``source_format="iast"``).

    Raises:
        ValueError: If source_format is not recognized. Includes a hint for
            users who pass Tibetan pipeline format strings ("unicode", "wylie").
    """
    if text is None:
        return ""

    source_format = source_format.lower()
    normalized = unicodedata.normalize("NFC", text).replace("\ufeff", "")
    if preserve_lines:
        lines = (_HORIZONTAL_WS_RE.sub(" ", line).strip() for line in normalized.split("\n"))
        cleaned = "\n".join(line for line in lines if line)
    else:
        cleaned = _MULTISPACE_RE.sub(" ", normalized.strip())

    if source_format == "devanagari":
        return cleaned

    if source_format == "iast":
        if not transliterate:
            return cleaned
        if not _INDIC_AVAILABLE:
            raise ImportError(
                "indic-transliteration is required for IAST input. "
                "Install it with: pip install indic-transliteration"
            )
        converted = _transliterate(cleaned, sanscript.IAST, sanscript.DEVANAGARI)
        return unicodedata.normalize("NFC", converted).strip()

    if source_format in {"unicode", "wylie"}:
        raise ValueError(
            f"Unsupported source format {source_format!r}. "
            "The Sanskrit pipeline uses 'devanagari' and 'iast', not 'unicode' or 'wylie'. "
            "Those are Tibetan pipeline format names."
        )

    raise ValueError(
        f"Unsupported source format: {source_format!r}. Use 'devanagari' or 'iast'."
    )
