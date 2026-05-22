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


def normalize_text(text: str, source_format: str = "devanagari") -> str:
    """Normalize incoming text to Unicode Devanagari.

    Args:
        text: Raw input text.
        source_format: ``"devanagari"`` (default) or ``"iast"``.
            NFC normalization is applied in both cases.

    Returns:
        Normalized Devanagari string.

    Raises:
        ValueError: If source_format is not recognized. Includes a hint for
            users who pass Tibetan pipeline format strings ("unicode", "wylie").
    """
    if text is None:
        return ""

    source_format = source_format.lower()
    cleaned = unicodedata.normalize("NFC", text).strip()
    cleaned = _MULTISPACE_RE.sub(" ", cleaned)

    if source_format == "devanagari":
        return cleaned

    if source_format == "iast":
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
