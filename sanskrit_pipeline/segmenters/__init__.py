"""Sentence segmentation backends for Sanskrit."""

from .base import BaseSegmenter, Segment
from .dandas import DandasSegmenter
from .hybrid import HybridSegmenter
from .prose import ProseSegmenter
from .stanza_segmenter import StanzaSegmenter, stanza_available

__all__ = [
    "BaseSegmenter",
    "DandasSegmenter",
    "HybridSegmenter",
    "ProseSegmenter",
    "Segment",
    "StanzaSegmenter",
    "stanza_available",
]
