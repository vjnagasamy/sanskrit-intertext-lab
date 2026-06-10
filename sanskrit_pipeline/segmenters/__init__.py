"""Sentence segmentation backends for Sanskrit."""

from .base import BaseSegmenter, Segment
from .dandas import DandasSegmenter
from .lines import LineSegmenter
from .stanza_segmenter import StanzaSegmenter, stanza_available

__all__ = [
    "BaseSegmenter",
    "DandasSegmenter",
    "LineSegmenter",
    "Segment",
    "StanzaSegmenter",
    "stanza_available",
]
