"""HTML report generators for corpus pairwise workflows."""

from .bidirectional_synthesis_report import generate_bidirectional_synthesis_report
from .corpus_pairwise_report import generate_corpus_pairwise_report

__all__ = [
    "generate_bidirectional_synthesis_report",
    "generate_corpus_pairwise_report",
]
