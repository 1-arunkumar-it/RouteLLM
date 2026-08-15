"""Lightweight prompt normalization and tokenization.

Milestone 1 uses simple, dependency-free preprocessing. spaCy is deliberately
not introduced yet: per SPEC.md and ARCHITECTURE.md it must be benchmarked
against simpler preprocessing before it is retained as a dependency.
"""

import unicodedata

_EDGE_PUNCTUATION = ".,;:!?()[]{}'\"\\`"


def normalize(text: str) -> str:
    """Normalize a prompt: NFKC, lowercase, and collapsed whitespace."""
    normalized = unicodedata.normalize("NFKC", text).lower()
    return " ".join(normalized.split())


def tokenize(text: str) -> tuple[str, ...]:
    """Split a prompt into cleaned tokens.

    Sentence punctuation is stripped from token edges, while technical terms
    such as ``C++``, ``C#``, ``TF-IDF``, and ``₹18,500`` are preserved.
    """
    tokens = []
    for token in normalize(text).split():
        cleaned = token.strip(_EDGE_PUNCTUATION)
        if cleaned:
            tokens.append(cleaned)
    return tuple(tokens)
