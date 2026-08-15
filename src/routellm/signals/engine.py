"""Detection of keyword, phrase, and numeric-expression signals."""

from routellm.domain.signal import Signal
from routellm.signals.keyword_rules import SIGNAL_RULES

MULTIPLICATION_OPERATOR = "×"


def phrase_matches(phrase: tuple[str, ...], tokens: tuple[str, ...]) -> bool:
    """Return True if ``phrase`` appears as a contiguous run of ``tokens``."""
    return _phrase_span(phrase, tokens) is not None


def _phrase_span(phrase: tuple[str, ...], tokens: tuple[str, ...]) -> int | None:
    """Return the start index of ``phrase`` in ``tokens``, or None."""
    length = len(phrase)
    if length == 0 or len(tokens) < length:
        return None
    for index in range(len(tokens) - length + 1):
        if tokens[index : index + length] == phrase:
            return index
    return None


def _is_numeric(token: str) -> bool:
    return bool(token) and token.replace(",", "").replace(".", "").isdigit()


def _numeric_multiplication_span(tokens: tuple[str, ...]) -> tuple[int, int] | None:
    """Detect the SPEC multiplication form: a number ``×`` a number."""
    for index in range(len(tokens) - 2):
        if (
            _is_numeric(tokens[index])
            and tokens[index + 1] == MULTIPLICATION_OPERATOR
            and _is_numeric(tokens[index + 2])
        ):
            return index, index + 3
    return None


def _apply_precedence(
    matches: list[tuple[int, int, str, str]],
) -> list[tuple[int, int, str, str]]:
    """Drop signals wholly contained in a longer signal of the same category."""
    kept = []
    for start, end, category, phrase in matches:
        if _is_contained(start, end, category, matches):
            continue
        kept.append((start, end, category, phrase))
    return kept


def _is_contained(
    start: int,
    end: int,
    category: str,
    matches: list[tuple[int, int, str, str]],
) -> bool:
    return any(
        other_start <= start
        and other_end >= end
        and (other_start, other_end) != (start, end)
        and other_category == category
        for other_start, other_end, other_category, _ in matches
    )


def detect_signals(
    tokens: tuple[str, ...],
    rules: dict[str, tuple[tuple[str, ...], ...]] = SIGNAL_RULES,
) -> tuple[Signal, ...]:
    """Detect all routing signals in ``tokens``.

    Keyword rules match first. The numeric-expression rule emits a single
    ``math`` signal derived from the matched prompt tokens. Overlapping
    signals in the same category are reduced to the longest phrase, so a
    contained word such as ``api`` inside ``rest api`` is not double-counted.
    """
    matches = []
    for category, phrases in rules.items():
        for phrase in phrases:
            start = _phrase_span(phrase, tokens)
            if start is not None:
                matches.append((start, start + len(phrase), category, " ".join(phrase)))
    numeric = _numeric_multiplication_span(tokens)
    if numeric is not None:
        start, end = numeric
        matches.append((start, end, "math", " ".join(tokens[start:end])))
    return tuple(
        Signal(phrase=phrase, category=category)
        for start, end, category, phrase in sorted(
            _apply_precedence(matches), key=lambda match: (match[0], match[1], match[2])
        )
    )
