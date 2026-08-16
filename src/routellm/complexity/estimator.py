"""Lightweight heuristic complexity estimation (Milestone 5).

The estimator blends two normalized signals into a composite score in [0, 1]
and maps it to an ordinal level through validated thresholds (SPEC section
41): prompt length and the number of distinct matched indicators across the
reasoning, operation, technical, code, and clause vocabularies. It uses only
the existing preprocessing tokenizer — no LLM and no external model. Every
signal it reports is real: the explanation lists the actual matched
indicators or counts, never an invented reason.
"""

from routellm.complexity.config import ComplexityConfig
from routellm.domain.complexity import ComplexityEstimate
from routellm.preprocessing import preprocessor


def _matched_indicators(tokens: tuple[str, ...], indicators: tuple[str, ...]) -> tuple[str, ...]:
    """Return the distinct indicator phrases that appear in ``tokens``.

    A single-token indicator matches when present in ``tokens``. A
    multi-token indicator must appear as a contiguous run.
    """
    matched = []
    for indicator in indicators:
        phrase = tuple(indicator.split())
        if len(phrase) == 1:
            if phrase[0] in tokens:
                matched.append(indicator)
        elif _is_contiguous(phrase, tokens):
            matched.append(indicator)
    return tuple(matched)


def _is_contiguous(phrase: tuple[str, ...], tokens: tuple[str, ...]) -> bool:
    length = len(phrase)
    if length == 0 or len(tokens) < length:
        return False
    return any(
        tokens[index : index + length] == phrase for index in range(len(tokens) - length + 1)
    )


def estimate(text: str, config: ComplexityConfig | None = None) -> ComplexityEstimate:
    """Estimate the complexity level of ``text`` from heuristic signals."""
    config = config or ComplexityConfig()
    weights = config.feature_weights
    caps = config.feature_caps
    tokens = preprocessor.tokenize(text)
    reasoning = _matched_indicators(tokens, config.reasoning_indicators)
    operations = _matched_indicators(tokens, config.operation_indicators)
    technical = _matched_indicators(tokens, config.technical_glossary)
    code = _matched_indicators(tokens, config.code_indicators)
    clauses = _matched_indicators(tokens, config.clause_indicators)
    signal_sets = (
        set(reasoning),
        set(operations),
        set(technical),
        set(code),
        set(clauses),
    )
    distinct_count = len(set().union(*signal_sets))
    length_value = min(len(tokens) / caps["length"], 1.0)
    signals_value = min(distinct_count / caps["signals"], 1.0)
    score = weights["length"] * length_value + weights["signals"] * signals_value
    if score < config.low_threshold:
        level = config.levels[0]
    elif score < config.high_threshold:
        level = config.levels[1]
    else:
        level = config.levels[2]
    parts = [f"length ({len(tokens)} tokens)"]
    if reasoning:
        parts.append(f"reasoning indicators ({', '.join(reasoning)})")
    if operations:
        parts.append(f"operations ({', '.join(operations)})")
    if technical:
        parts.append(f"technical terms ({', '.join(technical)})")
    if code:
        parts.append(f"code complexity ({', '.join(code)})")
    if clauses:
        parts.append(f"clause markers ({', '.join(clauses)})")
    return ComplexityEstimate(text=text, level=level, score=round(score, 4), signals=tuple(parts))
