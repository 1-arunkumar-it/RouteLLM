"""Tests for keyword, phrase, and numeric-expression signal detection."""

from routellm.preprocessing import preprocessor
from routellm.signals.engine import detect_signals, phrase_matches


def test_phrase_matches_requires_contiguous_tokens():
    assert phrase_matches(("spring", "boot"), ("write", "spring", "boot", "app"))
    assert not phrase_matches(("spring", "boot"), ("spring", "then", "boot"))


def test_single_word_does_not_match_inside_longer_token():
    assert not phrase_matches(("api",), ("capital",))


def test_detect_signals_for_coding_prompt():
    tokens = preprocessor.tokenize("Write a Java Spring Boot REST API")
    signals = detect_signals(tokens)
    phrases = {signal.phrase for signal in signals}
    categories = {signal.category for signal in signals}
    assert phrases == {"java", "spring boot", "rest api"}
    assert categories == {"coding"}


def test_longest_phrase_wins_over_contained_word():
    signals = detect_signals(preprocessor.tokenize("Write a REST API"))
    assert {signal.phrase for signal in signals} == {"rest api"}


def test_independent_signals_are_all_retained():
    signals = detect_signals(preprocessor.tokenize("Write a Java Spring Boot REST API"))
    phrases = {signal.phrase for signal in signals}
    assert {"java", "spring boot", "rest api"} <= phrases


def test_detect_signals_for_translation_prompt():
    signals = detect_signals(preprocessor.tokenize("Translate this into Tamil"))
    phrases = {signal.phrase for signal in signals}
    assert phrases == {"translate", "tamil"}


def test_detect_signals_returns_empty_for_unknown_prompt():
    assert detect_signals(preprocessor.tokenize("zzz qqq florb")) == ()


def test_numeric_multiplication_creates_one_math_signal():
    signals = detect_signals(("25", "×", "48"))
    assert len(signals) == 1
    assert signals[0].category == "math"
    assert signals[0].phrase == "25 × 48"


def test_multiplication_operator_without_numeric_operands_is_ignored():
    signals = detect_signals(("write", "×", "hello"))
    assert not any(signal.category == "math" for signal in signals)
