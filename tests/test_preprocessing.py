"""Tests for prompt preprocessing."""

from routellm.preprocessing import preprocessor


def test_normalize_lowercases_and_collapses_whitespace():
    assert preprocessor.normalize("  Hello   WORLD ") == "hello world"


def test_normalize_empty_and_whitespace_only():
    assert preprocessor.normalize("") == ""
    assert preprocessor.normalize("   ") == ""


def test_tokenize_strips_sentence_punctuation():
    assert preprocessor.tokenize("Translate this into Tamil.") == (
        "translate",
        "this",
        "into",
        "tamil",
    )


def test_tokenize_preserves_technical_terms():
    assert preprocessor.tokenize("Write a C++ program in C#.") == (
        "write",
        "a",
        "c++",
        "program",
        "in",
        "c#",
    )


def test_tokenize_preserves_numbers_and_currency():
    assert preprocessor.tokenize("Claim ₹18,500, partially approved.") == (
        "claim",
        "₹18,500",
        "partially",
        "approved",
    )


def test_tokenize_drops_blank_tokens():
    assert preprocessor.tokenize("...") == ()
    assert preprocessor.tokenize("") == ()
