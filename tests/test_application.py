"""Integration tests for the application routing use case."""

import pytest

from routellm.application.route_service import RouteService
from routellm.signals.keyword_rules import SIGNAL_RULES
from routellm.signals.validation import validate_rules


def test_coding_prompt():
    decision = RouteService().route(
        "Write a Java Spring Boot REST API for user authentication"
    )
    assert decision.category == "coding"
    assert decision.route == "coding-local"
    assert decision.confidence is None
    assert any(signal.phrase == "java" for signal in decision.signals)


def test_translation_prompt():
    decision = RouteService().route("Translate this paragraph into Tamil")
    assert decision.category == "translation"
    assert decision.route == "translation"


def test_math_prompt():
    decision = RouteService().route("What is 25 times 48?")
    assert decision.category == "math"
    assert decision.route == "calculator"


def test_summarization_prompt():
    decision = RouteService().route("Summarize this article for me")
    assert decision.category == "summarization"
    assert decision.route == "general-local"


def test_unknown_prompt():
    decision = RouteService().route("The zebra florbled quizzically at dusk")
    assert decision.category == "unknown"
    assert decision.route == "fallback"
    assert decision.signals == ()


def test_empty_prompt_is_unknown():
    decision = RouteService().route("")
    assert decision.category == "unknown"
    assert decision.reason == "No keyword signals were detected."


def test_only_actual_signals_are_reported():
    decision = RouteService().route("Translate this into Tamil")
    assert {signal.phrase for signal in decision.signals} == {"translate", "tamil"}


def test_reason_reflects_detected_signals():
    decision = RouteService().route("Translate this into Tamil")
    assert "translation" in decision.reason
    assert "2" in decision.reason


def test_custom_rules_can_be_injected():
    rules = {"coding": (("python",),), "math": (("times",),)}
    decision = RouteService(signal_rules=rules).route("python times")
    assert decision.category == "coding"


def test_multiplication_expression_prompt():
    decision = RouteService().route("What is 25 × 48?")
    assert decision.category == "math"
    assert decision.route == "calculator"
    assert any(signal.phrase == "25 × 48" for signal in decision.signals)


def test_overlapping_phrase_is_reported_once():
    decision = RouteService().route("Write a REST API")
    assert decision.category == "coding"
    assert decision.route == "coding-local"
    assert {signal.phrase for signal in decision.signals} == {"rest api"}


def test_invalid_category_raises_value_error():
    with pytest.raises(ValueError, match="Unknown rule category"):
        RouteService(signal_rules={"other": (("x",),)})


def test_empty_phrase_raises_value_error():
    with pytest.raises(ValueError, match="Empty phrase"):
        RouteService(signal_rules={"coding": (("x",), ())})


def test_duplicate_phrase_raises_value_error():
    with pytest.raises(ValueError, match="Duplicate phrase"):
        RouteService(signal_rules={"coding": (("x",), ("x",))})


def test_default_rules_validate_successfully():
    RouteService()
    validate_rules(SIGNAL_RULES)
