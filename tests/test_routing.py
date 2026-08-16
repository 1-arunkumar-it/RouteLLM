"""Tests for the deterministic routing policy."""

import pytest

from routellm.domain.categories import CATEGORY_PRIORITY
from routellm.domain.routes import CATEGORY_ROUTES, ROUTES
from routellm.domain.signal import Signal
from routellm.routing import policy
from routellm.signals.keyword_rules import SIGNAL_RULES


def signal(category: str, phrase: str = "x") -> Signal:
    return Signal(phrase=phrase, category=category)


def test_no_signals_maps_to_unknown():
    assert policy.decide_category(()) == "unknown"


def test_majority_category_wins():
    signals = (
        signal("coding", "python"),
        signal("coding", "java"),
        signal("general_qa", "why"),
    )
    assert policy.decide_category(signals) == "coding"


def test_tie_is_broken_by_category_priority():
    signals = (signal("math", "times"), signal("general_qa", "what is"))
    assert policy.decide_category(signals) == "math"


def test_route_for_known_categories():
    assert policy.route_for("coding") == "coding-local"
    assert policy.route_for("math") == "calculator"
    assert policy.route_for("translation") == "translation"
    assert policy.route_for("general_qa") == "general-local"
    assert policy.route_for("unknown") == "fallback"


def test_every_category_has_a_route():
    for category in CATEGORY_PRIORITY:
        assert category in CATEGORY_ROUTES


def test_rule_categories_are_part_of_the_taxonomy():
    for category in SIGNAL_RULES:
        assert category in CATEGORY_PRIORITY


def test_routes_are_valid_logical_labels():
    assert ROUTES == {
        "coding-local",
        "calculator",
        "translation",
        "general-local",
        "fallback",
        "reasoning",
    }


def test_route_for_is_unaffected_by_low_complexity():
    assert policy.route_for("general_qa", "low") == "general-local"
    assert policy.route_for("summarization", "low") == "general-local"


def test_high_complexity_general_qa_reroutes_to_reasoning():
    assert policy.route_for("general_qa", "high") == "reasoning"


def test_high_complexity_summarization_reroutes_to_reasoning():
    assert policy.route_for("summarization", "high") == "reasoning"


def test_medium_complexity_keeps_default_routes():
    assert policy.route_for("general_qa", "medium") == "general-local"
    assert policy.route_for("summarization", "medium") == "general-local"


@pytest.mark.parametrize(
    "category",
    ["coding", "math", "translation", "creative_writing", "unknown"],
)
def test_complexity_never_reroutes_unlisted_categories(category):
    assert policy.route_for(category, "high") == CATEGORY_ROUTES[category]


def test_complexity_reroute_must_point_at_a_real_route():
    with pytest.raises(ValueError, match="Unknown reroute"):
        policy.validate_complexity_routes({"general_qa": {"high": "not-a-route"}})


def test_complexity_reroute_must_use_known_category():
    with pytest.raises(ValueError, match="Unknown reroute category"):
        policy.validate_complexity_routes({"other": {"high": "reasoning"}})


def test_complexity_reroute_must_use_known_level():
    with pytest.raises(ValueError, match="Unknown complexity level"):
        policy.validate_complexity_routes({"general_qa": {"extreme": "reasoning"}})


def test_complexity_reroute_cannot_target_unknown_category():
    with pytest.raises(ValueError, match="must never be re-routed"):
        policy.validate_complexity_routes({"unknown": {"high": "reasoning"}})
