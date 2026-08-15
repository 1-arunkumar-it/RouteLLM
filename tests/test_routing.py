"""Tests for the deterministic routing policy."""

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
    assert ROUTES == {"coding-local", "calculator", "translation", "general-local", "fallback"}
