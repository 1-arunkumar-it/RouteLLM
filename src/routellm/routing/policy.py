"""Deterministic routing policy for rule-based decisions.

Milestone 1 routing is fully deterministic: category = category with the most
matched signals; ties are broken by category priority; no signals maps to
``unknown``. Milestone 5 adds complexity-aware route selection: the
category determines the base route and a high-complexity estimate may
re-route specific categories to ``reasoning`` (domain.routes.
``COMPLEXITY_REROUTES``), never changing the category decision itself.
"""

from collections import Counter

from routellm.domain.categories import CATEGORY_PRIORITY
from routellm.domain.routes import (
    CATEGORY_ROUTES,
    COMPLEXITY_LEVELS,
    COMPLEXITY_REROUTES,
    ROUTES,
)
from routellm.domain.signal import Signal


def decide_category(
    signals: tuple[Signal, ...],
    priority: tuple[str, ...] = CATEGORY_PRIORITY,
) -> str:
    """Choose a category from detected signals."""
    if not signals:
        return "unknown"
    counts = Counter(signal.category for signal in signals)
    highest = max(counts.values())
    tied = [category for category, count in counts.items() if count == highest]
    if len(tied) == 1:
        return tied[0]
    for category in priority:
        if category in tied:
            return category
    return "unknown"


def route_for(category: str, complexity_level: str = COMPLEXITY_LEVELS[0]) -> str:
    """Return the logical route for a category.

    A complexity level may re-route a category through
    ``COMPLEXITY_REROUTES``; otherwise the category's base route is used.
    """
    reroute = COMPLEXITY_REROUTES.get(category, {}).get(complexity_level)
    if reroute is not None:
        return reroute
    return CATEGORY_ROUTES[category]


def validate_complexity_routes(
    reroutes: dict[str, dict[str, str]] = COMPLEXITY_REROUTES,
    levels: tuple[str, ...] = COMPLEXITY_LEVELS,
    routes: frozenset[str] = ROUTES,
) -> None:
    """Validate a complexity reroute matrix, raising ``ValueError`` on problems."""
    for category, by_level in reroutes.items():
        if category not in CATEGORY_ROUTES:
            raise ValueError(f"Unknown reroute category: {category!r}.")
        if category == "unknown":
            raise ValueError("The 'unknown' category must never be re-routed by complexity.")
        for level, route in by_level.items():
            if level not in levels:
                raise ValueError(f"Unknown complexity level {level!r} for category {category!r}.")
            if route not in routes:
                raise ValueError(f"Unknown reroute {route!r} for category {category!r}.")
