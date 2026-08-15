"""Deterministic routing policy for rule-based decisions.

Milestone 1 routing is fully deterministic: category = category with the most
matched signals; ties are broken by category priority; no signals maps to
``unknown``.
"""

from collections import Counter

from routellm.domain.categories import CATEGORY_PRIORITY
from routellm.domain.routes import CATEGORY_ROUTES
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


def route_for(category: str) -> str:
    """Return the logical route for a category."""
    return CATEGORY_ROUTES[category]
