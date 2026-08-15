"""Validation of rule configuration."""

from routellm.domain.categories import CATEGORIES
from routellm.domain.routes import CATEGORY_ROUTES


def validate_rules(rules: dict[str, tuple[tuple[str, ...], ...]]) -> None:
    """Validate a rule configuration, raising ``ValueError`` on problems."""
    for category, phrases in rules.items():
        if category not in CATEGORIES:
            raise ValueError(f"Unknown rule category: {category!r}.")
        if category not in CATEGORY_ROUTES:
            raise ValueError(f"No route is defined for category: {category!r}.")
        seen = set()
        for phrase in phrases:
            if not phrase or any(not token for token in phrase):
                raise ValueError(f"Empty phrase in category: {category!r}.")
            if phrase in seen:
                raise ValueError(f"Duplicate phrase {phrase!r} in category: {category!r}.")
            seen.add(phrase)
