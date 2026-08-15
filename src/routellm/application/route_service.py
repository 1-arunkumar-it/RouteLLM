"""Application layer orchestrating one routing use case."""

from dataclasses import dataclass

from routellm.domain.route_decision import RouteDecision
from routellm.domain.signal import Signal
from routellm.preprocessing import preprocessor
from routellm.routing import policy
from routellm.signals import engine
from routellm.signals.keyword_rules import SIGNAL_RULES
from routellm.signals.validation import validate_rules


@dataclass(frozen=True)
class RouteService:
    """Coordinate preprocessing, signal detection, and routing policy."""

    signal_rules: dict[str, tuple[tuple[str, ...], ...]] | None = None

    def __post_init__(self) -> None:
        rules = self.signal_rules if self.signal_rules is not None else SIGNAL_RULES
        validate_rules(rules)

    def route(self, prompt: str) -> RouteDecision:
        """Route a prompt to a category and logical route."""
        rules = self.signal_rules if self.signal_rules is not None else SIGNAL_RULES
        tokens = preprocessor.tokenize(prompt)
        signals = engine.detect_signals(tokens, rules)
        category = policy.decide_category(signals)
        route = policy.route_for(category)
        return RouteDecision(
            prompt=prompt,
            category=category,
            route=route,
            signals=signals,
            confidence=None,
            reason=build_reason(category, signals),
        )


def build_reason(category: str, signals: tuple[Signal, ...]) -> str:
    """Build a truthful explanation from the detected signals."""
    if not signals:
        return "No keyword signals were detected."
    count = sum(1 for signal in signals if signal.category == category)
    return f"Selected category '{category}' with {count} matched signal(s), the highest count."
