"""Application layer orchestrating one routing use case."""

from dataclasses import dataclass

from routellm.classification.cascade_model import CascadeModel
from routellm.configuration.cascade import CascadeConfig
from routellm.domain.route_decision import RouteDecision
from routellm.domain.signal import Signal
from routellm.preprocessing import preprocessor
from routellm.routing import policy
from routellm.routing.cascade import CascadeOutcome, apply_cascade
from routellm.signals import engine
from routellm.signals.keyword_rules import SIGNAL_RULES
from routellm.signals.validation import validate_rules


@dataclass(frozen=True)
class RouteService:
    """Coordinate preprocessing, signal detection, and routing policy.

    When a ``model`` (a fitted ``CascadeModel``) is provided, the cascade
    policy combines rule evidence with calibrated classifier confidence and a
    validated fallback. Without a model the service stays rule-only: every
    decision has ``source="rules"`` and ``confidence=None``.
    """

    signal_rules: dict[str, tuple[tuple[str, ...], ...]] | None = None
    model: CascadeModel | None = None
    cascade_config: CascadeConfig | None = None

    def __post_init__(self) -> None:
        rules = self.signal_rules if self.signal_rules is not None else SIGNAL_RULES
        validate_rules(rules)
        if self.model is not None and not 0 <= self.model.threshold <= 1:
            raise ValueError(
                f"Cascade model threshold must be in [0, 1], got {self.model.threshold}."
            )
        if self.cascade_config is not None and self.model is None:
            raise ValueError("cascade_config requires a cascade model.")

    def _effective_threshold(self) -> float:
        if self.cascade_config is not None and self.cascade_config.threshold is not None:
            return self.cascade_config.threshold
        return self.model.threshold

    def route(self, prompt: str) -> RouteDecision:
        """Route a prompt to a category and logical route."""
        rules = self.signal_rules if self.signal_rules is not None else SIGNAL_RULES
        tokens = preprocessor.tokenize(prompt)
        signals = engine.detect_signals(tokens, rules)
        if self.model is None:
            category = policy.decide_category(signals)
            route = policy.route_for(category)
            return RouteDecision(
                prompt=prompt,
                category=category,
                route=route,
                signals=signals,
                confidence=None,
                source="rules",
                reason=build_reason(category, signals),
            )
        prediction = self.model.predict(prompt)
        threshold = self._effective_threshold()
        outcome = apply_cascade(
            signals,
            prediction,
            threshold=threshold,
            override_categories=self.model.override_categories,
        )
        return RouteDecision(
            prompt=prompt,
            category=outcome.category,
            route=policy.route_for(outcome.category),
            signals=signals,
            confidence=outcome.confidence,
            source=outcome.source,
            reason=build_cascade_reason(outcome, len(signals), threshold),
        )


def build_reason(category: str, signals: tuple[Signal, ...]) -> str:
    """Build a truthful explanation from the detected signals."""
    if not signals:
        return "No keyword signals were detected."
    count = sum(1 for signal in signals if signal.category == category)
    return f"Selected category '{category}' with {count} matched signal(s)."


def build_cascade_reason(outcome: CascadeOutcome, signal_count: int, threshold: float) -> str:
    """Build a truthful explanation naming the source of a cascade decision."""
    if outcome.source == "rules":
        return (
            f"Rule signals ({signal_count} matched) selected category "
            f"'{outcome.category}' with validated precision on held-out data."
        )
    if outcome.source == "classifier":
        return (
            f"Calibrated classifier predicted '{outcome.category}' with confidence "
            f"{outcome.confidence:.2f} at threshold {threshold:.2f}."
        )
    if outcome.confidence is None:
        return "No confident decision was available; routed to fallback."
    return (
        f"Confidence {outcome.confidence:.2f} was below threshold {threshold:.2f}; "
        "routed to fallback."
    )
