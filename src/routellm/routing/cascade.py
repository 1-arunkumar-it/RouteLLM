"""Cascaded routing policy combining rule evidence and calibrated confidence.

Milestone 4 policy, following the "cheapest reliable mechanism" principle:
rule evidence that is precise on held-out data wins; otherwise the calibrated
classifier is used above its validated confidence threshold; anything else
routes to ``unknown``/``fallback`` without ever presenting uncertainty as
certainty (SPEC sections 6, 20, and 38).

``threshold`` and ``override_categories`` are not hard-coded here: they are
selected at training time from the validation split by ``routellm cascade``
and stored in the cascade model. On the current dataset
(``data/datasets/prompts.csv``, seed 42) that run selected threshold 0.35
(validation macro F1 0.925) and override categories coding, creative_writing,
math, summarization, and translation; the cascade measured 0.973 accuracy and
0.976 macro F1 on the held-out test split. These numbers are data-dependent
and regenerated on every training run.

Milestone 7 adds cost-aware and latency-aware constraint application.
"""

from dataclasses import dataclass

from routellm.configuration.providers import RouteProfile, RoutingConstraints
from routellm.domain.classifier_prediction import ClassifierPrediction
from routellm.domain.signal import Signal
from routellm.routing import policy

SOURCE_RULES = "rules"
SOURCE_CLASSIFIER = "classifier"
SOURCE_FALLBACK = "fallback"


@dataclass(frozen=True)
class CascadeOutcome:
    """The result of one cascaded routing decision."""

    category: str
    source: str
    confidence: float | None


def cascade_outcomes(
    rule_categories: tuple[str, ...],
    predictions: tuple[ClassifierPrediction, ...],
    *,
    threshold: float,
    override_categories: frozenset[str],
) -> tuple[CascadeOutcome, ...]:
    """Apply the cascade policy to aligned rule and classifier results.

    For each prompt: a rule category that is override-safe wins immediately;
    otherwise a calibrated confidence at or above ``threshold`` accepts the
    classifier's category; anything else falls back to ``unknown``.
    """
    if len(rule_categories) != len(predictions):
        raise ValueError("rule_categories and predictions must have equal length.")
    if not 0 <= threshold <= 1:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}.")
    outcomes = []
    for rule_category, prediction in zip(rule_categories, predictions):
        if rule_category in override_categories:
            outcomes.append(CascadeOutcome(rule_category, SOURCE_RULES, None))
        elif prediction.confidence is not None and prediction.confidence >= threshold:
            outcomes.append(
                CascadeOutcome(prediction.category, SOURCE_CLASSIFIER, prediction.confidence)
            )
        else:
            outcomes.append(CascadeOutcome("unknown", SOURCE_FALLBACK, prediction.confidence))
    return tuple(outcomes)


def apply_cascade(
    signals: tuple[Signal, ...],
    prediction: ClassifierPrediction,
    *,
    threshold: float,
    override_categories: frozenset[str],
) -> CascadeOutcome:
    """Apply the cascade policy to one prompt's signals and prediction."""
    rule_category = policy.decide_category(signals)
    return cascade_outcomes(
        (rule_category,),
        (prediction,),
        threshold=threshold,
        override_categories=override_categories,
    )[0]


def estimate_cost(word_count: int, profile: RouteProfile) -> float | None:
    """Estimate the cost of generating a response for the given word count.

    Returns the estimated cost in USD, or None if the profile has no cost data.
    Uses the approximation that 1 word ≈ 1.3 tokens.
    """
    if profile.cost_per_1k_tokens is None:
        return None
    tokens = word_count * 1.3
    return (tokens / 1000) * profile.cost_per_1k_tokens


def apply_constraints(
    route: str,
    category: str,
    profiles: dict[str, RouteProfile],
    constraints: RoutingConstraints,
    *,
    prompt_word_count: int,
) -> tuple[str, float | None, float | None]:
    """Apply cost and latency constraints to reroute when necessary.

    Returns ``(route, estimated_cost, estimated_latency_ms)`` where the route
    may differ from the input when a constraint is violated and a suitable
    alternative exists.
    """
    profile = profiles.get(route)
    if profile is None:
        return route, None, None

    cost = estimate_cost(prompt_word_count, profile)
    latency = profile.estimated_latency_ms

    cost_violated = (
        constraints.max_cost_per_prompt is not None
        and cost is not None
        and cost > constraints.max_cost_per_prompt
    )
    latency_violated = (
        constraints.max_latency_ms is not None
        and latency is not None
        and latency > constraints.max_latency_ms
    )

    if not cost_violated and not latency_violated:
        return route, cost, latency

    alternative = _find_alternative(
        route, profiles, constraints, cost_violated, latency_violated,
        prompt_word_count=prompt_word_count,
    )
    if alternative is not None:
        alt_profile = profiles.get(alternative)
        alt_cost = estimate_cost(prompt_word_count, alt_profile) if alt_profile else None
        alt_latency = alt_profile.estimated_latency_ms if alt_profile else None
        return alternative, alt_cost, alt_latency

    return route, cost, latency


def _find_alternative(
    route: str,
    profiles: dict[str, RouteProfile],
    constraints: RoutingConstraints,
    cost_violated: bool,
    latency_violated: bool,
    *,
    prompt_word_count: int,
) -> str | None:
    """Find a cheaper or faster alternative route with overlapping capabilities.

    The alternative must satisfy all violated constraints (not merely be
    better than the current route). When multiple candidates qualify, the
    one with the lowest combined cost-and-latency score wins.
    """
    current = profiles.get(route)
    if current is None:
        return None
    best: str | None = None
    best_score: float | None = None
    for alt_route, alt_profile in profiles.items():
        if alt_route == route:
            continue
        if not alt_profile.capabilities & current.capabilities:
            continue
        if cost_violated:
            if alt_profile.cost_per_1k_tokens is None:
                continue
            if current.cost_per_1k_tokens is not None:
                if alt_profile.cost_per_1k_tokens >= current.cost_per_1k_tokens:
                    continue
            alt_cost = estimate_cost(prompt_word_count, alt_profile)
            if (
                alt_cost is not None
                and constraints.max_cost_per_prompt is not None
                and alt_cost > constraints.max_cost_per_prompt
            ):
                continue
        if latency_violated:
            if alt_profile.estimated_latency_ms is None:
                continue
            if current.estimated_latency_ms is not None:
                if alt_profile.estimated_latency_ms >= current.estimated_latency_ms:
                    continue
            if (
                constraints.max_latency_ms is not None
                and alt_profile.estimated_latency_ms > constraints.max_latency_ms
            ):
                continue
        cost_val = alt_profile.cost_per_1k_tokens or 0
        lat_val = (alt_profile.estimated_latency_ms or 0) / 1000
        score = cost_val + lat_val
        if best_score is None or score < best_score:
            best = alt_route
            best_score = score
    return best
