"""Structured result of the routing use case (Milestone 6, extended in M7)."""

from dataclasses import dataclass

from routellm.domain.complexity import ComplexityEstimate
from routellm.domain.signal import Signal


@dataclass(frozen=True)
class RouteDecision:
    """The result of routing one prompt.

    ``confidence`` is ``None`` until a calibrated confidence source exists
    (Milestone 2+). A ``None`` value must never be presented as certainty.
    ``source`` names the origin of the decision: ``"rules"``, ``"classifier"``,
    or ``"fallback"`` (Milestone 4). ``complexity`` (Milestone 5) is the
    heuristic estimate that may re-route a category to ``reasoning``.

    Milestone 7 adds optional ``estimated_cost`` and ``estimated_latency_ms``
    fields for cost-aware and latency-aware routing.
    """

    prompt: str
    category: str
    route: str
    signals: tuple[Signal, ...] = ()
    confidence: float | None = None
    source: str = ""
    reason: str = ""
    complexity: ComplexityEstimate | None = None
    estimated_cost: float | None = None
    estimated_latency_ms: float | None = None
