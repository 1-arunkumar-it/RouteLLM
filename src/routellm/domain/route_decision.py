"""Structured result of the routing use case."""

from dataclasses import dataclass

from routellm.domain.signal import Signal


@dataclass(frozen=True)
class RouteDecision:
    """The result of routing one prompt.

    ``confidence`` is ``None`` until a calibrated confidence source exists
    (Milestone 2+). A ``None`` value must never be presented as certainty.
    ``source`` names the origin of the decision: ``"rules"``, ``"classifier"``,
    or ``"fallback"`` (Milestone 4).
    """

    prompt: str
    category: str
    route: str
    signals: tuple[Signal, ...] = ()
    confidence: float | None = None
    source: str = ""
    reason: str = ""
