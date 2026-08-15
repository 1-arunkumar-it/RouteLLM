"""Structured result of the routing use case."""

from dataclasses import dataclass

from routellm.domain.signal import Signal


@dataclass(frozen=True)
class RouteDecision:
    """The result of routing one prompt.

    ``confidence`` is ``None`` until a calibrated confidence source exists
    (Milestone 2+). A ``None`` value must never be presented as certainty.
    """

    prompt: str
    category: str
    route: str
    signals: tuple[Signal, ...] = ()
    confidence: float | None = None
    reason: str = ""
