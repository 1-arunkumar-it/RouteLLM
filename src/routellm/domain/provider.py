"""Structured provider resolution and execution results (Milestone 6).

Routing and execution are separate concerns: routing selects a logical route
and execution resolves that route to a configured provider/model and calls it.
The domain holds only the result shapes; it must never depend on a provider
client. ``ProviderResponse.status`` reports how execution ended so the CLI can
present the outcome truthfully without inventing details (SPEC section 38).
"""

from dataclasses import dataclass

STATUS_OK = "ok"
STATUS_NOT_CONFIGURED = "not_configured"
STATUS_UNAVAILABLE = "unavailable"
STATUS_ERROR = "error"

PROVIDER_STATUSES = frozenset(
    {STATUS_OK, STATUS_NOT_CONFIGURED, STATUS_UNAVAILABLE, STATUS_ERROR}
)


@dataclass(frozen=True)
class ResolvedProvider:
    """A logical route resolved to a concrete provider and model.

    ``provider`` names a registered provider type (for example ``ollama``) and
    ``model`` is the configured model identifier for that provider. Both names
    come from configuration, never from routing code (SPEC sections 22-23).
    """

    route: str
    provider: str
    model: str


@dataclass(frozen=True)
class ProviderResponse:
    """The outcome of executing one routed prompt through a provider.

    ``requested_route`` is the route chosen by the routing engine; ``route`` is
    the route actually executed, which may differ after a configured fallback.
    ``status`` is one of ``ok``, ``not_configured``, ``unavailable``, or
    ``error``. ``text`` holds the model output on success; ``error`` explains
    failure and is never invented.
    """

    requested_route: str
    route: str
    provider: str | None
    model: str | None
    status: str
    text: str
    error: str
    latency_ms: float | None

    def __post_init__(self) -> None:
        if self.status not in PROVIDER_STATUSES:
            raise ValueError(f"Unknown provider status {self.status!r}.")
