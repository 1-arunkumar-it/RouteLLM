"""Structured provider resolution, execution results, and health checks.

Routing and execution are separate concerns: routing selects a logical route
and execution resolves that route to a configured provider/model and calls it.
The domain holds only the result shapes; it must never depend on a provider
client. ``ProviderResponse.status`` reports how execution ended so the CLI can
present the outcome truthfully without inventing details (SPEC section 38).

Milestone 7 adds health checks and capability profiles for cost-aware and
latency-aware routing.
"""

from dataclasses import dataclass, field

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


@dataclass(frozen=True)
class HealthCheckResult:
    """The outcome of checking one provider route's availability.

    ``available`` is True when the model is reachable, False when the server
    is up but the model is missing, and None when the server itself is down.
    ``response_time_ms`` is the ping latency; None when the server is down.
    """

    route: str
    provider: str | None
    model: str | None
    available: bool | None
    response_time_ms: float | None
    checked_at: str
    error: str | None = None


@dataclass(frozen=True)
class ProviderProfile:
    """Static metadata about a provider model's cost, latency, and capabilities.

    ``cost_per_1k_tokens`` is the estimated cost in USD per 1000 tokens.
    ``estimated_latency_ms`` is the expected response time in milliseconds.
    ``capabilities`` is the set of capability tags this model supports (for
    example ``{"code", "reasoning"}``).
    """

    route: str
    cost_per_1k_tokens: float | None = None
    estimated_latency_ms: float | None = None
    capabilities: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.cost_per_1k_tokens is not None and self.cost_per_1k_tokens < 0:
            raise ValueError(
                f"cost_per_1k_tokens must be >= 0, got {self.cost_per_1k_tokens}."
            )
        if self.estimated_latency_ms is not None and self.estimated_latency_ms < 0:
            raise ValueError(
                f"estimated_latency_ms must be >= 0, got {self.estimated_latency_ms}."
            )
