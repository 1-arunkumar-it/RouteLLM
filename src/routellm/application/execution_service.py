"""Application layer executing a routed prompt through a configured provider.

Milestone 6 separates routing from execution: ``RouteService`` selects a
logical route and ``ExecutionService`` resolves that route to a configured
provider/model and calls it. The router therefore stays fully usable without
Ollama. When the primary provider is unavailable, a configured single-hop
fallback route is attempted before reporting ``unavailable`` (SPEC section 38).
"""

import time
from dataclasses import dataclass

from routellm.configuration.providers import ProviderConfig
from routellm.domain.provider import (
    STATUS_UNAVAILABLE,
    ProviderResponse,
    ResolvedProvider,
)
from routellm.domain.route_decision import RouteDecision
from routellm.providers.ollama import OllamaAdapter, OllamaError
from routellm.providers.registry import ProviderRegistry

_ROUTE_ORDER = (
    "coding-local",
    "calculator",
    "translation",
    "general-local",
    "reasoning",
    "fallback",
)


@dataclass(frozen=True)
class ProviderStatusRow:
    """One row of the provider status report for the ``providers`` command.

    ``available`` is ``None`` when the route has no configured provider; it is
    a bool when the route is configured, reflecting live server availability.
    """

    route: str
    provider: str | None
    model: str | None
    available: bool | None


@dataclass(frozen=True)
class ExecutionService:
    """Resolve a ``RouteDecision`` to a provider and execute the prompt.

    ``adapter_factory`` is an injectable ``(OllamaConfig) -> OllamaAdapter``
    seam so tests never contact the network.
    """

    config: ProviderConfig | None = None
    adapter_factory: callable | None = None

    def __post_init__(self) -> None:
        if self.config is not None and not isinstance(self.config, ProviderConfig):
            raise ValueError("config must be a ProviderConfig instance.")

    def _effective_config(self) -> ProviderConfig:
        return self.config if self.config is not None else ProviderConfig()

    def _registry(self) -> ProviderRegistry:
        return ProviderRegistry(config=self._effective_config())

    def _make_adapter(self) -> OllamaAdapter:
        config = self._effective_config().ollama
        if self.adapter_factory is not None:
            return self.adapter_factory(config)
        return OllamaAdapter(config=config)

    def execute(self, decision: RouteDecision) -> ProviderResponse:
        """Execute ``decision`` through its configured provider with fallback."""
        registry = self._registry()
        resolved = registry.resolve(decision.route)
        if resolved is None:
            return ProviderResponse(
                requested_route=decision.route,
                route=decision.route,
                provider=None,
                model=None,
                status="not_configured",
                text="",
                error=f"No provider is configured for route {decision.route!r}.",
                latency_ms=None,
            )
        adapter = self._make_adapter()
        if not adapter.has_model(resolved.model):
            return self._unavailable(decision, resolved, registry, adapter)
        try:
            return self._generate(decision, resolved, adapter, fallback_error="")
        except OllamaError as error:
            return ProviderResponse(
                requested_route=decision.route,
                route=resolved.route,
                provider=resolved.provider,
                model=resolved.model,
                status="error",
                text="",
                error=f"Provider generation failed: {error}",
                latency_ms=None,
            )

    def _unavailable(
        self,
        decision: RouteDecision,
        resolved: ResolvedProvider,
        registry: ProviderRegistry,
        adapter: OllamaAdapter,
    ) -> ProviderResponse:
        fallback_route = registry.fallback_for(decision.route)
        if fallback_route is None:
            return ProviderResponse(
                requested_route=decision.route,
                route=resolved.route,
                provider=resolved.provider,
                model=resolved.model,
                status=STATUS_UNAVAILABLE,
                text="",
                error=f"Provider for route {decision.route!r} is unavailable and no "
                "fallback route is configured.",
                latency_ms=None,
            )
        fallback = registry.resolve(fallback_route)
        if fallback is None or not adapter.has_model(fallback.model):
            return ProviderResponse(
                requested_route=decision.route,
                route=fallback_route,
                provider=fallback.provider if fallback else None,
                model=fallback.model if fallback else None,
                status=STATUS_UNAVAILABLE,
                text="",
                error=f"Provider for route {decision.route!r} is unavailable; fallback "
                f"route {fallback_route!r} is also unavailable.",
                latency_ms=None,
            )
        try:
            return self._generate(
                decision,
                fallback,
                adapter,
                fallback_error=f"Primary route {decision.route!r} was unavailable; "
                f"used fallback route {fallback_route!r}.",
            )
        except OllamaError as error:
            return ProviderResponse(
                requested_route=decision.route,
                route=fallback_route,
                provider=fallback.provider,
                model=fallback.model,
                status=STATUS_UNAVAILABLE,
                text="",
                error=f"Primary route {decision.route!r} and fallback route "
                f"{fallback_route!r} are unavailable: {error}",
                latency_ms=None,
            )

    def _generate(
        self,
        decision: RouteDecision,
        resolved: ResolvedProvider,
        adapter: OllamaAdapter,
        *,
        fallback_error: str,
    ) -> ProviderResponse:
        start = time.perf_counter()
        text = adapter.generate(resolved.model, decision.prompt)
        elapsed = time.perf_counter() - start
        return ProviderResponse(
            requested_route=decision.route,
            route=resolved.route,
            provider=resolved.provider,
            model=resolved.model,
            status="ok",
            text=text,
            error=fallback_error,
            latency_ms=elapsed * 1000,
        )

    def status_table(self) -> tuple[ProviderStatusRow, ...]:
        """Report configured providers and live availability for every route."""
        registry = self._registry()
        adapter = self._make_adapter()
        try:
            models = adapter.list_models()
            server_up = True
        except OllamaError:
            models = ()
            server_up = False
        rows = []
        for route in _ROUTE_ORDER:
            resolved = registry.resolve(route)
            if resolved is None:
                rows.append(
                    ProviderStatusRow(route=route, provider=None, model=None, available=None)
                )
            else:
                available = server_up and resolved.model in models
                rows.append(
                    ProviderStatusRow(
                        route=route,
                        provider=resolved.provider,
                        model=resolved.model,
                        available=available,
                    )
                )
        return tuple(rows)
