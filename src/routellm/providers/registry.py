"""Provider registry resolving logical routes to configured providers.

The registry is a pure mapping layer: given a logical route it returns the
configured ``(provider, model)`` pair or the configured unavailability
fallback route. It performs no network activity and never imports a provider
client (ARCHITECTURE: providers -> Domain + Configuration only).
"""

from dataclasses import dataclass

from routellm.configuration.providers import ProviderConfig
from routellm.domain.provider import ResolvedProvider


@dataclass(frozen=True)
class ProviderRegistry:
    """Resolve logical routes against a validated ``ProviderConfig``."""

    config: ProviderConfig

    def resolve(self, route: str) -> ResolvedProvider | None:
        """Return the configured provider for ``route``, or None if unconfigured."""
        entry = self.config.routes.get(route)
        if entry is None:
            return None
        provider, model = entry
        return ResolvedProvider(route=route, provider=provider, model=model)

    def fallback_for(self, route: str) -> str | None:
        """Return the configured fallback route for ``route``, or None."""
        return self.config.fallbacks.get(route)

    def configured_routes(self) -> tuple[str, ...]:
        """Return the routes that have a provider, in configuration order."""
        return tuple(self.config.routes)
