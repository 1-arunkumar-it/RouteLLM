"""Validated provider configuration (Milestone 6, extended in Milestone 7).

Configuration is the only place where provider names, model names, hosts, and
fallbacks are defined; routing code never names a provider or model directly
(SPEC sections 22-23). Every value is validated at construction time, the
mapping fields are deeply immutable, and nothing here may contain secrets.

A logical route maps to a ``provider:model`` pair. Only ``ollama`` is a known
provider in this milestone. Fallbacks are single-hop route mappings used when
a provider is unavailable; they never alter the routing decision itself.

Milestone 7 adds ``[profiles]``, ``[constraints]``, and ``[health]`` TOML
sections for cost-aware routing, latency-aware routing, and health checks.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from routellm.domain.routes import ROUTES

KNOWN_PROVIDERS = frozenset({"ollama"})

DEFAULT_PROVIDER = "ollama"
DEFAULT_MODEL = "qwen2.5-coder:3b"

DEFAULT_ROUTES = {
    "coding-local": (DEFAULT_PROVIDER, DEFAULT_MODEL),
    "general-local": (DEFAULT_PROVIDER, DEFAULT_MODEL),
    "reasoning": (DEFAULT_PROVIDER, DEFAULT_MODEL),
    "translation": (DEFAULT_PROVIDER, DEFAULT_MODEL),
}

DEFAULT_FALLBACKS = {
    "coding-local": "general-local",
}


def _require_http_host(host: object) -> str:
    if not isinstance(host, str):
        raise ValueError(f"host must be a string, got {type(host).__name__}.")
    if not host:
        raise ValueError("host must be a non-empty string.")
    if not (host.startswith("http://") or host.startswith("https://")):
        raise ValueError(f"host must start with http:// or https://, got {host!r}.")
    return host.rstrip("/")


@dataclass(frozen=True)
class OllamaConfig:
    """Settings for the local Ollama server and its request timeouts.

    ``temperature`` and ``num_predict`` are optional generation options sent to
    the Ollama API only when set; ``None`` means "use the server default".
    """

    host: str = "http://localhost:11434"
    timeout_ping: float = 2.0
    timeout_generate: float = 300.0
    temperature: float | None = None
    num_predict: int | None = None

    def __post_init__(self) -> None:
        host = _require_http_host(self.host)
        if not isinstance(self.timeout_ping, (int, float)):
            raise ValueError(
                f"timeout_ping must be a number, got {type(self.timeout_ping).__name__}."
            )
        if self.timeout_ping <= 0:
            raise ValueError(f"timeout_ping must be > 0, got {self.timeout_ping}.")
        if not isinstance(self.timeout_generate, (int, float)):
            raise ValueError(
                f"timeout_generate must be a number, got {type(self.timeout_generate).__name__}."
            )
        if self.timeout_generate <= 0:
            raise ValueError(f"timeout_generate must be > 0, got {self.timeout_generate}.")
        if self.temperature is not None and not isinstance(self.temperature, (int, float)):
            raise ValueError(
                f"temperature must be a number or None, got {type(self.temperature).__name__}."
            )
        if self.temperature is not None and not 0 <= self.temperature <= 2:
            raise ValueError(
                f"temperature must be in [0, 2] or None, got {self.temperature}."
            )
        if self.num_predict is not None and not isinstance(self.num_predict, int):
            raise ValueError(
                f"num_predict must be an integer or None, got {type(self.num_predict).__name__}."
            )
        if self.num_predict is not None and self.num_predict < -1:
            raise ValueError(
                f"num_predict must be >= -1 or None, got {self.num_predict}."
            )
        object.__setattr__(self, "host", host)


KNOWN_CAPABILITIES = frozenset(
    {"code", "reasoning", "qa", "translation", "math", "creative", "summarization"}
)


@dataclass(frozen=True)
class HealthConfig:
    """Settings for provider health checks."""

    timeout: float = 5.0

    def __post_init__(self) -> None:
        if not isinstance(self.timeout, (int, float)):
            raise ValueError(
                f"health timeout must be a number, got {type(self.timeout).__name__}."
            )
        if self.timeout <= 0:
            raise ValueError(f"health timeout must be > 0, got {self.timeout}.")


@dataclass(frozen=True)
class RouteProfile:
    """Per-route cost, latency, and capability metadata.

    ``cost_per_1k_tokens`` is the estimated cost in USD per 1000 tokens.
    ``estimated_latency_ms`` is the expected response time in milliseconds.
    ``capabilities`` is the set of capability tags this route's model supports.
    """

    cost_per_1k_tokens: float | None = None
    estimated_latency_ms: float | None = None
    capabilities: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.cost_per_1k_tokens is not None and self.cost_per_1k_tokens < 0:
            raise ValueError(
                f"cost_per_1k_tokens must be >= 0, got {self.cost_per_1k_tokens}."
            )
        if self.estimated_latency_ms is not None and self.estimated_latency_ms < 0:
            raise ValueError(
                f"estimated_latency_ms must be >= 0, got {self.estimated_latency_ms}."
            )
        unknown = self.capabilities - KNOWN_CAPABILITIES
        if unknown:
            raise ValueError(
                f"Unknown capabilities {sorted(unknown)!r}; "
                f"known capabilities: {sorted(KNOWN_CAPABILITIES)!r}."
            )


@dataclass(frozen=True)
class RoutingConstraints:
    """Global routing constraints for cost-aware and latency-aware routing.

    ``max_cost_per_prompt`` is the maximum allowed estimated cost per prompt
    in USD. ``max_latency_ms`` is the maximum allowed provider latency in
    milliseconds. None means "no constraint".
    """

    max_cost_per_prompt: float | None = None
    max_latency_ms: float | None = None

    def __post_init__(self) -> None:
        if self.max_cost_per_prompt is not None and self.max_cost_per_prompt < 0:
            raise ValueError(
                f"max_cost_per_prompt must be >= 0, got {self.max_cost_per_prompt}."
            )
        if self.max_latency_ms is not None and self.max_latency_ms < 0:
            raise ValueError(
                f"max_latency_ms must be >= 0, got {self.max_latency_ms}."
            )


@dataclass(frozen=True)
class ProviderConfig:
    """Validated route-to-provider mapping and provider-unavailability fallbacks.

    ``routes`` maps a logical route to a ``(provider, model)`` pair and
    ``fallbacks`` maps a route to another route used when its provider is
    unavailable. A fallback must point at a route that has a provider, must not
    point at itself, and must not chain (a fallback target cannot itself have a
    fallback), so the execution path is always at most one hop.
    """

    routes: dict[str, tuple[str, str]] | None = None
    fallbacks: dict[str, str] | None = None
    ollama: OllamaConfig | None = None
    profiles: dict[str, RouteProfile] | None = None
    constraints: RoutingConstraints | None = None
    health: HealthConfig | None = None

    def __post_init__(self) -> None:
        routes = dict(self.routes if self.routes is not None else DEFAULT_ROUTES)
        fallbacks = dict(self.fallbacks if self.fallbacks is not None else DEFAULT_FALLBACKS)
        ollama = self.ollama if self.ollama is not None else OllamaConfig()
        profiles = dict(self.profiles) if self.profiles is not None else {}
        constraints = self.constraints if self.constraints is not None else RoutingConstraints()
        health = self.health if self.health is not None else HealthConfig()
        if not isinstance(ollama, OllamaConfig):
            raise ValueError("ollama must be an OllamaConfig instance.")
        if not isinstance(constraints, RoutingConstraints):
            raise ValueError("constraints must be a RoutingConstraints instance.")
        if not isinstance(health, HealthConfig):
            raise ValueError("health must be a HealthConfig instance.")
        for route, (provider, model) in routes.items():
            if route not in ROUTES:
                raise ValueError(f"Unknown route {route!r} in provider config.")
            if provider not in KNOWN_PROVIDERS:
                raise ValueError(f"Unknown provider {provider!r} for route {route!r}.")
            if not model or not model.strip():
                raise ValueError(f"Model must be a non-empty name for route {route!r}.")
        for source, target in fallbacks.items():
            if not isinstance(target, str):
                raise ValueError(
                    f"Invalid fallback target for {source!r}; "
                    f"expected a route name, got {target!r}."
                )
            if source not in ROUTES:
                raise ValueError(f"Unknown fallback source route {source!r}.")
            if target not in ROUTES:
                raise ValueError(f"Unknown fallback target route {target!r}.")
            if source == target:
                raise ValueError(f"Fallback must not point at itself: {source!r}.")
            if target not in routes:
                raise ValueError(
                    f"Fallback target {target!r} for {source!r} has no configured provider."
                )
            if target in fallbacks:
                raise ValueError(
                    f"Fallback chains are not allowed: {source!r} -> {target!r} "
                    "but the target has its own fallback."
                )
        for route in profiles:
            if route not in ROUTES:
                raise ValueError(f"Unknown route {route!r} in profiles config.")
        object.__setattr__(self, "routes", MappingProxyType(routes))
        object.__setattr__(self, "fallbacks", MappingProxyType(fallbacks))
        object.__setattr__(self, "ollama", ollama)
        object.__setattr__(self, "profiles", MappingProxyType(profiles))
        object.__setattr__(self, "constraints", constraints)
        object.__setattr__(self, "health", health)


def _parse_provider_model(value: str, route: str) -> tuple[str, str]:
    if not isinstance(value, str) or ":" not in value:
        raise ValueError(
            f"Provider entry for route {route!r} must be 'provider:model', got {value!r}."
        )
    provider, model = value.split(":", 1)
    if not provider or not model.strip():
        raise ValueError(
            f"Provider entry for route {route!r} needs a provider and a model, got {value!r}."
        )
    return provider, model


def _require_table(data: dict, key: str, file_path: Path) -> dict:
    """Return the section for ``key``, rejecting any non-table TOML value."""
    section = data.get(key, {})
    if not isinstance(section, dict):
        raise ValueError(
            f"Invalid TOML in {file_path}: section [{key}] must be a table, "
            f"got {type(section).__name__}."
        )
    return section


def load_provider_config(path: str | None = None) -> ProviderConfig:
    """Load provider configuration, optionally overridden by a TOML file.

    With no ``path`` the validated defaults are returned. Otherwise the TOML
    file may define ``[ollama]`` (host, timeouts, temperature, num_predict),
    ``[routes]`` (``route = "provider:model"``), ``[fallbacks]``
    (``route = "fallback-route"``), ``[profiles.<route>]`` (cost, latency,
    capabilities), ``[constraints]`` (max_cost_per_prompt, max_latency_ms),
    and ``[health]`` (timeout); each section must be a table. Keys present
    in the file override the defaults and the whole result is validated.
    """
    if path is None:
        return ProviderConfig()
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"No provider configuration found at {file_path}.")
    try:
        with file_path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"Invalid TOML in {file_path}: {error}") from error
    routes = dict(DEFAULT_ROUTES)
    table = _require_table(data, "routes", file_path)
    for route, value in table.items():
        routes[route] = _parse_provider_model(value, route)
    fallbacks = dict(DEFAULT_FALLBACKS)
    table = _require_table(data, "fallbacks", file_path)
    fallbacks.update(table)
    table = _require_table(data, "ollama", file_path)
    ollama_kwargs = dict(table)
    known = {"host", "timeout_ping", "timeout_generate", "temperature", "num_predict"}
    unknown = set(ollama_kwargs) - known
    if unknown:
        raise ValueError(
            f"Unknown [ollama] option(s) {sorted(unknown)!r}; expected one of {sorted(known)!r}."
        )
    profiles: dict[str, RouteProfile] = {}
    profiles_table = _require_table(data, "profiles", file_path)
    for route, profile_data in profiles_table.items():
        if not isinstance(profile_data, dict):
            raise ValueError(
                f"Invalid TOML in {file_path}: [profiles.{route}] must be a table, "
                f"got {type(profile_data).__name__}."
            )
        caps = profile_data.get("capabilities", [])
        if not isinstance(caps, list):
            raise ValueError(
                f"Invalid TOML in {file_path}: capabilities for {route!r} must be a list."
            )
        profiles[route] = RouteProfile(
            cost_per_1k_tokens=profile_data.get("cost_per_1k_tokens"),
            estimated_latency_ms=profile_data.get("estimated_latency_ms"),
            capabilities=frozenset(caps),
        )
    constraints_data = _require_table(data, "constraints", file_path)
    constraints = RoutingConstraints(
        max_cost_per_prompt=constraints_data.get("max_cost_per_prompt"),
        max_latency_ms=constraints_data.get("max_latency_ms"),
    )
    health_data = _require_table(data, "health", file_path)
    health = HealthConfig(timeout=health_data.get("timeout", 5.0))
    return ProviderConfig(
        routes=routes,
        fallbacks=fallbacks,
        ollama=OllamaConfig(**ollama_kwargs),
        profiles=profiles or None,
        constraints=constraints,
        health=health,
    )
