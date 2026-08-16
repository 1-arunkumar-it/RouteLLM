"""Tests for the provider registry (Milestone 6, extended in Milestone 7)."""

from routellm.configuration.providers import ProviderConfig, RouteProfile
from routellm.domain.provider import ResolvedProvider
from routellm.providers.registry import ProviderRegistry


def test_resolve_returns_configured_provider():
    registry = ProviderRegistry(config=ProviderConfig())
    resolved = registry.resolve("coding-local")
    assert resolved == ResolvedProvider(
        route="coding-local",
        provider="ollama",
        model="qwen2.5-coder:3b",
    )


def test_resolve_unconfigured_route_returns_none():
    registry = ProviderRegistry(config=ProviderConfig())
    assert registry.resolve("calculator") is None
    assert registry.resolve("fallback") is None


def test_fallback_for_returns_configured_fallback():
    registry = ProviderRegistry(config=ProviderConfig())
    assert registry.fallback_for("coding-local") == "general-local"
    assert registry.fallback_for("general-local") is None


def test_configured_routes_returns_configuration_order():
    registry = ProviderRegistry(config=ProviderConfig())
    assert registry.configured_routes() == (
        "coding-local",
        "general-local",
        "reasoning",
        "translation",
    )


# --- Milestone 7: Profile lookup tests ---


def test_get_profile_returns_configured_profile():
    profile = RouteProfile(cost_per_1k_tokens=0.002, estimated_latency_ms=150)
    config = ProviderConfig(profiles={"coding-local": profile})
    registry = ProviderRegistry(config=config)
    result = registry.get_profile("coding-local")
    assert result is profile


def test_get_profile_returns_none_for_unconfigured():
    registry = ProviderRegistry(config=ProviderConfig())
    assert registry.get_profile("coding-local") is None
    assert registry.get_profile("calculator") is None


def test_get_profile_returns_none_when_no_profiles():
    registry = ProviderRegistry(config=ProviderConfig())
    assert registry.get_profile("reasoning") is None
