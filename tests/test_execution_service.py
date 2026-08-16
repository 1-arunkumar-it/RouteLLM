"""Tests for the execution service (Milestone 6)."""

import pytest

from routellm.application.execution_service import ExecutionService
from routellm.configuration.providers import ProviderConfig
from routellm.domain.route_decision import RouteDecision
from routellm.providers.ollama import OllamaError


class FakeAdapter:
    """In-memory stand-in for OllamaAdapter following the same public methods."""

    def __init__(self, models=(), generated="", list_error=False, generate_error=False):
        self.models = set(models)
        self.generated = generated
        self.list_error = list_error
        self.generate_error = generate_error
        self.calls = []

    def has_model(self, model):
        self.calls.append(("has_model", model))
        return model in self.models

    def list_models(self):
        if self.list_error:
            raise OllamaError("server down")
        return tuple(sorted(self.models))

    def generate(self, model, prompt):
        self.calls.append(("generate", model, prompt))
        if self.generate_error:
            raise OllamaError("generation failed")
        return self.generated


def _decision(route, prompt="Write a Python function"):
    return RouteDecision(prompt=prompt, category="coding", route=route)


def _service(adapter, config=None):
    return ExecutionService(
        config=config or ProviderConfig(),
        adapter_factory=lambda ollama_config: adapter,
    )


def test_execute_returns_ok_on_success():
    adapter = FakeAdapter(models={"qwen2.5-coder:3b"}, generated="def f(): pass")
    response = _service(adapter).execute(_decision("coding-local"))
    assert response.status == "ok"
    assert response.requested_route == "coding-local"
    assert response.route == "coding-local"
    assert response.provider == "ollama"
    assert response.model == "qwen2.5-coder:3b"
    assert response.text == "def f(): pass"
    assert response.error == ""
    assert response.latency_ms is not None
    assert adapter.calls[0] == ("has_model", "qwen2.5-coder:3b")
    assert adapter.calls[1] == ("generate", "qwen2.5-coder:3b", "Write a Python function")


def test_execute_unconfigured_route_reports_not_configured():
    adapter = FakeAdapter()
    response = _service(adapter).execute(_decision("calculator"))
    assert response.status == "not_configured"
    assert response.provider is None
    assert response.text == ""
    assert "No provider is configured" in response.error


def test_execute_unavailable_without_fallback():
    config = ProviderConfig(fallbacks={})
    adapter = FakeAdapter(models=set())
    response = _service(adapter, config).execute(_decision("coding-local"))
    assert response.status == "unavailable"
    assert response.route == "coding-local"
    assert "no fallback route is configured" in response.error


def test_execute_unavailable_uses_fallback():
    config = ProviderConfig(
        routes={
            "coding-local": ("ollama", "coding-model"),
            "general-local": ("ollama", "general-model"),
        }
    )
    adapter = FakeAdapter(models={"general-model"}, generated="fallback answer")
    response = _service(adapter, config).execute(_decision("coding-local"))
    assert response.status == "ok"
    assert response.requested_route == "coding-local"
    assert response.route == "general-local"
    assert response.text == "fallback answer"
    assert "fallback route 'general-local'" in response.error


def test_execute_fallback_also_unavailable():
    adapter = FakeAdapter(models={})
    response = _service(adapter).execute(_decision("coding-local"))
    assert response.status == "unavailable"
    assert response.route == "general-local"
    assert "fallback route 'general-local' is also unavailable" in response.error


def test_execute_generation_error_reports_error():
    adapter = FakeAdapter(models={"qwen2.5-coder:3b"}, generate_error=True)
    response = _service(adapter).execute(_decision("coding-local"))
    assert response.status == "error"
    assert response.route == "coding-local"
    assert "generation failed" in response.error


def test_status_table_lists_all_routes():
    adapter = FakeAdapter(models={"qwen2.5-coder:3b"})
    rows = _service(adapter).status_table()
    routes = [row.route for row in rows]
    assert routes == [
        "coding-local",
        "calculator",
        "translation",
        "general-local",
        "reasoning",
        "fallback",
    ]
    by_route = {row.route: row for row in rows}
    assert by_route["coding-local"].available is True
    assert by_route["translation"].available is True
    assert by_route["calculator"].provider is None
    assert by_route["calculator"].available is None


def test_status_table_marks_unavailable_when_server_down():
    adapter = FakeAdapter(models={"qwen2.5-coder:3b"}, list_error=True)
    rows = {row.route: row for row in _service(adapter).status_table()}
    assert rows["coding-local"].available is False


def test_service_rejects_non_provider_config():
    with pytest.raises(ValueError, match="ProviderConfig"):
        ExecutionService(config={"routes": {}})


# --- Milestone 7: Constraint application tests ---


def test_execute_applies_cost_constraints_and_reroutes():
    from routellm.configuration.providers import ProviderConfig, RouteProfile, RoutingConstraints

    config = ProviderConfig(
        routes={
            "coding-local": ("ollama", "coding-model"),
            "general-local": ("ollama", "general-model"),
        },
        profiles={
            "coding-local": RouteProfile(
                cost_per_1k_tokens=0.1, estimated_latency_ms=150,
                capabilities=frozenset({"code"}),
            ),
            "general-local": RouteProfile(
                cost_per_1k_tokens=0.001, estimated_latency_ms=100,
                capabilities=frozenset({"code", "qa"}),
            ),
        },
        constraints=RoutingConstraints(max_cost_per_prompt=0.001),
    )
    adapter = FakeAdapter(
        models={"coding-model", "general-model"}, generated="rerouted"
    )
    prompt = " ".join(["word"] * 100)
    response = _service(adapter, config).execute(
        _decision("coding-local", prompt=prompt)
    )
    assert response.status == "ok"
    assert response.requested_route == "coding-local"
    assert response.route == "general-local"


def test_execute_does_not_reroute_when_no_profiles():
    adapter = FakeAdapter(models={"qwen2.5-coder:3b"}, generated="ok")
    response = _service(adapter).execute(_decision("coding-local"))
    assert response.status == "ok"
    assert response.route == "coding-local"


def test_execute_does_not_reroute_when_no_constraints():
    from routellm.configuration.providers import ProviderConfig, RouteProfile

    config = ProviderConfig(
        profiles={
            "coding-local": RouteProfile(cost_per_1k_tokens=0.01),
        },
    )
    adapter = FakeAdapter(models={"qwen2.5-coder:3b"}, generated="ok")
    response = _service(adapter, config).execute(_decision("coding-local"))
    assert response.status == "ok"
    assert response.route == "coding-local"
