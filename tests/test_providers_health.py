"""Tests for provider health checks (Milestone 7)."""

import json

from routellm.application.execution_service import (
    ExecutionService,
    load_health_history,
    save_health_check,
)
from routellm.configuration.providers import ProviderConfig
from routellm.domain.provider import HealthCheckResult


def _json_body(data: dict) -> bytes:
    return json.dumps(data).encode("utf-8")


def test_health_check_returns_results_for_all_routes():
    def request(method, url, payload, timeout):
        return 200, _json_body({"models": [{"name": "qwen2.5-coder:3b"}]})

    service = ExecutionService(
        config=ProviderConfig(),
        adapter_factory=lambda config: __import__(
            "routellm.providers.ollama", fromlist=["OllamaAdapter"]
        ).OllamaAdapter(config=config, request=request),
    )
    results = service.health_check()
    assert len(results) == 6
    routes = [r.route for r in results]
    assert "coding-local" in routes
    assert "calculator" in routes


def test_health_check_available_model():
    def request(method, url, payload, timeout):
        return 200, _json_body({"models": [{"name": "qwen2.5-coder:3b"}]})

    service = ExecutionService(
        config=ProviderConfig(),
        adapter_factory=lambda config: __import__(
            "routellm.providers.ollama", fromlist=["OllamaAdapter"]
        ).OllamaAdapter(config=config, request=request),
    )
    results = service.health_check()
    coding = next(r for r in results if r.route == "coding-local")
    assert coding.available is True
    assert coding.response_time_ms is not None
    assert coding.error is None


def test_health_check_unavailable_model():
    def request(method, url, payload, timeout):
        return 200, _json_body({"models": [{"name": "other-model"}]})

    service = ExecutionService(
        config=ProviderConfig(),
        adapter_factory=lambda config: __import__(
            "routellm.providers.ollama", fromlist=["OllamaAdapter"]
        ).OllamaAdapter(config=config, request=request),
    )
    results = service.health_check()
    coding = next(r for r in results if r.route == "coding-local")
    assert coding.available is False
    assert "not found" in coding.error


def test_health_check_server_down():
    from routellm.providers.ollama import OllamaError

    def request(method, url, payload, timeout):
        raise OllamaError("Could not reach Ollama")

    service = ExecutionService(
        config=ProviderConfig(),
        adapter_factory=lambda config: __import__(
            "routellm.providers.ollama", fromlist=["OllamaAdapter"]
        ).OllamaAdapter(config=config, request=request),
    )
    results = service.health_check()
    coding = next(r for r in results if r.route == "coding-local")
    assert coding.available is None
    assert "Could not reach" in coding.error


def test_health_check_unconfigured_route():
    service = ExecutionService(config=ProviderConfig())
    results = service.health_check()
    calc = next(r for r in results if r.route == "calculator")
    assert calc.available is None
    assert calc.provider is None
    assert "No provider configured" in calc.error


def test_save_health_check_creates_file(tmp_path):
    results = (
        HealthCheckResult(
            route="coding-local",
            provider="ollama",
            model="qwen2.5-coder:3b",
            available=True,
            response_time_ms=45.2,
            checked_at="2026-08-17T12:00:00+00:00",
            error=None,
        ),
    )
    path = save_health_check(results, directory=tmp_path)
    assert path.exists()
    assert path.suffix == ".json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data) == 1
    assert data[0]["route"] == "coding-local"
    assert data[0]["available"] is True


def test_load_health_history_returns_most_recent(tmp_path):
    (tmp_path / "2026-08-17T10:00:00.json").write_text(
        json.dumps([
            {"route": "coding-local", "provider": "ollama", "model": "m",
             "available": True, "response_time_ms": 50.0,
             "checked_at": "2026-08-17T10:00:00+00:00", "error": None}
        ]),
        encoding="utf-8",
    )
    (tmp_path / "2026-08-17T12:00:00.json").write_text(
        json.dumps([
            {"route": "coding-local", "provider": "ollama", "model": "m",
             "available": False, "response_time_ms": 60.0,
             "checked_at": "2026-08-17T12:00:00+00:00", "error": "not found"}
        ]),
        encoding="utf-8",
    )
    results = load_health_history(directory=tmp_path)
    assert len(results) == 1
    assert results[0].available is False
    assert results[0].checked_at == "2026-08-17T12:00:00+00:00"


def test_load_health_history_empty_directory(tmp_path):
    results = load_health_history(directory=tmp_path)
    assert results == ()


def test_load_health_history_no_directory(tmp_path):
    missing = tmp_path / "nonexistent"
    results = load_health_history(directory=missing)
    assert results == ()


def test_health_check_uses_configured_timeout():
    from routellm.configuration.providers import HealthConfig, ProviderConfig

    captured_timeouts = []

    def request(method, url, payload, timeout):
        captured_timeouts.append(timeout)
        return 200, _json_body({"models": [{"name": "qwen2.5-coder:3b"}]})

    config = ProviderConfig(health=HealthConfig(timeout=10.0))
    service = ExecutionService(
        config=config,
        adapter_factory=lambda ollama_cfg: __import__(
            "routellm.providers.ollama", fromlist=["OllamaAdapter"]
        ).OllamaAdapter(config=ollama_cfg, request=request),
    )
    results = service.health_check()
    configured_routes = [r for r in results if r.provider is not None]
    assert len(configured_routes) > 0
    assert all(t == 10.0 for t in captured_timeouts)
