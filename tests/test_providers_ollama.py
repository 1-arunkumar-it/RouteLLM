"""Tests for the Ollama adapter with an injected HTTP seam (Milestone 6)."""

import json

import pytest

from routellm.configuration.providers import OllamaConfig
from routellm.providers.ollama import OllamaAdapter, OllamaError


def _json_body(data: dict) -> bytes:
    return json.dumps(data).encode("utf-8")


def test_list_models_parses_names():
    def request(method, url, payload, timeout):
        assert method == "GET"
        assert url == "http://localhost:11434/api/tags"
        return 200, _json_body({"models": [{"name": "qwen2.5-coder:3b"}, {"name": "llama3"}]})

    adapter = OllamaAdapter(config=OllamaConfig(), request=request)
    assert adapter.list_models() == ("qwen2.5-coder:3b", "llama3")


def test_list_models_empty_when_no_models():
    def request(method, url, payload, timeout):
        return 200, _json_body({"models": []})

    adapter = OllamaAdapter(config=OllamaConfig(), request=request)
    assert adapter.list_models() == ()


def test_list_models_raises_on_http_error():
    def request(method, url, payload, timeout):
        return 500, b"boom"

    adapter = OllamaAdapter(config=OllamaConfig(), request=request)
    with pytest.raises(OllamaError, match="HTTP 500"):
        adapter.list_models()


def test_list_models_raises_on_connection_error():
    def request(method, url, payload, timeout):
        raise OllamaError("Could not reach Ollama")

    adapter = OllamaAdapter(config=OllamaConfig(), request=request)
    with pytest.raises(OllamaError, match="Could not reach Ollama"):
        adapter.list_models()


def test_list_models_raises_on_bad_json():
    def request(method, url, payload, timeout):
        return 200, b"not json"

    adapter = OllamaAdapter(config=OllamaConfig(), request=request)
    with pytest.raises(OllamaError, match="parse"):
        adapter.list_models()


def test_has_model_reflects_listing():
    def request(method, url, payload, timeout):
        return 200, _json_body({"models": [{"name": "qwen2.5-coder:3b"}]})

    adapter = OllamaAdapter(config=OllamaConfig(), request=request)
    assert adapter.has_model("qwen2.5-coder:3b")
    assert not adapter.has_model("other:model")


def test_has_model_false_on_connection_error():
    def request(method, url, payload, timeout):
        raise OllamaError("down")

    adapter = OllamaAdapter(config=OllamaConfig(), request=request)
    assert not adapter.has_model("anything")


def test_available_reports_server_health():
    def request(method, url, payload, timeout):
        return 200, _json_body({"models": []})

    adapter = OllamaAdapter(config=OllamaConfig(), request=request)
    assert adapter.available()


def test_generate_returns_response_text():
    captured = {}

    def request(method, url, payload, timeout):
        captured.update(method=method, url=url, payload=payload, timeout=timeout)
        return 200, _json_body({"response": "def f():\n    return 1\n"})

    adapter = OllamaAdapter(config=OllamaConfig(), request=request)
    text = adapter.generate("qwen2.5-coder:3b", "write python")
    assert text == "def f():\n    return 1\n"
    assert captured["method"] == "POST"
    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["payload"]["model"] == "qwen2.5-coder:3b"
    assert captured["payload"]["prompt"] == "write python"
    assert captured["payload"]["stream"] is False
    assert captured["timeout"] == OllamaConfig().timeout_generate


def test_generate_sends_options_when_configured():
    captured = {}

    def request(method, url, payload, timeout):
        captured["payload"] = payload
        return 200, _json_body({"response": "ok"})

    config = OllamaConfig(temperature=0.2, num_predict=64)
    adapter = OllamaAdapter(config=config, request=request)
    adapter.generate("m", "p")
    assert captured["payload"]["options"] == {"temperature": 0.2, "num_predict": 64}


def test_generate_omits_options_by_default():
    captured = {}

    def request(method, url, payload, timeout):
        captured["payload"] = payload
        return 200, _json_body({"response": "ok"})

    adapter = OllamaAdapter(config=OllamaConfig(), request=request)
    adapter.generate("m", "p")
    assert "options" not in captured["payload"]


def test_generate_raises_on_http_error():
    def request(method, url, payload, timeout):
        return 500, b"boom"

    adapter = OllamaAdapter(config=OllamaConfig(), request=request)
    with pytest.raises(OllamaError, match="HTTP 500"):
        adapter.generate("m", "p")


def test_generate_raises_on_missing_text():
    def request(method, url, payload, timeout):
        return 200, _json_body({"response": ""})

    adapter = OllamaAdapter(config=OllamaConfig(), request=request)
    with pytest.raises(OllamaError, match="no text"):
        adapter.generate("m", "p")
