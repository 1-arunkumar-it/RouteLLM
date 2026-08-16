"""Tests for validated provider configuration (Milestone 6)."""

import pytest

from routellm.configuration.providers import (
    DEFAULT_FALLBACKS,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    OllamaConfig,
    ProviderConfig,
    load_provider_config,
)
from routellm.domain.routes import ROUTES


def test_default_config_is_valid_and_known_routes_only():
    config = ProviderConfig()
    assert set(config.routes) == {"coding-local", "general-local", "reasoning", "translation"}
    assert set(config.routes) <= ROUTES
    for route, (provider, model) in config.routes.items():
        assert provider == DEFAULT_PROVIDER
        assert model == DEFAULT_MODEL
    assert dict(config.fallbacks) == dict(DEFAULT_FALLBACKS)
    assert config.ollama.host == "http://localhost:11434"


def test_config_is_deeply_immutable():
    config = ProviderConfig()
    with pytest.raises(TypeError):
        config.routes["coding-local"] = ("ollama", "other")
    with pytest.raises(TypeError):
        config.fallbacks["coding-local"] = "reasoning"


def test_config_copies_passed_mappings():
    routes = {"general-local": ("ollama", "some-model")}
    config = ProviderConfig(routes=routes)
    routes["general-local"] = ("ollama", "other")
    assert config.routes["general-local"] == ("ollama", "some-model")


def test_unknown_route_is_rejected():
    with pytest.raises(ValueError, match="Unknown route"):
        ProviderConfig(routes={"not-a-route": ("ollama", "m")})


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="Unknown provider"):
        ProviderConfig(routes={"general-local": ("remote", "m")})


def test_empty_model_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        ProviderConfig(routes={"general-local": ("ollama", " ")})


def test_fallback_to_unknown_route_is_rejected():
    with pytest.raises(ValueError, match="Unknown fallback"):
        ProviderConfig(fallbacks={"coding-local": "not-a-route"})


def test_self_fallback_is_rejected():
    with pytest.raises(ValueError, match="at itself"):
        ProviderConfig(fallbacks={"coding-local": "coding-local"})


def test_fallback_to_unconfigured_route_is_rejected():
    with pytest.raises(ValueError, match="no configured provider"):
        ProviderConfig(
            routes={"general-local": ("ollama", "m")},
            fallbacks={"coding-local": "reasoning"},
        )


def test_chained_fallback_is_rejected():
    with pytest.raises(ValueError, match="Fallback chains are not allowed"):
        ProviderConfig(fallbacks={"coding-local": "general-local", "general-local": "reasoning"})


def test_ollama_config_validation():
    with pytest.raises(ValueError, match="host"):
        OllamaConfig(host="")
    with pytest.raises(ValueError, match="http"):
        OllamaConfig(host="ftp://localhost")
    with pytest.raises(ValueError, match="timeout_ping"):
        OllamaConfig(timeout_ping=0)
    with pytest.raises(ValueError, match="timeout_generate"):
        OllamaConfig(timeout_generate=-1)
    with pytest.raises(ValueError, match="temperature"):
        OllamaConfig(temperature=3.0)
    with pytest.raises(ValueError, match="num_predict"):
        OllamaConfig(num_predict=-2)


@pytest.mark.parametrize(
    "kwargs, expected_field",
    [
        ({"host": 123}, "host"),
        ({"timeout_ping": "fast"}, "timeout_ping"),
        ({"timeout_generate": "fast"}, "timeout_generate"),
        ({"temperature": "warm"}, "temperature"),
        ({"num_predict": "two"}, "num_predict"),
    ],
)
def test_ollama_config_rejects_wrong_types(kwargs, expected_field):
    with pytest.raises(ValueError, match=expected_field):
        OllamaConfig(**kwargs)


def test_ollama_config_wrong_types_via_toml(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text(
        "[ollama]\n"
        'host = 123\n'
        'timeout_ping = "fast"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="host"):
        load_provider_config(str(path))


def test_ollama_host_trailing_slash_is_normalized():
    assert OllamaConfig(host="http://localhost:11434/").host == "http://localhost:11434"


def test_load_provider_config_without_path_returns_defaults():
    config = load_provider_config()
    assert config.routes["coding-local"] == ("ollama", DEFAULT_MODEL)


def test_load_provider_config_overrides_toml(tmp_path):
    path = tmp_path / "providers.toml"
    path.write_text(
        "[ollama]\n"
        'host = "http://127.0.0.1:11434"\n'
        "timeout_ping = 1.0\n"
        "\n"
        "[routes]\n"
        'coding-local = "ollama:qwen2.5-coder:14b"\n'
        "\n"
        "[fallbacks]\n"
        'coding-local = "reasoning"\n',
        encoding="utf-8",
    )
    config = load_provider_config(str(path))
    assert config.ollama.host == "http://127.0.0.1:11434"
    assert config.ollama.timeout_ping == 1.0
    assert config.routes["coding-local"] == ("ollama", "qwen2.5-coder:14b")
    assert config.routes["reasoning"] == ("ollama", DEFAULT_MODEL)
    assert config.fallbacks["coding-local"] == "reasoning"


def test_load_provider_config_rejects_malformed_toml(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("this is = not [valid toml", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid TOML"):
        load_provider_config(str(path))


def test_load_provider_config_rejects_unregistered_provider(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text('[routes]\ncoding-local = "qwen2.5-coder:3b"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown provider 'qwen2.5-coder'"):
        load_provider_config(str(path))


def test_load_provider_config_rejects_unknown_ollama_option(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text('[ollama]\nport = 9999\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown \\[ollama\\]"):
        load_provider_config(str(path))


def test_load_provider_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_provider_config(str(tmp_path / "missing.toml"))


def test_fallback_value_must_be_a_string():
    with pytest.raises(ValueError, match="Invalid fallback target"):
        ProviderConfig(fallbacks={"coding-local": ["general-local"]})


def test_fallback_value_must_be_a_string_toml(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text(
        '[fallbacks]\ncoding-local = ["general-local"]\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Invalid fallback target"):
        load_provider_config(str(path))


@pytest.mark.parametrize(
    "body",
    [
        'routes = "not-a-table"\n',
        'fallbacks = "not-a-table"\n',
        'ollama = "not-a-table"\n',
        'fallbacks = "also-not-a-table"\n[routes]\ncoding-local = "ollama:model"\n',
    ],
)
def test_load_provider_config_rejects_non_table_sections(tmp_path, body):
    path = tmp_path / "bad.toml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError, match=r"section \[(routes|fallbacks|ollama)\] must be a table"):
        load_provider_config(str(path))
