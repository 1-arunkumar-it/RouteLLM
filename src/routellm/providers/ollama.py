"""Ollama provider adapter (Milestone 6, extended in Milestone 7).

Talks to a local Ollama server over its REST API using only the Python
standard library (``urllib``), so no dependency is added. Prompt contents are
never logged. The HTTP seam ``request(method, url, payload, timeout)`` returns
 ``(status, body)`` and raises ``OllamaError`` on connection failures, which
lets the normal test suite run fully offline with injected fakes.

Milestone 7 adds ``check_health()`` for provider health checks.
"""

import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from routellm.configuration.providers import OllamaConfig
from routellm.domain.provider import HealthCheckResult


class OllamaError(Exception):
    """Raised when Ollama cannot be reached or returns an unexpected result."""


def default_request(
    method: str, url: str, payload: dict | None, timeout: float
) -> tuple[int, bytes]:
    """Perform one HTTP request with ``urllib`` and return ``(status, body)``."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise OllamaError(f"Could not reach Ollama at {url}: {error}") from error


def _parse_json(body: bytes, url: str) -> dict:
    try:
        data = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as error:
        raise OllamaError(f"Could not parse Ollama response from {url}: {error}") from error
    if not isinstance(data, dict):
        raise OllamaError(f"Ollama response from {url} was not a JSON object.")
    return data


class OllamaAdapter:
    """Minimal client for the Ollama REST API.

    ``request`` is an injectable HTTP seam; the default uses ``urllib`` and the
    config's timeouts. Availability is defined as the configured model being
    listed by the server (SPEC section 24 treats Ollama as a provider).
    """

    def __init__(
        self,
        config: OllamaConfig,
        request: callable = default_request,
    ) -> None:
        self._config = config
        self._request = request

    def _endpoint(self, path: str) -> str:
        return f"{self._config.host}{path}"

    def list_models(self) -> tuple[str, ...]:
        """Return the model names the Ollama server reports via ``/api/tags``."""
        url = self._endpoint("/api/tags")
        status, body = self._request("GET", url, None, self._config.timeout_ping)
        if status != 200:
            raise OllamaError(f"Ollama returned HTTP {status} for {url}.")
        data = _parse_json(body, url)
        return tuple(
            name for name in (entry.get("name") for entry in data.get("models", [])) if name
        )

    def has_model(self, model: str) -> bool:
        """Return True when the server currently lists ``model``."""
        try:
            return model in self.list_models()
        except OllamaError:
            return False

    def available(self) -> bool:
        """Return True when the server responds to a model listing."""
        try:
            self.list_models()
            return True
        except OllamaError:
            return False

    def generate(self, model: str, prompt: str) -> str:
        """Generate a non-streaming completion and return the response text."""
        url = self._endpoint("/api/generate")
        options = {}
        if self._config.temperature is not None:
            options["temperature"] = self._config.temperature
        if self._config.num_predict is not None:
            options["num_predict"] = self._config.num_predict
        payload: dict = {"model": model, "prompt": prompt, "stream": False}
        if options:
            payload["options"] = options
        status, body = self._request("POST", url, payload, self._config.timeout_generate)
        if status != 200:
            raise OllamaError(f"Ollama returned HTTP {status} for {url}.")
        data = _parse_json(body, url)
        text = data.get("response")
        if not isinstance(text, str) or not text:
            raise OllamaError(f"Ollama response from {url} contained no text.")
        return text

    def check_health(
        self, route: str, provider: str, model: str, *, timeout: float | None = None
    ) -> HealthCheckResult:
        """Check whether the server is up and the model is available.

        Returns a ``HealthCheckResult`` with timing and availability data.
        The server is considered down only when ``/api/tags`` itself fails;
        if the server responds but the model is missing, ``available`` is
        False rather than None.

        ``timeout`` overrides the default ping timeout for this check.
        """
        now = datetime.now(timezone.utc).isoformat()
        effective_timeout = timeout if timeout is not None else self._config.timeout_ping
        start = time.perf_counter()
        try:
            url = self._endpoint("/api/tags")
            status, body = self._request("GET", url, None, effective_timeout)
            if status != 200:
                raise OllamaError(f"Ollama returned HTTP {status} for {url}.")
            data = _parse_json(body, url)
            models = tuple(
                name
                for name in (entry.get("name") for entry in data.get("models", []))
                if name
            )
            elapsed = (time.perf_counter() - start) * 1000
            available = model in models
            return HealthCheckResult(
                route=route,
                provider=provider,
                model=model,
                available=available,
                response_time_ms=elapsed,
                checked_at=now,
                error=None if available else f"Model {model!r} not found on server.",
            )
        except OllamaError as error:
            elapsed = (time.perf_counter() - start) * 1000
            return HealthCheckResult(
                route=route,
                provider=provider,
                model=model,
                available=None,
                response_time_ms=elapsed,
                checked_at=now,
                error=str(error),
            )
