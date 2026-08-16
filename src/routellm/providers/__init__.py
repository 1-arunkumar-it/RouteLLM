"""Provider resolution and execution for logical routes (Milestone 6)."""

from routellm.providers.ollama import OllamaAdapter, OllamaError
from routellm.providers.registry import ProviderRegistry

__all__ = ["OllamaAdapter", "OllamaError", "ProviderRegistry"]
