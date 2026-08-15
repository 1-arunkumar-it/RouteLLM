"""Logical routes for RouteLLM.

A route is a logical label, not an instruction to execute a model. Providers
resolve these labels only from Milestone 6 onward.
"""

CATEGORY_ROUTES = {
    "coding": "coding-local",
    "math": "calculator",
    "translation": "translation",
    "summarization": "general-local",
    "creative_writing": "general-local",
    "general_qa": "general-local",
    "unknown": "fallback",
}

ROUTES = frozenset(CATEGORY_ROUTES.values())
