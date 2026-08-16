"""Logical routes for RouteLLM.

A route is a logical label, not an instruction to execute a model. Providers
resolve these labels only from Milestone 6 onward.

``COMPLEXITY_LEVELS`` is the ordinal scale introduced in Milestone 5.
``COMPLEXITY_REROUTES`` maps (category, level) pairs to a different logical
route: high-complexity general questions and summarization requests route to
``reasoning`` while every other pair keeps its category default. Complexity
never changes the category decision and ``unknown`` is never re-routed.
"""

COMPLEXITY_LEVELS = ("low", "medium", "high")

CATEGORY_ROUTES = {
    "coding": "coding-local",
    "math": "calculator",
    "translation": "translation",
    "summarization": "general-local",
    "creative_writing": "general-local",
    "general_qa": "general-local",
    "unknown": "fallback",
}

COMPLEXITY_REROUTES = {
    "general_qa": {"high": "reasoning"},
    "summarization": {"high": "reasoning"},
}

ROUTES = frozenset({*CATEGORY_ROUTES.values(), "reasoning"})
