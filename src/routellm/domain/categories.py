"""Category taxonomy for RouteLLM.

Categories are ordered by specificity. Earlier categories win signal-count
ties, so specific categories beat the more general ones.
"""

CATEGORY_PRIORITY = (
    "coding",
    "math",
    "translation",
    "summarization",
    "creative_writing",
    "general_qa",
    "unknown",
)

CATEGORIES = frozenset(CATEGORY_PRIORITY)
