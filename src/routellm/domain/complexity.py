"""Structured result of a lightweight complexity estimation.

Milestone 5 adds an ordinal complexity estimate (``low``/``medium``/``high``)
that selects the route only; it never changes the category decision. The
estimate is heuristic, not an LLM judgment, and must never be presented as a
measured ground truth (SPEC section 41 and ROADMAP Milestone 5).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ComplexityEstimate:
    """The heuristic complexity of one prompt.

    ``level`` is one of the configured ordinal levels (``low``, ``medium``,
    ``high`` by default). ``score`` is the raw composite heuristic score in
    [0, 1] before thresholding, kept for transparency. ``signals`` lists the
    per-signal explanations that contributed to the score, so the estimate
    stays inspectable and truthful.
    """

    text: str
    level: str
    score: float
    signals: tuple[str, ...] = ()
