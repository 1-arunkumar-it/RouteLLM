"""Validated configuration for complexity estimation.

Milestone 5 configuration is the single place where complexity knobs live.
Every value is validated at construction time and nothing here may contain
secrets. The blend weights, feature caps, and level thresholds are documented
design constants: they are measured against a hand-labeled evaluation set
rather than tuned on it, so the reported quality stays honest.
"""

from dataclasses import dataclass
from types import MappingProxyType

from routellm.domain.routes import COMPLEXITY_LEVELS

REASONING_INDICATORS = (
    "because",
    "therefore",
    "thus",
    "however",
    "although",
    "whereas",
    "unless",
    "prove",
    "prove that",
    "demonstrate",
    "compare",
    "contrast",
    "analyze",
    "analyse",
    "evaluate",
    "derive",
    "distinguish",
    "implication",
    "implications",
    "hypothesis",
    "conclusion",
    "explain",
    "assess",
    "justify",
    "argue",
    "investigate",
    "evidence",
    "mechanism",
    "consequence",
    "trade off",
    "tradeoffs",
    "between",
    "versus",
    "reasoning",
    "predict",
    "why",
    "how",
    "proving",
    "comparing",
    "analyzing",
    "evaluating",
    "assessing",
    "justifying",
    "arguing",
    "investigating",
    "demonstrating",
    "deriving",
    "explaining",
    "recommending",
)

OPERATION_INDICATORS = (
    "solve",
    "calculate",
    "compute",
    "implement",
    "design",
    "build",
    "create",
    "write",
    "debug",
    "refactor",
    "optimize",
    "convert",
    "synthesize",
    "generate",
    "fix",
    "test",
    "discuss",
    "assess",
    "justify",
    "propose",
    "argue",
    "research",
    "investigate",
    "plan",
    "identify",
    "outline",
    "derive",
    "prove",
    "summarize",
    "translate",
    "explain",
    "analyze",
    "compare",
    "recommend",
    "condense",
    "review",
    "deploy",
    "handle",
    "manage",
    "set up",
    "setting up",
    "solve",
    "writing",
    "building",
    "creating",
    "designing",
    "solving",
    "calculating",
    "computing",
    "implementing",
    "debugging",
    "refactoring",
    "optimizing",
    "converting",
    "synthesizing",
    "generating",
    "fixing",
    "testing",
    "discussing",
    "assessing",
    "justifying",
    "proposing",
    "arguing",
    "researching",
    "investigating",
    "planning",
    "identifying",
    "outlining",
    "deriving",
    "proving",
    "summarizing",
    "translating",
    "explaining",
    "analyzing",
    "comparing",
    "evaluating",
    "recommending",
    "condensing",
    "reviewing",
    "deploying",
    "handling",
    "managing",
)

TECHNICAL_GLOSSARY = (
    "python",
    "java",
    "javascript",
    "typescript",
    "golang",
    "rust",
    "c++",
    "c#",
    "sql",
    "nosql",
    "docker",
    "kubernetes",
    "spring boot",
    "rest api",
    "api",
    "database",
    "algorithm",
    "binary search",
    "data structure",
    "tf-idf",
    "machine learning",
    "deep learning",
    "neural",
    "regression",
    "svm",
    "vectorizer",
    "compiler",
    "function",
    "recursion",
    "concurrency",
    "multithreading",
    "distributed",
    "load balancer",
    "blockchain",
    "cryptography",
    "encryption",
    "security",
    "network",
    "protocol",
    "linux",
    "react",
    "node",
    "django",
    "flask",
    "pandas",
    "numpy",
    "tensor",
    "gradient",
    "probability",
    "statistics",
    "inference",
    "correlation",
    "causation",
    "equation",
    "integral",
    "derivative",
    "calculus",
    "algebra",
    "geometry",
    "quantum",
    "entropy",
    "simulation",
    "experiment",
    "methodology",
    "parameter",
    "variable",
    "equilibrium",
    "theory",
    "framework",
    "architecture",
    "model",
    "system",
    "evidence",
    "analysis",
    "research",
)

CODE_COMPLEXITY_INDICATORS = (
    "loop",
    "loops",
    "recursive",
    "recursion",
    "nested",
    "thread",
    "threads",
    "multithreading",
    "concurrent",
    "parallel",
    "async",
    "streaming",
    "pipeline",
    "iterative",
    "race condition",
)

CLAUSE_INDICATORS = (
    "because",
    "although",
    "whereas",
    "unless",
    "while",
    "since",
    "though",
    "which",
    "who",
    "when",
    "where",
    "that",
    "and",
    "but",
    "or",
    "so",
    "yet",
)

DEFAULT_FEATURE_WEIGHTS = MappingProxyType({
    "length": 0.55,
    "signals": 0.45,
})

DEFAULT_FEATURE_CAPS = MappingProxyType({
    "length": 20,
    "signals": 6,
})


@dataclass(frozen=True)
class ComplexityConfig:
    """Settings for the heuristic complexity estimator.

    The composite score blends two normalized signals: prompt length and the
    number of distinct matched indicators across the reasoning, operation,
    technical, code, and clause vocabularies. ``feature_weights`` sum to 1 and
    ``feature_caps`` bound each raw count before normalization, so the
    composite ``score`` always lies in [0, 1]. A score below ``low_threshold``
    maps to the first level, a score at or above ``high_threshold`` maps to
    the last level, and anything in between maps to the middle level.

    The configuration is deeply immutable: ``feature_weights`` and
    ``feature_caps`` are stored as read-only mapping proxies, so the checked
    invariants cannot be invalidated after construction.
    """

    levels: tuple[str, ...] = ("low", "medium", "high")
    low_threshold: float = 0.30
    high_threshold: float = 0.55
    feature_weights: dict[str, float] | None = None
    feature_caps: dict[str, int] | None = None
    reasoning_indicators: tuple[str, ...] = REASONING_INDICATORS
    operation_indicators: tuple[str, ...] = OPERATION_INDICATORS
    technical_glossary: tuple[str, ...] = TECHNICAL_GLOSSARY
    code_indicators: tuple[str, ...] = CODE_COMPLEXITY_INDICATORS
    clause_indicators: tuple[str, ...] = CLAUSE_INDICATORS

    def __post_init__(self) -> None:
        weights = dict(
            self.feature_weights if self.feature_weights is not None else DEFAULT_FEATURE_WEIGHTS
        )
        caps = dict(
            self.feature_caps if self.feature_caps is not None else DEFAULT_FEATURE_CAPS
        )
        if self.levels != COMPLEXITY_LEVELS:
            raise ValueError(
                f"levels must be the fixed routing-policy scale {COMPLEXITY_LEVELS!r}, "
                f"got {self.levels!r}."
            )
        if not 0 < self.low_threshold < self.high_threshold < 1:
            raise ValueError(
                "thresholds must satisfy 0 < low_threshold < high_threshold < 1, got "
                f"low={self.low_threshold} high={self.high_threshold}."
            )
        if set(weights) != set(caps):
            raise ValueError("feature_weights and feature_caps must name the same features.")
        total = sum(weights.values())
        if not 0.999 <= total <= 1.001:
            raise ValueError(f"feature_weights must sum to 1, got {total}.")
        if any(value <= 0 for value in weights.values()):
            raise ValueError("feature_weights must be positive.")
        if any(cap < 1 for cap in caps.values()):
            raise ValueError("feature_caps must be >= 1.")
        object.__setattr__(self, "feature_weights", MappingProxyType(weights))
        object.__setattr__(self, "feature_caps", MappingProxyType(caps))
