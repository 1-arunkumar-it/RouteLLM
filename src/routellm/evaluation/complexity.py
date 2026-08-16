"""Evaluation of the heuristic complexity estimator (Milestone 5).

The estimator's level predictions are measured against a hand-labeled
evaluation set with the same metrics machinery as category classification.
A route-policy summary then shows how often the estimate actually changed a
route under the evaluated routing configuration (SPEC section 41 and ROADMAP
Milestone 5): a route changes only when the (category, level) pair is in
``COMPLEXITY_REROUTES``. Categories and routes come from the same
``RouteService`` path used by ``routellm route``, so the impact count is
honest about the configuration that produced it (rule-only or cascade).
"""

import time
from dataclasses import dataclass

from routellm.application.route_service import RouteService
from routellm.classification.cascade_model import CascadeModel
from routellm.complexity.config import ComplexityConfig
from routellm.complexity.dataset import load_complexity_dataset
from routellm.evaluation.report import ClassMetrics, compute_metrics
from routellm.routing import policy


@dataclass(frozen=True)
class ComplexityEvaluationReport:
    """Measured quality of the estimator and its effect on routing.

    ``routing_source`` records which routing configuration produced the
    route-policy numbers: ``"rules"`` (no model) or ``"cascade"``.
    """

    dataset_path: str
    n_prompts: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class: tuple[ClassMetrics, ...]
    classes: tuple[str, ...]
    confusion: tuple[tuple[int, ...], ...]
    mean_latency_ms: float | None
    routing_source: str
    n_rerouted: int
    rerouted_by_level: dict[str, int]


def evaluate_complexity(
    dataset_path: str,
    config: ComplexityConfig | None = None,
    model: CascadeModel | None = None,
    signal_rules: dict[str, tuple[tuple[str, ...], ...]] | None = None,
) -> ComplexityEvaluationReport:
    """Measure estimator levels and route changes on a labeled evaluation set.

    Categories and routes are produced by a ``RouteService`` built from the
    same ``model`` and ``signal_rules`` the caller would use for routing, so
    the route-policy impact is scoped to that configuration.
    """
    config = config or ComplexityConfig()
    dataset = load_complexity_dataset(dataset_path, config.levels)
    service = RouteService(model=model, signal_rules=signal_rules, complexity_config=config)
    start = time.perf_counter()
    predicted: list[str] = []
    n_rerouted = 0
    rerouted_by_level: dict[str, int] = {}
    for text in dataset.texts:
        decision = service.route(text)
        predicted.append(decision.complexity.level)
        baseline = policy.route_for(decision.category, config.levels[0])
        if decision.route != baseline:
            n_rerouted += 1
            rerouted_by_level[decision.complexity.level] = (
                rerouted_by_level.get(decision.complexity.level, 0) + 1
            )
    elapsed = time.perf_counter() - start
    metrics = compute_metrics(
        y_true=dataset.levels,
        y_pred=tuple(predicted),
        confidences=(None,) * len(dataset),
        classes=config.levels,
        low_confidence_threshold=1.0,
        elapsed_seconds=elapsed,
    )
    return ComplexityEvaluationReport(
        dataset_path=str(dataset_path),
        n_prompts=len(dataset),
        accuracy=metrics.accuracy,
        macro_precision=metrics.macro_precision,
        macro_recall=metrics.macro_recall,
        macro_f1=metrics.macro_f1,
        per_class=metrics.per_class,
        classes=metrics.classes,
        confusion=metrics.confusion,
        mean_latency_ms=metrics.mean_latency_ms,
        routing_source="cascade" if model is not None else "rules",
        n_rerouted=n_rerouted,
        rerouted_by_level=rerouted_by_level,
    )


def format_complexity_evaluation_report(report: ComplexityEvaluationReport) -> str:
    """Render a complexity evaluation report as plain text."""
    latency = "n/a" if report.mean_latency_ms is None else f"{report.mean_latency_ms:.2f} ms"
    routing = "rule-only routing" if report.routing_source == "rules" else "cascade routing"
    lines = [
        "Complexity estimator evaluation on the labeled set "
        f"(n={report.n_prompts})",
        "",
        f"{'Dataset':<20}: {report.dataset_path}",
        f"{'Accuracy':<20}: {report.accuracy:.3f}",
        f"{'Macro precision':<20}: {report.macro_precision:.3f}",
        f"{'Macro recall':<20}: {report.macro_recall:.3f}",
        f"{'Macro F1':<20}: {report.macro_f1:.3f}",
        f"{'Mean latency':<20}: {latency}",
        "",
        "Per-level metrics:",
    ]
    lines.append(f"{'level':<12}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for metrics in report.per_class:
        lines.append(
            f"{metrics.category:<12}{metrics.precision:>10.3f}{metrics.recall:>10.3f}"
            f"{metrics.f1:>10.3f}{metrics.support:>10d}"
        )
    lines.append("")
    lines.append("Confusion matrix (rows=true, columns=predicted):")
    cell_width = max(len(level) for level in report.classes) + 1
    lines.append(
        f"{'':<{cell_width}}"
        + "".join(f"{level:>{cell_width}}" for level in report.classes)
    )
    for row_level, row in zip(report.classes, report.confusion):
        lines.append(
            f"{row_level:<{cell_width}}"
            + "".join(f"{value:>{cell_width}d}" for value in row)
        )
    lines.extend(
        [
            "",
            "Route-policy behavior under the evaluated routing configuration "
            f"({routing}); complexity may re-route general_qa and summarization "
            "to 'reasoning' at high complexity:",
            f"  Prompts whose route changed: {report.n_rerouted}",
        ]
    )
    for level in report.classes:
        count = report.rerouted_by_level.get(level, 0)
        lines.append(f"  Re-routed at '{level}' complexity: {count}")
    return "\n".join(lines)
