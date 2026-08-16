"""Rendering of routing and classification results for the terminal."""

import sys
from io import TextIOBase

from routellm.application.classifier_service import TrainReport
from routellm.application.execution_service import ProviderStatusRow
from routellm.domain.provider import HealthCheckResult, ProviderResponse
from routellm.domain.route_decision import RouteDecision
from routellm.evaluation.benchmark import BenchmarkResult
from routellm.evaluation.cascade_report import (
    CascadeEvaluationReport,
    CascadeTrainReport,
    format_cascade_evaluation_report,
    format_cascade_train_report,
)
from routellm.evaluation.complexity import (
    ComplexityEvaluationReport,
    format_complexity_evaluation_report,
)
from routellm.evaluation.report import EvaluationReport


def format_decision(decision: RouteDecision) -> str:
    """Render a decision as plain text without inventing any rationale."""
    confidence = "n/a" if decision.confidence is None else f"{decision.confidence:.2f}"
    lines = [
        f"{'Category':<14}: {decision.category}",
        f"{'Route':<14}: {decision.route}",
        f"{'Confidence':<14}: {confidence}",
    ]
    if decision.source:
        lines.append(f"{'Source':<14}: {decision.source}")
    if decision.complexity is not None:
        lines.append(f"{'Complexity':<14}: {decision.complexity.level}")
    if decision.signals:
        lines.append("")
        lines.append("Signals:")
        lines.extend(f"  {signal.phrase}" for signal in decision.signals)
    if decision.complexity is not None and decision.complexity.signals:
        lines.append("")
        lines.append("Complexity signals:")
        lines.extend(f"  {signal}" for signal in decision.complexity.signals)
    lines.append("")
    lines.append(f"Reason: {decision.reason}")
    return "\n".join(lines)


def render_decision(decision: RouteDecision, out: TextIOBase | None = None) -> None:
    """Print a decision to the given stream (default: stdout)."""
    print(format_decision(decision), file=out if out is not None else sys.stdout)


def format_execution_result(decision: RouteDecision, response: ProviderResponse) -> str:
    """Render a routing decision plus the outcome of executing its model.

    The status and detail lines come directly from the ``ProviderResponse`` and
    are never invented (SPEC section 38).
    """
    lines = format_decision(decision).splitlines()
    lines.append("")
    lines.append("Model output:")
    lines.append(response.text if response.status == "ok" else "(no output)")
    lines.append("")
    lines.append(f"{'Status':<14}: {response.status}")
    if response.provider:
        lines.append(f"{'Provider':<14}: {response.provider}")
        lines.append(f"{'Model':<14}: {response.model}")
    if response.requested_route != response.route:
        lines.append(f"{'Executed via':<14}: {response.route}")
    if response.latency_ms is not None:
        lines.append(f"{'Latency':<14}: {response.latency_ms:.1f} ms")
    if response.error:
        lines.append(f"{'Detail':<14}: {response.error}")
    return "\n".join(lines)


def render_execution_result(
    decision: RouteDecision, response: ProviderResponse, out: TextIOBase | None = None
) -> None:
    """Print a routing decision and its execution outcome (default: stdout)."""
    print(
        format_execution_result(decision, response),
        file=out if out is not None else sys.stdout,
    )


def format_provider_status(rows: tuple[ProviderStatusRow, ...]) -> str:
    """Render the provider status table as plain text."""
    header = ("route", "provider", "model", "available")
    entries: list[tuple[str, str, str, str]] = []
    for row in rows:
        available = (
            "n/a"
            if row.available is None
            else ("yes" if row.available else "no")
        )
        entries.append(
            (
                row.route,
                row.provider if row.provider is not None else "n/a",
                row.model if row.model is not None else "n/a",
                available,
            )
        )
    widths = [
        max(len(header[index]), max((len(entry[index]) for entry in entries), default=0))
        for index in range(len(header))
    ]
    lines = ["Provider configuration:"]
    lines.append(
        "  "
        + "  ".join(f"{value:<{width}}" for value, width in zip(header, widths))
    )
    for entry in entries:
        lines.append(
            "  " + "  ".join(f"{value:<{width}}" for value, width in zip(entry, widths))
        )
    return "\n".join(lines)


def render_provider_status(
    rows: tuple[ProviderStatusRow, ...], out: TextIOBase | None = None
) -> None:
    """Print the provider status table to the given stream (default: stdout)."""
    print(format_provider_status(rows), file=out if out is not None else sys.stdout)


def format_train_report(report: TrainReport) -> str:
    """Render a training summary as plain text."""
    metrics = report.validation_metrics
    lines = [
        "Training complete.",
        "",
        f"{'Dataset':<16}: {report.dataset_path}",
        f"{'Model saved':<16}: {report.model_path}",
        f"{'Split sizes':<16}: train={report.n_train} validation={report.n_validation} "
        f"test={report.n_test}",
        "",
        "Validation metrics (held-out validation split):",
        f"{'Accuracy':<16}: {metrics.accuracy:.3f}",
        f"{'Macro precision':<16}: {metrics.macro_precision:.3f}",
        f"{'Macro recall':<16}: {metrics.macro_recall:.3f}",
        f"{'Macro F1':<16}: {metrics.macro_f1:.3f}",
        f"{'Low-confidence rate':<16}: {metrics.low_confidence_rate:.3f}",
    ]
    return "\n".join(lines)


def format_evaluation_report(report: EvaluationReport) -> str:
    """Render an evaluation report as plain text."""
    latency = "n/a" if report.mean_latency_ms is None else f"{report.mean_latency_ms:.2f} ms"
    lines = [
        f"Evaluation on the held-out test split (n={report.n_prompts}, "
        f"low-confidence threshold {report.low_confidence_threshold:.2f})",
        "",
        f"{'Accuracy':<20}: {report.accuracy:.3f}",
        f"{'Macro precision':<20}: {report.macro_precision:.3f}",
        f"{'Macro recall':<20}: {report.macro_recall:.3f}",
        f"{'Macro F1':<20}: {report.macro_f1:.3f}",
        f"{'Low-confidence rate':<20}: {report.low_confidence_rate:.3f}",
        f"{'Mean latency':<20}: {latency}",
        "",
        "Per-class metrics:",
    ]
    lines.append(f"{'category':<12}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for metrics in report.per_class:
        lines.append(
            f"{metrics.category:<12}{metrics.precision:>10.3f}{metrics.recall:>10.3f}"
            f"{metrics.f1:>10.3f}{metrics.support:>10d}"
        )
    lines.append("")
    lines.append("Confusion matrix (rows=true, columns=predicted):")
    cell_width = max(len(category) for category in report.classes) + 1
    lines.append(
        f"{'':<{cell_width}}"
        + "".join(f"{category:>{cell_width}}" for category in report.classes)
    )
    for row_category, row in zip(report.classes, report.confusion):
        lines.append(
            f"{row_category:<{cell_width}}"
            + "".join(f"{value:>{cell_width}d}" for value in row)
        )
    return "\n".join(lines)


def render_train_report(report: TrainReport, out: TextIOBase | None = None) -> None:
    """Print a training summary to the given stream (default: stdout)."""
    print(format_train_report(report), file=out if out is not None else sys.stdout)


def render_evaluation_report(
    report: EvaluationReport, out: TextIOBase | None = None
) -> None:
    """Print an evaluation report to the given stream (default: stdout)."""
    print(format_evaluation_report(report), file=out if out is not None else sys.stdout)


def _format_rate(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate:.3f}"


def format_benchmark_report(result: BenchmarkResult) -> str:
    """Render a benchmark run as plain text for the terminal."""
    has_cost = any(row.total_cost is not None for row in result.rows)
    has_provider_latency = any(row.provider_latency_ms is not None for row in result.rows)
    lines = [
        "Benchmark complete.",
        "",
        f"{'Dataset':<18}: {result.dataset_path}",
        f"{'Report':<18}: {result.report_path}",
        "",
        "Candidate metrics (validation split):",
    ]
    header = (
        f"{'candidate':<26}{'acc':>8}{'f1':>8}{'low-conf':>9}"
        f"{'latency_ms':>12}{'size_bytes':>12}"
    )
    if has_cost:
        header += f"{'cost':>10}"
    if has_provider_latency:
        header += f"{'prov_lat':>10}"
    lines.append(header)
    for row in result.rows:
        metrics = row.validation_metrics
        line = (
            f"{row.name:<26}{metrics.accuracy:>8.3f}{metrics.macro_f1:>8.3f}"
            f"{_format_rate(metrics.low_confidence_rate):>9}"
            f"{row.mean_latency_ms:>12.3f}{row.size_bytes:>12d}"
        )
        if has_cost:
            cost_str = f"{row.total_cost:.6f}" if row.total_cost is not None else "n/a"
            line += f"{cost_str:>10}"
        if has_provider_latency:
            if row.provider_latency_ms is not None:
                pl_str = f"{row.provider_latency_ms:.1f}"
            else:
                pl_str = "n/a"
            line += f"{pl_str:>10}"
        lines.append(line)
    selected = next(row for row in result.rows if row.name == result.selected_name)
    test_metrics = selected.test_metrics
    lines.extend(
        [
            "",
            f"Selected candidate: {result.selected_name} "
            f"(validation macro F1 {selected.validation_metrics.macro_f1:.3f})",
            "",
            "Selected candidate on the held-out test split:",
            f"  Accuracy: {test_metrics.accuracy:.3f}",
            f"  Macro precision: {test_metrics.macro_precision:.3f}",
            f"  Macro recall: {test_metrics.macro_recall:.3f}",
            f"  Macro F1: {test_metrics.macro_f1:.3f}",
        ]
    )
    if result.cost_summary is not None:
        cs = result.cost_summary
        lines.extend(
            [
                "",
                "Cost summary:",
                f"  Total: ${cs.total:.6f}",
                f"  Mean per prompt: ${cs.mean_per_prompt:.6f}",
            ]
        )
        if cs.by_route:
            lines.append("  By route:")
            for route, cost in cs.by_route.items():
                lines.append(f"    {route}: ${cost:.6f}")
    if result.latency_summary is not None:
        ls = result.latency_summary
        lines.extend(
            [
                "",
                "Latency summary:",
                f"  Mean: {ls.mean_ms:.1f} ms",
                f"  P50: {ls.p50_ms:.1f} ms",
                f"  P95: {ls.p95_ms:.1f} ms",
            ]
        )
        if ls.by_route:
            lines.append("  By route:")
            for route, lat in ls.by_route.items():
                lines.append(f"    {route}: {lat:.1f} ms")
    return "\n".join(lines)


def render_benchmark_report(result: BenchmarkResult, out: TextIOBase | None = None) -> None:
    """Print a benchmark run to the given stream (default: stdout)."""
    print(format_benchmark_report(result), file=out if out is not None else sys.stdout)


def render_cascade_train_report(
    report: CascadeTrainReport, out: TextIOBase | None = None
) -> None:
    """Print a cascade training summary to the given stream (default: stdout)."""
    print(format_cascade_train_report(report), file=out if out is not None else sys.stdout)


def render_cascade_evaluation_report(
    report: CascadeEvaluationReport, out: TextIOBase | None = None
) -> None:
    """Print a cascade evaluation report to the given stream (default: stdout)."""
    print(
        format_cascade_evaluation_report(report),
        file=out if out is not None else sys.stdout,
    )


def render_complexity_evaluation(
    report: ComplexityEvaluationReport, out: TextIOBase | None = None
) -> None:
    """Print a complexity evaluation to the given stream (default: stdout)."""
    print(
        format_complexity_evaluation_report(report),
        file=out if out is not None else sys.stdout,
    )


def format_health_check(results: tuple[HealthCheckResult, ...]) -> str:
    """Render health check results as plain text."""
    if not results:
        return "No health check results."
    header = ("route", "provider", "model", "available", "latency_ms", "error")
    entries: list[tuple[str, str, str, str, str, str]] = []
    for r in results:
        available = "n/a" if r.available is None else ("yes" if r.available else "no")
        latency = f"{r.response_time_ms:.1f}" if r.response_time_ms is not None else "n/a"
        error = r.error if r.error else ""
        entries.append((
            r.route,
            r.provider if r.provider is not None else "n/a",
            r.model if r.model is not None else "n/a",
            available,
            latency,
            error,
        ))
    widths = [
        max(len(header[i]), max((len(e[i]) for e in entries), default=0))
        for i in range(len(header))
    ]
    lines = [f"Health check ({results[0].checked_at}):"]
    lines.append(
        "  "
        + "  ".join(f"{v:<{w}}" for v, w in zip(header, widths))
    )
    for entry in entries:
        lines.append(
            "  " + "  ".join(f"{v:<{w}}" for v, w in zip(entry, widths))
        )
    return "\n".join(lines)


def render_health_check(
    results: tuple[HealthCheckResult, ...], out: TextIOBase | None = None
) -> None:
    """Print health check results to the given stream (default: stdout)."""
    print(format_health_check(results), file=out if out is not None else sys.stdout)
