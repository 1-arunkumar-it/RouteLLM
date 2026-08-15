"""Report types and formatting for the cascaded router (Milestone 4)."""

from dataclasses import dataclass

from routellm.evaluation.report import ClassMetrics


@dataclass(frozen=True)
class CascadeTrainReport:
    """Summary of a completed cascade training run."""

    dataset_path: str
    model_path: str
    report_path: str
    fingerprint: str
    n_train: int
    n_validation: int
    n_test: int
    threshold: float
    validation_macro_f1: float
    override_categories: frozenset[str]
    rule_precision: dict[str, float]
    calibration_method: str
    calibration_cv: int


@dataclass(frozen=True)
class CascadeEvaluationReport:
    """Measured behavior of the cascade: quality, overrides, and fallbacks."""

    n_prompts: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class: tuple[ClassMetrics, ...]
    classes: tuple[str, ...]
    confusion: tuple[tuple[int, ...], ...]
    mean_latency_ms: float | None
    threshold: float
    n_rules: int
    n_classifier: int
    n_fallback: int
    override_rate: float
    fallback_rate: float


def format_cascade_train_report(report: CascadeTrainReport) -> str:
    """Render a cascade training summary as plain text."""
    override = (
        ", ".join(sorted(report.override_categories)) if report.override_categories else "(none)"
    )
    lines = [
        "Cascade model trained.",
        "",
        f"{'Dataset':<18}: {report.dataset_path}",
        f"{'Model saved':<18}: {report.model_path}",
        f"{'Report':<18}: {report.report_path}",
        f"{'Split sizes':<18}: train={report.n_train} validation={report.n_validation} "
        f"test={report.n_test}",
        f"{'Calibration':<18}: {report.calibration_method}, cv={report.calibration_cv}",
        "",
        f"Selected threshold: {report.threshold:.2f} "
        f"(validation macro F1 {report.validation_macro_f1:.3f})",
        f"Rule override categories: {override}",
        "",
        "Rule precision on the validation split (per category):",
        f"{'category':<16}{'precision':>10}",
    ]
    for category, precision in report.rule_precision.items():
        lines.append(f"{category:<16}{precision:>10.3f}")
    return "\n".join(lines)


def format_cascade_evaluation_report(report: CascadeEvaluationReport) -> str:
    """Render a cascade evaluation report as plain text."""
    latency = "n/a" if report.mean_latency_ms is None else f"{report.mean_latency_ms:.2f} ms"
    lines = [
        f"Evaluation of the cascade on the held-out test split "
        f"(n={report.n_prompts}, threshold {report.threshold:.2f})",
        "",
        f"{'Accuracy':<20}: {report.accuracy:.3f}",
        f"{'Macro precision':<20}: {report.macro_precision:.3f}",
        f"{'Macro recall':<20}: {report.macro_recall:.3f}",
        f"{'Macro F1':<20}: {report.macro_f1:.3f}",
        f"{'Rule decisions':<20}: {report.n_rules} ({report.override_rate:.1%})",
        f"{'Classifier decisions':<20}: {report.n_classifier}",
        f"{'Fallback decisions':<20}: {report.n_fallback} ({report.fallback_rate:.1%})",
        f"{'Mean latency':<20}: {latency}",
        "",
        "Per-class metrics:",
    ]
    lines.append(
        f"{'category':<12}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}"
    )
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
