"""Entry point for the ``routellm`` command."""

import argparse
import sys

from routellm import __version__
from routellm.application.classifier_service import ClassifierService, load_model
from routellm.application.execution_service import ExecutionService
from routellm.application.route_service import RouteService
from routellm.classification.cascade_model import CascadeModel
from routellm.cli.render import (
    render_benchmark_report,
    render_cascade_evaluation_report,
    render_cascade_train_report,
    render_complexity_evaluation,
    render_decision,
    render_evaluation_report,
    render_execution_result,
    render_provider_status,
    render_train_report,
)
from routellm.configuration.providers import load_provider_config
from routellm.evaluation.cascade_report import CascadeEvaluationReport
from routellm.evaluation.complexity import evaluate_complexity


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the ``routellm`` command."""
    parser = argparse.ArgumentParser(
        prog="routellm",
        description="Explainable local routing layer for natural-language requests.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"routellm {__version__}",
        help="show the program version and exit",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    route_parser = subparsers.add_parser(
        "route",
        help="classify a prompt and select a logical route",
    )
    route_parser.add_argument(
        "prompt",
        nargs="+",
        help="natural-language prompt; words are joined with spaces",
    )
    route_parser.add_argument(
        "--model",
        default=None,
        help="path to a cascade model for confidence-based routing "
        "(default: rule-based routing only)",
    )
    run_parser = subparsers.add_parser(
        "run",
        help="route a prompt and execute the selected model through its provider",
    )
    run_parser.add_argument(
        "prompt",
        nargs="+",
        help="natural-language prompt; words are joined with spaces",
    )
    run_parser.add_argument(
        "--model",
        default=None,
        help="path to a cascade model for confidence-based routing "
        "(default: rule-based routing only)",
    )
    run_parser.add_argument(
        "--config",
        default=None,
        help="path to a provider configuration TOML file (default: built-in defaults)",
    )
    providers_parser = subparsers.add_parser(
        "providers",
        help="show configured routes, their providers/models, and availability",
    )
    providers_parser.add_argument(
        "--config",
        default=None,
        help="path to a provider configuration TOML file (default: built-in defaults)",
    )
    train_parser = subparsers.add_parser(
        "train",
        help="train the statistical classifier on the labeled dataset",
    )
    train_parser.add_argument(
        "--dataset",
        default="data/datasets/prompts.csv",
        help="labeled CSV dataset (default: data/datasets/prompts.csv)",
    )
    train_parser.add_argument(
        "--out",
        default="models/classifier.joblib",
        help="path to write the trained model (default: models/classifier.joblib)",
    )
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="evaluate a trained classifier on the held-out test split",
    )
    evaluate_parser.add_argument(
        "--model",
        default="models/classifier.joblib",
        help="path to the trained model (default: models/classifier.joblib)",
    )
    evaluate_parser.add_argument(
        "--dataset",
        default=None,
        help="dataset path; defaults to the one recorded at training time",
    )
    evaluate_parser.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="low-confidence threshold for reporting (default: 0.80)",
    )
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="compare candidate classifiers on the labeled dataset",
    )
    benchmark_parser.add_argument(
        "--dataset",
        default="data/datasets/prompts.csv",
        help="labeled CSV dataset (default: data/datasets/prompts.csv)",
    )
    benchmark_parser.add_argument(
        "--out",
        default="models/benchmarks",
        help="directory for benchmark artifacts and report (default: models/benchmarks)",
    )
    cascade_parser = subparsers.add_parser(
        "cascade",
        help="train the calibrated cascade routing model",
    )
    cascade_parser.add_argument(
        "--dataset",
        default="data/datasets/prompts.csv",
        help="labeled CSV dataset (default: data/datasets/prompts.csv)",
    )
    cascade_parser.add_argument(
        "--out",
        default="models/cascade.joblib",
        help="path to write the cascade model (default: models/cascade.joblib)",
    )
    complexity_parser = subparsers.add_parser(
        "complexity",
        help="evaluate the heuristic complexity estimator on a labeled set",
    )
    complexity_parser.add_argument(
        "--dataset",
        default="data/datasets/complexity.csv",
        help="labeled CSV dataset with text,complexity "
        "(default: data/datasets/complexity.csv)",
    )
    complexity_parser.add_argument(
        "--model",
        default=None,
        help="path to a cascade model for route-policy impact under cascade "
        "routing (default: rule-only routing)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface and return the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "route":
        prompt = " ".join(args.prompt)
        model = None
        if args.model:
            try:
                model = load_model(args.model)
            except (ValueError, OSError) as error:
                print(f"route failed: {error}", file=sys.stderr)
                return 1
            if not isinstance(model, CascadeModel):
                print(
                    f"route failed: {args.model} is not a cascade model; "
                    "train one with 'routellm cascade'.",
                    file=sys.stderr,
                )
                return 1
        decision = RouteService(model=model).route(prompt)
        render_decision(decision)
        return 0
    if args.command == "run":
        prompt = " ".join(args.prompt)
        try:
            config = load_provider_config(args.config)
        except (ValueError, OSError) as error:
            print(f"run failed: {error}", file=sys.stderr)
            return 1
        model = None
        if args.model:
            try:
                model = load_model(args.model)
            except (ValueError, OSError) as error:
                print(f"run failed: {error}", file=sys.stderr)
                return 1
            if not isinstance(model, CascadeModel):
                print(
                    f"run failed: {args.model} is not a cascade model; "
                    "train one with 'routellm cascade'.",
                    file=sys.stderr,
                )
                return 1
        decision = RouteService(model=model).route(prompt)
        response = ExecutionService(config=config).execute(decision)
        render_execution_result(decision, response)
        return 0
    if args.command == "providers":
        try:
            config = load_provider_config(args.config)
            rows = ExecutionService(config=config).status_table()
        except (ValueError, OSError) as error:
            print(f"providers failed: {error}", file=sys.stderr)
            return 1
        render_provider_status(rows)
        return 0
    if args.command == "train":
        try:
            report = ClassifierService().train(args.dataset, args.out)
        except (ValueError, OSError) as error:
            print(f"train failed: {error}", file=sys.stderr)
            return 1
        render_train_report(report)
        return 0
    if args.command == "evaluate":
        try:
            report = ClassifierService().evaluate(
                args.model,
                dataset_path=args.dataset,
                low_confidence_threshold=args.threshold,
            )
        except (ValueError, OSError) as error:
            print(f"evaluate failed: {error}", file=sys.stderr)
            return 1
        if isinstance(report, CascadeEvaluationReport):
            render_cascade_evaluation_report(report)
        else:
            render_evaluation_report(report)
        return 0
    if args.command == "benchmark":
        try:
            result = ClassifierService().benchmark(args.dataset, out_dir=args.out)
        except (ValueError, OSError) as error:
            print(f"benchmark failed: {error}", file=sys.stderr)
            return 1
        render_benchmark_report(result)
        return 0
    if args.command == "cascade":
        try:
            report = ClassifierService().train_cascade(args.dataset, args.out)
        except (ValueError, OSError) as error:
            print(f"cascade failed: {error}", file=sys.stderr)
            return 1
        render_cascade_train_report(report)
        return 0
    if args.command == "complexity":
        model = None
        if args.model:
            try:
                model = load_model(args.model)
            except (ValueError, OSError) as error:
                print(f"complexity failed: {error}", file=sys.stderr)
                return 1
            if not isinstance(model, CascadeModel):
                print(
                    f"complexity failed: {args.model} is not a cascade model; "
                    "train one with 'routellm cascade'.",
                    file=sys.stderr,
                )
                return 1
        try:
            report = evaluate_complexity(args.dataset, model=model)
        except (ValueError, OSError) as error:
            print(f"complexity failed: {error}", file=sys.stderr)
            return 1
        render_complexity_evaluation(report)
        return 0
    return 0
