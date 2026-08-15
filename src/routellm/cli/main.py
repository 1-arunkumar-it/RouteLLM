"""Entry point for the ``routellm`` command."""

import argparse
import sys

from routellm import __version__
from routellm.application.classifier_service import ClassifierService
from routellm.application.route_service import RouteService
from routellm.cli.render import (
    render_benchmark_report,
    render_decision,
    render_evaluation_report,
    render_train_report,
)


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
        decision = RouteService().route(prompt)
        render_decision(decision)
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
    return 0
