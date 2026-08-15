"""Entry point for the ``routellm`` command."""

import argparse

from routellm import __version__


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
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface and return the process exit code."""
    parser = build_parser()
    parser.parse_args(argv)
    if not argv:
        parser.print_help()
    return 0
