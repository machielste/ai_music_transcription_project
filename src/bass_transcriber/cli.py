"""Command-line interface for bass-transcriber."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from bass_transcriber import __version__


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level command-line parser."""
    parser = argparse.ArgumentParser(
        prog="bass-transcriber",
        description="Transcribe songs into synchronized bass tablature for ToneLib Jam.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0

