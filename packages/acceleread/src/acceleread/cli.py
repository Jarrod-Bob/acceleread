# SPDX-License-Identifier: Apache-2.0
"""Command-line entry point. Commands arrive with later build issues (docs/spec/v0.md §9)."""

import argparse

from acceleread import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="acceleread")
    parser.add_argument("--version", action="version", version=f"acceleread {__version__}")
    parser.parse_args(argv)
    parser.print_help()
    return 0
