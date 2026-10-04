# SPDX-License-Identifier: Apache-2.0
"""`acceleread-eval` entry point. Suites and variants arrive with the evaluation-harness issue."""

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="acceleread-eval")
    parser.parse_args(argv)
    parser.print_help()
    return 0
