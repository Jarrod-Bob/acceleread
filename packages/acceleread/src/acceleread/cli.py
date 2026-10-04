# SPDX-License-Identifier: Apache-2.0
"""Command-line entry point (docs/spec/v0.md §9). The tracer ships `run`; more commands follow."""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import TextIO

from acceleread import __version__
from acceleread.classifier import Classifier
from acceleread.jev import JevClassifier
from acceleread.models import DEFAULT_JEV_MODEL, JobSpec, Taxonomy
from acceleread.pipeline import run


def make_classifier(model: str) -> Classifier:
    return JevClassifier(model=model)


async def _run(spec: JobSpec, out: TextIO) -> int:
    failed = 0
    async for record in run(spec, make_classifier(spec.model)):
        out.write(record.model_dump_json(exclude_none=True) + "\n")
        out.flush()
        failed += record.status != "ok"
        print(f"{record.source.filename}: {record.status}", file=sys.stderr)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="acceleread")
    parser.add_argument("--version", action="version", version=f"acceleread {__version__}")
    commands = parser.add_subparsers(dest="command")

    run_cmd = commands.add_parser("run", help="ingest Documents and print Records as JSONL")
    run_cmd.add_argument("inputs", nargs="+", type=Path, help="PDF files")
    run_cmd.add_argument("--taxonomy", type=Path, required=True, help="Taxonomy YAML or JSON")
    run_cmd.add_argument("--model", default=DEFAULT_JEV_MODEL, help="Jev model")
    run_cmd.add_argument("-o", "--output", type=Path, help="write JSONL here instead of stdout")

    args = parser.parse_args(argv)
    if args.command != "run":
        parser.print_help()
        return 0
    spec = JobSpec(inputs=args.inputs, taxonomy=Taxonomy.from_file(args.taxonomy), model=args.model)
    if args.output:
        with args.output.open("w", encoding="utf-8") as out:
            return asyncio.run(_run(spec, out))
    return asyncio.run(_run(spec, sys.stdout))
