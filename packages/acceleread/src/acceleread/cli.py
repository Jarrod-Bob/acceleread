# SPDX-License-Identifier: Apache-2.0
"""Command-line entry point (docs/spec/v0.md §9). The tracer ships `run`; more commands follow."""

import argparse
import re
import sys
from datetime import timedelta
from pathlib import Path

from acceleread import __version__
from acceleread.classifier import Classifier
from acceleread.doctor import report, run_checks
from acceleread.jev import JevClassifier
from acceleread.jobcli import JOB_COMMANDS, add_job_commands, dispatch
from acceleread.languages import LanguagePackError, add_language, workspace_tessdata
from acceleread.workspace import (
    JobRunningError,
    NetworkFilesystemError,
    Workspace,
    resolve_workspace_path,
)


def make_classifier(model: str) -> Classifier:
    return JevClassifier(model=model)


_UNITS = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days", "w": "weeks"}


def parse_duration(text: str) -> timedelta:
    """Parse `90s`, `15m`, `12h`, `30d` or `2w`."""
    match = re.fullmatch(r"(\d+)([smhdw])", text.strip())
    if match is None:
        raise ValueError(f"invalid duration {text!r}: use a number and s, m, h, d or w")
    return timedelta(**{_UNITS[match[2]]: int(match[1])})


def _duration_arg(text: str) -> timedelta:
    try:
        return parse_duration(text)
    except ValueError as err:
        raise argparse.ArgumentTypeError(str(err)) from err


def _housekeeping(args: argparse.Namespace) -> int:
    """`jobs delete`, `jobs prune`, `cache prune` and `cache clear`."""
    try:
        workspace = Workspace.open(args.workspace, allow_network_fs=args.allow_network_fs)
    except NetworkFilesystemError as err:
        print(f"acceleread: {err}", file=sys.stderr)
        return 1
    with workspace:
        if args.command == "cache":
            if args.cache_command == "clear":
                print(f"cleared {workspace.cache.clear()} cached Judgments")
            else:
                print(
                    f"pruned {workspace.cache.prune(older_than=args.older_than)} cached Judgments"
                )
            return 0
        try:
            if args.jobs_command == "delete":
                workspace.delete_job(args.job_id)
                print(f"deleted {args.job_id}")
            else:
                for job_id in workspace.prune_jobs(
                    older_than=args.older_than, keep_records=args.keep_records, kind=args.kind
                ):
                    print(f"pruned {job_id}")
        except JobRunningError as err:
            print(f"acceleread: {err}", file=sys.stderr)
            return 1
        except KeyError as err:
            print(f"acceleread: no such Job: {err.args[0]}", file=sys.stderr)
            return 1
    return 0


def _doctor(args: argparse.Namespace) -> int:
    checks = run_checks(resolve_workspace_path(args.workspace))
    print(report(checks))
    return 1 if any(c.status == "error" for c in checks) else 0


def _add_language(args: argparse.Namespace) -> int:
    tessdata = workspace_tessdata(resolve_workspace_path(args.workspace))
    try:
        path = add_language(args.language, tessdata, from_file=args.from_file)
    except LanguagePackError as err:
        print(f"acceleread: {err}", file=sys.stderr)
        return 1
    print(f"installed {path.stem} into {path.parent}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="acceleread")
    parser.add_argument("--version", action="version", version=f"acceleread {__version__}")
    parser.add_argument(
        "--workspace",
        type=Path,
        help="Workspace directory (default: $ACCELEREAD_HOME or ~/.acceleread)",
    )
    parser.add_argument(
        "--allow-network-fs",
        action="store_true",
        help="allow a Workspace on NFS or SMB, where SQLite WAL is unsafe",
    )
    parser.add_argument(
        "--server",
        metavar="URL",
        help="send Job commands to this acceleread server instead of running in-process",
    )
    commands = parser.add_subparsers(dest="command")

    jobs_cmd = commands.add_parser("jobs", help="manage Jobs in the Workspace")
    jobs_cmd.add_argument("--all", action="store_true", help="include single-Document ingests")
    jobs = jobs_cmd.add_subparsers(dest="jobs_command")  # no sub-command lists the Jobs
    delete_cmd = jobs.add_parser("delete", help="delete a Job's directory (refused while running)")
    delete_cmd.add_argument("job_id")
    prune_cmd = jobs.add_parser("prune", help="free disk from finished Jobs")
    prune_cmd.add_argument("--older-than", type=_duration_arg, required=True, metavar="DURATION")
    prune_cmd.add_argument("--keep-records", action="store_true", help="remove only inputs")
    prune_cmd.add_argument("--kind", choices=["job", "ingest"], help="only Jobs of this kind")

    cache_cmd = commands.add_parser("cache", help="manage the Judgment cache")
    cache = cache_cmd.add_subparsers(dest="cache_command", required=True)
    cache_prune = cache.add_parser("prune", help="drop Judgments not used recently")
    cache_prune.add_argument("--older-than", type=_duration_arg, required=True, metavar="DURATION")
    cache.add_parser("clear", help="drop every cached Judgment")

    ocr_cmd = commands.add_parser("ocr", help="manage OCR language packs")
    ocr = ocr_cmd.add_subparsers(dest="ocr_command", required=True)
    add_lang = ocr.add_parser(
        "add-language", help="install a tessdata_fast pack into the Workspace"
    )
    add_lang.add_argument("language", metavar="xx", help="ISO 639-1 code, such as de")
    add_lang.add_argument("--from-file", type=Path, help="install this .traineddata, no download")
    commands.add_parser("doctor", help="report Tesseract, language packs, models and extras")

    add_job_commands(commands)

    args = parser.parse_args(argv)
    if args.command == "jobs" and args.jobs_command is None:
        return dispatch(args, make_classifier)
    if args.command in ("jobs", "cache"):
        return _housekeeping(args)
    if args.command == "ocr":
        return _add_language(args)
    if args.command == "doctor":
        return _doctor(args)
    if args.command in JOB_COMMANDS:
        return dispatch(args, make_classifier)
    parser.print_help()
    return 0
