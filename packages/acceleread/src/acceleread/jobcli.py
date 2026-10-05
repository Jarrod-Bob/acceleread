# SPDX-License-Identifier: Apache-2.0
"""The CLI's Job commands (docs/spec/v0.md §9): `run`, `validate`, `status`, `jobs`, `cancel`,
`resume`, `retry --failed`, `records --flagged` and `export`, in-process by default or through
the HTTP API with `--server URL`.

`run` writes JSONL to stdout (or `-o`) and progress to stderr. Exit codes: 0 success, 1 a Job,
Document or command failed, 2 a spec that does not validate.
"""

import argparse
import asyncio
import json
import sys
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, TextIO

import yaml
from pydantic import ValidationError

from acceleread.classifier import Classifier
from acceleread.config import load_config
from acceleread.extract import VENDORED_TESSDATA
from acceleread.jobcontrol import (
    JobFinishedError,
    Requeue,
    cancel_job,
    job_summary,
    load_manifest,
    manifest_path,
    resume_job,
    retry_failed,
)
from acceleread.languages import installed_languages, workspace_tessdata
from acceleread.models import DEFAULT_JEV_MODEL, JobSpec, Taxonomy
from acceleread.runner import Runner, run, stream_job
from acceleread.serverclient import ServerClient, ServerError
from acceleread.validate import Finding, SpecError, validate
from acceleread.workers import WorkerSettings
from acceleread.workspace import (
    JobPrunedError,
    JobRunningError,
    NetworkFilesystemError,
    StorageVersionError,
    Workspace,
    resolve_workspace_path,
)

ClassifierFactory = Callable[[str], Classifier]
JOB_COMMANDS = frozenset(
    {"run", "validate", "status", "cancel", "resume", "retry", "records", "export", "jobs"}
)
# What `--server` can serve. `run`, `validate` and `records` need request shapes the API issue
# (#41) has yet to fix, so they stay in-process.
SERVER_COMMANDS = frozenset({"status", "cancel", "resume", "retry", "export", "jobs"})


class CliError(Exception):
    def __init__(self, message: str, code: int = 1) -> None:
        super().__init__(message)
        self.code = code


def _bool(text: str) -> bool:
    if text.lower() in ("true", "yes", "1"):
        return True
    if text.lower() in ("false", "no", "0"):
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {text!r}")


# Parsers


def add_run_arguments(run_cmd: argparse.ArgumentParser) -> None:
    """The Job spec flags, shared by `run` and `validate`."""
    run_cmd.add_argument("inputs", nargs="*", type=Path, help="PDF or HTML files, or globs")
    run_cmd.add_argument("--job", type=Path, help="a job.yaml (the Job spec); flags override it")
    run_cmd.add_argument("--taxonomy", type=Path, help="Taxonomy YAML or JSON")
    run_cmd.add_argument(
        "--questions", type=Path, action="append", help="Question Set (repeatable)"
    )
    run_cmd.add_argument("--model", help=f"Jev model (default {DEFAULT_JEV_MODEL})")
    run_cmd.add_argument("--profile", choices=["fast", "quality"], help="Extraction Profile")
    run_cmd.add_argument(
        "--ocr-language", action="append", metavar="xx", help="ISO 639-1 code (repeatable)"
    )
    run_cmd.add_argument("--max-cost-usd", type=float, help="auto-cancel at this estimated spend")
    run_cmd.add_argument("--no-cache", action="store_true", help="bypass the Judgment cache")


def add_job_commands(commands: Any) -> None:
    """Register the Job commands on the main parser's sub-commands."""
    run_cmd = commands.add_parser("run", help="ingest Documents and print Records as JSONL")
    add_run_arguments(run_cmd)
    run_cmd.add_argument("-o", "--output", type=Path, help="write JSONL here instead of stdout")
    run_cmd.add_argument("--ocr-workers", type=int, help="extraction workers (default: by Profile)")
    run_cmd.add_argument("--threads-per-worker", type=int, help="threads per extraction worker")

    check_cmd = commands.add_parser("validate", help="dry-run every submit check")
    add_run_arguments(check_cmd)

    status_cmd = commands.add_parser("status", help="a Job's summary (default: the latest Job)")
    status_cmd.add_argument("job_id", nargs="?")
    status_cmd.add_argument("--json", action="store_true", help="print the summary as JSON")

    cancel_cmd = commands.add_parser("cancel", help="cancel a Job")
    cancel_cmd.add_argument("job_id")

    resume_cmd = commands.add_parser(
        "resume", help="re-queue a Job's cancelled and unstarted Documents"
    )
    resume_cmd.add_argument("job_id")
    retry_cmd = commands.add_parser("retry", help="re-queue a Job's failed Documents")
    retry_cmd.add_argument("job_id")
    retry_cmd.add_argument("--failed", action="store_true", required=True)
    for cmd in (resume_cmd, retry_cmd):
        cmd.add_argument("--ocr-workers", type=int, help="extraction workers")
        cmd.add_argument("--threads-per-worker", type=int, help="threads per extraction worker")

    records_cmd = commands.add_parser("records", help="print a Job's Records as JSONL")
    records_cmd.add_argument("job_id")
    records_cmd.add_argument(
        "--flagged", action="store_true", help="only Judgments flagged for Review"
    )
    records_cmd.add_argument("--status", choices=["ok", "partial", "failed", "cancelled"])
    records_cmd.add_argument("--include-text", type=_bool, default=True, metavar="true|false")

    export_cmd = commands.add_parser("export", help="a Job's JSONL, one line per Document")
    export_cmd.add_argument("job_id")
    export_cmd.add_argument("--include-text", type=_bool, default=True, metavar="true|false")
    export_cmd.add_argument("-o", "--output", type=Path, help="write here, manifest beside it")


# Building the spec


def build_spec(args: argparse.Namespace) -> JobSpec:
    """The Job spec from `--job job.yaml` and flags. A flag overrides the file."""
    data: dict[str, Any] = {}
    if args.job is not None:
        try:
            data = yaml.safe_load(args.job.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as err:
            raise CliError(f"cannot read {args.job}: {err}", 2) from err
    if args.inputs:
        data["inputs"] = [str(p) for p in args.inputs]
    if args.model:
        data["model"] = args.model
    if args.profile:
        data["extraction_profile"] = args.profile
    if args.ocr_language:
        data["ocr_languages"] = args.ocr_language
    if args.max_cost_usd is not None:
        data["max_cost_usd"] = args.max_cost_usd
    if args.no_cache:
        data["cache"] = False
    if args.questions:
        data["question_sets"] = [*data.get("question_sets", []), *map(str, args.questions)]
    if args.taxonomy:
        data["taxonomy"] = Taxonomy.from_file(args.taxonomy).model_dump(exclude_none=True)
    if not data.get("inputs"):
        raise CliError("no inputs: pass files or globs, or an `inputs` list in --job", 2)
    try:
        return JobSpec.model_validate(data)
    except ValidationError as err:
        raise CliError(f"invalid Job spec:\n{err}", 2) from err


def _findings(findings: list[Finding], label: str, out: TextIO) -> None:
    for finding in findings:
        print(f"{label}: {finding.code}: {finding.message}", file=out)


# Opening things


@contextmanager
def open_workspace(args: argparse.Namespace) -> Iterator[Workspace]:
    try:
        workspace = Workspace.open(args.workspace, allow_network_fs=args.allow_network_fs)
    except NetworkFilesystemError as err:
        raise CliError(str(err)) from err
    with workspace:
        yield workspace


# One interface for the in-process Workspace and a remote server: commands call a Backend and
# never ask which one it is.


class Backend(Protocol):
    def jobs(self, include_all: bool) -> list[dict[str, Any]]: ...

    def summary(self, job_id: str | None) -> dict[str, Any]: ...

    def cancel(self, job_id: str) -> str: ...

    def requeue(self, kind: Literal["resume", "retry"], job_id: str) -> Requeue: ...

    def export_lines(self, job_id: str, include_text: bool) -> Iterable[str]: ...

    def manifest_text(self, job_id: str) -> str: ...

    def drive(
        self, job_id: str, workers: WorkerSettings, make_classifier: ClassifierFactory
    ) -> int:
        """Run a re-queued Job to its end if this backend is the one running Jobs."""
        ...


class LocalBackend:
    def __init__(self, ws: Workspace) -> None:
        self.ws = ws

    def jobs(self, include_all: bool) -> list[dict[str, Any]]:
        jobs = []
        for info in self.ws.list_jobs(include_ingest=include_all):
            entry: dict[str, Any] = info.model_dump()
            try:
                with self.ws.read_job(info.id) as store:
                    entry["documents"] = len(store.states())
            except (OSError, StorageVersionError):
                entry["documents"] = None
            jobs.append(entry)
        return jobs

    def summary(self, job_id: str | None) -> dict[str, Any]:
        if job_id is None:
            listed = self.ws.list_jobs()
            if not listed:
                raise CliError("no Jobs in this Workspace")
            job_id = listed[-1].id
        try:
            return job_summary(self.ws, job_id)
        except KeyError:
            raise CliError(f"no such Job: {job_id}") from None

    def cancel(self, job_id: str) -> str:
        try:
            outcome = cancel_job(self.ws, job_id)
        except KeyError:
            raise CliError(f"no such Job: {job_id}") from None
        except JobFinishedError as err:
            raise CliError(str(err)) from err
        if outcome == "cancelled":
            return f"cancelled {job_id}"
        return f"cancel requested for {job_id}; the running Job stops shortly"

    def requeue(self, kind: Literal["resume", "retry"], job_id: str) -> Requeue:
        try:
            return (resume_job if kind == "resume" else retry_failed)(self.ws, job_id)
        except KeyError:
            raise CliError(f"no such Job: {job_id}") from None
        except (JobRunningError, JobPrunedError, StorageVersionError) as err:
            raise CliError(str(err)) from err

    def export_lines(self, job_id: str, include_text: bool) -> Iterator[str]:
        try:
            self.ws.get_job(job_id)
        except KeyError:
            raise CliError(f"no such Job: {job_id}") from None
        with self.ws.read_job(job_id) as store:
            for doc_id in store.states():
                record = store.get_record(doc_id, include_text=include_text)
                if record is not None:  # a running Job exports what is finished so far
                    yield record.model_dump_json(exclude_none=True)

    def manifest_text(self, job_id: str) -> str:
        return manifest_path(self.ws, job_id).read_text("utf-8")

    def drive(
        self, job_id: str, workers: WorkerSettings, make_classifier: ClassifierFactory
    ) -> int:
        """Run the Job here (or follow the runner that holds the lease), progress to stderr."""
        return asyncio.run(self._drive(job_id, workers, make_classifier))

    async def _drive(
        self, job_id: str, workers: WorkerSettings, make_classifier: ClassifierFactory
    ) -> int:
        runner = Runner(
            self.ws, make_classifier(load_manifest(self.ws, job_id).model), workers=workers
        )
        failed = 0
        async for record in stream_job(self.ws, runner, job_id):
            failed += record.status == "failed"
            print(f"{record.source.filename}: {record.status}", file=sys.stderr)
        return 1 if failed else 0


class ServerBackend:
    def __init__(self, client: ServerClient) -> None:
        self.client = client

    def jobs(self, include_all: bool) -> list[dict[str, Any]]:
        return self.client.jobs(include_ingest=include_all)

    def summary(self, job_id: str | None) -> dict[str, Any]:
        if job_id is None:
            jobs = self.client.jobs()
            if not jobs:
                raise CliError("no Jobs on the server")
            job_id = jobs[-1]["id"]
        return self.client.summary(job_id)

    def cancel(self, job_id: str) -> str:
        self.client.cancel(job_id)
        return f"cancel requested for {job_id}"

    def requeue(self, kind: Literal["resume", "retry"], job_id: str) -> Requeue:
        (self.client.resume if kind == "resume" else self.client.retry)(job_id)
        return Requeue(0, requested=True)

    def export_lines(self, job_id: str, include_text: bool) -> Iterable[str]:
        return self.client.export(job_id, include_text=include_text)

    def manifest_text(self, job_id: str) -> str:
        return self.client.manifest(job_id)

    def drive(
        self, job_id: str, workers: WorkerSettings, make_classifier: ClassifierFactory
    ) -> int:
        return 0  # the server runs it


@contextmanager
def open_backend(args: argparse.Namespace) -> Iterator[Backend]:
    if args.server:
        with ServerClient(args.server) as client:
            yield ServerBackend(client)
    else:
        with open_workspace(args) as ws:
            yield LocalBackend(ws)


# run


def write_manifest_beside(output: Path, manifest_text: str) -> None:
    """The Job manifest goes beside the JSONL (spec §6): `out.jsonl` -> `out.manifest.json`."""
    output.with_name(output.stem + ".manifest.json").write_text(manifest_text, encoding="utf-8")


async def _run(
    spec: JobSpec, out: TextIO, workers: WorkerSettings, ws: Workspace, classifier: Classifier
) -> tuple[int, str | None]:
    failed = 0
    job_id: str | None = None
    async for record in run(spec, classifier, workers, workspace=ws):
        out.write(record.model_dump_json(exclude_none=True) + "\n")
        out.flush()
        failed += record.status == "failed"
        job_id = record.job_id
        print(f"{record.source.filename}: {record.status}", file=sys.stderr)
    return (1 if failed else 0), job_id


def run_command(args: argparse.Namespace, make_classifier: ClassifierFactory) -> int:
    spec = build_spec(args)
    workers = WorkerSettings(args.ocr_workers, args.threads_per_worker)
    with open_workspace(args) as ws:
        classifier = make_classifier(spec.model)
        try:
            if args.output:
                with args.output.open("w", encoding="utf-8") as out:
                    code, job_id = asyncio.run(_run(spec, out, workers, ws, classifier))
                if job_id is not None:
                    write_manifest_beside(args.output, LocalBackend(ws).manifest_text(job_id))
                return code
            return asyncio.run(_run(spec, sys.stdout, workers, ws, classifier))[0]
        except SpecError as err:
            _findings(err.findings, "error", sys.stderr)
            return 2


def validate_command(args: argparse.Namespace, make_classifier: ClassifierFactory) -> int:
    spec = build_spec(args)
    workspace = resolve_workspace_path(args.workspace)
    config = load_config(workspace)
    languages = installed_languages([VENDORED_TESSDATA, workspace_tessdata(workspace)])
    report = validate(
        spec,
        capabilities=make_classifier(spec.model).capabilities,
        installed_languages=languages,
        rate_limit=config.rate_limit,
        prices=config.prices,
    )
    _findings(report.errors, "error", sys.stderr)
    _findings(report.warnings, "warning", sys.stderr)
    estimate = report.estimate
    line = f"{estimate.documents} Documents"
    if estimate.cost_usd is not None:
        line += f"; estimated cost up to ${estimate.cost_usd:.4f}"
    if estimate.duration_seconds is not None:
        line += f"; Classifier-bound duration about {estimate.duration_seconds:.0f}s"
    print(line)
    return 0 if report.ok else 2


# status, jobs, cancel, resume, retry


def format_summary(summary: dict[str, Any]) -> str:
    progress, classifier = summary["progress"], summary["classifier"]
    extraction, flags = summary["extraction"], summary["flags"]
    counts = ", ".join(f"{n} {state}" for state, n in sorted(progress["by_state"].items()))
    lines = [
        f"Job {summary['job_id']} ({summary['kind']}): {summary['state']}",
        f"documents: {progress['documents']}" + (f" ({counts})" if counts else ""),
    ]
    if progress["eta_seconds"] is not None:
        lines.append(f"eta: {progress['eta_seconds']:.0f}s at {progress['throughput_per_s']:.2f}/s")
    cost = f"${classifier['estimated_cost_usd']:.4f}"
    lines.append(
        f"classifier: {classifier['model']}, {classifier['requests']} requests, "
        f"{classifier['input_tokens']} input tokens, {classifier['cache_hits']} cache hits, "
        f"estimated cost {cost}" + ("" if classifier["priced"] else " (model unpriced)")
    )
    lines.append(f"extraction: {extraction['pages']} pages, {extraction['pages_ocr']} OCRed")
    for code, entry in sorted(summary["failures"].items()):
        lines.append(f"failures: {code} x{entry['count']} (e.g. {', '.join(entry['examples'])})")
    if summary["max_cost_usd"] is not None:
        lines.append(f"spend cap: ${summary['max_cost_usd']}")
    if flags["cancel_reason"]:
        lines.append(f"cancelled: {flags['cancel_reason']}")
    if flags["stalled"]:
        lines.append("stalled: no Classifier success for 15 minutes")
    return "\n".join(lines)


def _job_line(job: dict[str, Any]) -> str:
    when = datetime.fromtimestamp(job["created_at"], UTC).strftime("%Y-%m-%d %H:%M")
    docs = f"{job['documents']} docs" if job.get("documents") is not None else ""
    return f"{job['id']}  {job['state']:<9} {job['kind']:<6} {docs:<10} {when}"


def _jobs(args: argparse.Namespace) -> int:
    with open_backend(args) as backend:
        for job in backend.jobs(args.all):
            print(_job_line(job))
    return 0


def _status(args: argparse.Namespace) -> int:
    with open_backend(args) as backend:
        summary = backend.summary(args.job_id)
    print(json.dumps(summary, indent=2) if args.json else format_summary(summary))
    return 0


def _cancel(args: argparse.Namespace) -> int:
    with open_backend(args) as backend:
        print(backend.cancel(args.job_id))
    return 0


def _requeue(args: argparse.Namespace, make_classifier: ClassifierFactory) -> int:
    with open_backend(args) as backend:
        result = backend.requeue(args.command, args.job_id)
        if result.requested:
            print(f"{args.command} requested for {args.job_id}; the running runner applies it")
            return 0
        if result.count == 0:
            print(f"nothing to {args.command} in {args.job_id}")
            return 0
        print(f"re-queued {result.count} Documents of {args.job_id}", file=sys.stderr)
        workers = WorkerSettings(args.ocr_workers, args.threads_per_worker)
        return backend.drive(args.job_id, workers, make_classifier)


# records and export


def _records(args: argparse.Namespace) -> int:
    with open_workspace(args) as ws:
        try:
            store = ws.read_job(args.job_id)
        except KeyError:
            raise CliError(f"no such Job: {args.job_id}") from None
        with store:
            for doc_id in store.find_documents(
                status=args.status, escalation_status="flagged" if args.flagged else None
            ):
                record = store.get_record(doc_id, include_text=args.include_text)
                if record is not None:
                    print(record.model_dump_json(exclude_none=True))
    return 0


def _export(args: argparse.Namespace) -> int:
    with open_backend(args) as backend:
        lines = backend.export_lines(args.job_id, args.include_text)
        if args.output is None:
            for line in lines:
                print(line)
            return 0
        with args.output.open("w", encoding="utf-8") as out:
            for line in lines:
                out.write(line + "\n")
        write_manifest_beside(args.output, backend.manifest_text(args.job_id))
    return 0


# Dispatch


def dispatch(args: argparse.Namespace, make_classifier: ClassifierFactory) -> int:
    """Run one Job command; a `CliError` or `ServerError` becomes a message and an exit code."""
    try:
        if args.server and args.command not in SERVER_COMMANDS:
            raise CliError(
                f"`{args.command}` does not support --server yet; run it against the Workspace",
                2,
            )
        match args.command:
            case "run":
                return run_command(args, make_classifier)
            case "validate":
                return validate_command(args, make_classifier)
            case "status":
                return _status(args)
            case "jobs":
                return _jobs(args)
            case "cancel":
                return _cancel(args)
            case "resume" | "retry":
                return _requeue(args, make_classifier)
            case "records":
                return _records(args)
            case "export":
                return _export(args)
        raise CliError(f"unknown command {args.command}", 2)
    except (CliError, ServerError) as err:
        print(f"acceleread: {err}", file=sys.stderr)
        return err.code if isinstance(err, CliError) else 1
