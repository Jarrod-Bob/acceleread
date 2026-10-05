# SPDX-License-Identifier: Apache-2.0
"""Job control over a Workspace: cancel, resume, retry, the spend cap and the Job summary.

The runner lease holder is the single SQLite writer (docs/spec/v0.md §7.2). So a caller that is not
the holder never writes `job.sqlite` or the catalog. While a live holder exists, `cancel_job`,
`resume_job` and `retry_failed` post a small request file (written atomically) that the holder
applies; with no live holder the caller takes the lease itself for the duration of the write.
"""

import json
import math
import os
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from acceleread.config import load_config
from acceleread.models import DocumentRecord, Judgment, ResolvedManifest
from acceleread.pipeline import new_record
from acceleread.pricing import estimate_cost_usd, is_priced
from acceleread.ratelimit import STALL_AFTER
from acceleread.workspace import (
    JobPrunedError,
    JobRunningError,
    JobState,
    JobStore,
    StorageVersionError,
    Workspace,
)
from acceleread.workspace.catalog import FINISHED_STATES

type CancelReason = Literal["user", "crash_rate", "classifier_rejected", "spend_cap"]
type RequestKind = Literal["cancel", "resume", "retry"]
# These end the Job `cancelled` even when every Document had finished; a user cancel that
# arrives after the last Document cancels nothing.
AUTO_CANCEL_REASONS: frozenset[str] = frozenset({"crash_rate", "classifier_rejected", "spend_cap"})
MUTABLE_MANIFEST_FIELDS = frozenset({"max_cost_usd"})
LIVE_STATES = ("queued", "extracting", "extracted", "classifying")


class JobFinishedError(RuntimeError):
    """The Job has already finished, so there is nothing to cancel."""


class ManifestImmutableError(ValueError):
    """The manifest is immutable once the Job starts, except `max_cost_usd` (spec §7.1)."""


@dataclass(frozen=True)
class Requeue:
    """What `resume_job` or `retry_failed` did: `requested` means the holder will apply it."""

    count: int
    requested: bool = False


# Manifest


def manifest_path(ws: Workspace, job_id: str) -> Path:
    return ws.job_dir(job_id) / "manifest.json"


def load_manifest(ws: Workspace, job_id: str) -> ResolvedManifest:
    return ResolvedManifest.model_validate_json(manifest_path(ws, job_id).read_text("utf-8"))


def _write_atomic(path: Path, text: str) -> None:
    partial = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, path)


def update_manifest(ws: Workspace, job_id: str, changes: Mapping[str, Any]) -> ResolvedManifest:
    """Change `max_cost_usd`, the only manifest field that may change after the Job starts."""
    ws.get_job(job_id)
    for name in changes:
        if name not in MUTABLE_MANIFEST_FIELDS:
            raise ManifestImmutableError(
                f"'{name}' cannot change: the manifest is immutable once the Job starts, "
                "except max_cost_usd. Submit a new Job; the Judgment cache keeps unchanged "
                "Judgments free."
            )
    current = load_manifest(ws, job_id)
    updated = ResolvedManifest.model_validate({**current.model_dump(mode="json"), **changes})
    _write_atomic(
        manifest_path(ws, job_id),
        json.dumps(updated.model_dump(mode="json"), indent=2, sort_keys=True),
    )
    return updated


# Cost: one place owns pricing, and it reads the Workspace config


def workspace_prices(ws: Workspace) -> dict[str, float]:
    return load_config(ws.path).prices


def record_cost(
    record: DocumentRecord, default_model: str, prices: Mapping[str, float] | None = None
) -> float:
    """Estimated USD for what this Record's Judgments used. Cache hits cost nothing."""
    judged = [record.classification, *record.answers.values()]
    model = next((j.classifier.model for j in judged if isinstance(j, Judgment)), default_model)
    return estimate_cost_usd(model, record.usage.input_tokens, record.usage.output_tokens, prices)


def _ledger(ws: Workspace, job_id: str) -> Path:
    return ws.job_dir(job_id) / "spend.log"


def add_spend(ws: Workspace, job_id: str, doc_id: str, usd: float) -> None:
    """Append what one classification attempt cost. The Record holds only the latest attempt's
    usage, so this ledger keeps the Job's true spend (and the spend cap) across retries."""
    if usd > 0:
        with _ledger(ws, job_id).open("a", encoding="utf-8") as ledger:
            ledger.write(f"{doc_id} {usd!r}\n")


def job_spend(ws: Workspace, job_id: str) -> float:
    try:
        text = _ledger(ws, job_id).read_text("utf-8")
    except FileNotFoundError:
        return 0.0
    return sum(float(line.split()[1]) for line in text.splitlines() if line.strip())


def progress_marker(ws: Workspace, job_id: str) -> Path:
    """Its mtime is the last time the Job made progress; `stalled` is derived from it."""
    return ws.job_dir(job_id) / "last_success"


def touch_progress(ws: Workspace, job_id: str) -> None:
    marker = progress_marker(ws, job_id)
    marker.touch()
    os.utime(marker, None)


# Control requests


@dataclass(frozen=True)
class ControlRequest:
    kind: RequestKind
    job_id: str
    reason: CancelReason | None
    path: Path


def _control_dir(ws: Workspace) -> Path:
    return ws.path / "control"


def post_request(
    ws: Workspace, kind: RequestKind, job_id: str, reason: CancelReason | None = None
) -> None:
    """Ask the lease holder to cancel, resume or retry a Job. Atomic: a reader never sees a
    half-written file, because the temp name is not one `pending_requests` reads."""
    directory = _control_dir(ws)
    directory.mkdir(exist_ok=True)
    name = f"{time.time_ns():020d}-{uuid.uuid4().hex[:8]}.json"
    body = json.dumps({"kind": kind, "job_id": job_id, "reason": reason or "user"})
    partial = directory / f".{name}.tmp"
    partial.write_text(body, encoding="utf-8")
    os.replace(partial, directory / name)


def pending_requests(ws: Workspace) -> list[ControlRequest]:
    directory = _control_dir(ws)
    if not directory.is_dir():
        return []
    found: list[ControlRequest] = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text("utf-8"))
            found.append(ControlRequest(data["kind"], data["job_id"], data.get("reason"), path))
        except (OSError, ValueError, KeyError):
            path.unlink(missing_ok=True)  # unreadable: nothing can act on it
    return found


def discard_requests(ws: Workspace, job_id: str, kind: RequestKind | None = None) -> None:
    for request in pending_requests(ws):
        if request.job_id == job_id and kind in (None, request.kind):
            request.path.unlink(missing_ok=True)


@contextmanager
def writer_lease(ws: Workspace) -> Iterator[bool]:
    """Take the runner lease for a write if no live holder has it. Yields whether we hold it."""
    holder = f"control-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    held = ws.acquire_lease(holder)
    try:
        yield held
    finally:
        if held:
            ws.release_lease(holder)


def recover_orphans(ws: Workspace) -> None:
    """Whoever holds the lease knows no Job is running: a `running` Job lost its runner."""
    for job in ws.list_jobs(include_ingest=True):
        if job.state == "running":
            ws.set_job_state(job.id, "queued")


# Direct writes (lease holder only)


def sweep_unfinished(
    ws: Workspace, job_id: str, store: JobStore, manifest: ResolvedManifest
) -> int:
    """Give every Document the Job left unprocessed a `cancelled` Record (spec §6).

    A Document that was already extracted keeps its text, Pages and Sections, so resume never
    redoes its OCR.
    """
    swept = 0
    for doc_id, state in store.states().items():
        if state not in LIVE_STATES:
            continue
        record = store.get_record(doc_id)
        if record is None:
            record = new_record(
                job_id,
                manifest.inputs[int(doc_id)],
                store.get_input_ref(doc_id),
                manifest.extraction_profile,
                manifest.taxonomy,
            )
        record.status = "cancelled"
        store.save_record(doc_id, "cancelled", record)
        swept += 1
    return swept


def finish_cancelled(ws: Workspace, job_id: str, reason: CancelReason) -> None:
    manifest = load_manifest(ws, job_id)
    with ws.open_job(job_id) as store:
        sweep_unfinished(ws, job_id, store, manifest)
    finalize_job(ws, job_id, "cancelled", cancel_reason=reason)


def resume_direct(ws: Workspace, job_id: str) -> int:
    with ws.open_job(job_id) as store:
        waiting = [d for d, s in store.states().items() if s in ("cancelled", *LIVE_STATES)]
        for doc_id in waiting:
            store.set_state(doc_id, "queued")
    if waiting:
        discard_requests(ws, job_id, "cancel")  # a stale cancel must not kill the resumed Job
        ws.set_job_state(job_id, "queued")
    return len(waiting)


def retry_direct(ws: Workspace, job_id: str) -> int:
    """Re-queue failed Documents. Each Record stays until its new attempt replaces it, so a
    Document whose text was extracted is not extracted again, and `attempts` counts up."""
    with ws.open_job(job_id) as store:
        failed_docs = store.documents_in_state("failed")
        for doc_id in failed_docs:
            store.set_state(doc_id, "queued")
    if failed_docs:
        discard_requests(ws, job_id, "cancel")
        ws.set_job_state(job_id, "queued")
    return len(failed_docs)


def apply_requests(
    ws: Workspace,
    *,
    running: str | None = None,
    on_cancel: Callable[[CancelReason], None] | None = None,
) -> int:
    """Apply posted requests, oldest first. Call only while holding the lease.

    `running` is the Job being executed now: a cancel for it goes to `on_cancel`; a resume or
    retry for it is refused. A request for a Job that is gone or finished is dropped.
    """
    applied = 0
    for request in pending_requests(ws):
        try:
            info = ws.get_job(request.job_id)
            if request.kind == "cancel":
                if info.state in FINISHED_STATES:
                    pass
                elif request.job_id == running and on_cancel is not None:
                    on_cancel(request.reason or "user")
                    applied += 1
                elif info.state == "queued":
                    finish_cancelled(ws, request.job_id, request.reason or "user")
                    applied += 1
            elif info.state != "running":
                direct = resume_direct if request.kind == "resume" else retry_direct
                direct(ws, request.job_id)
                applied += 1
        except (KeyError, JobPrunedError, StorageVersionError, FileNotFoundError):
            pass  # the Job is gone or cannot be written: drop the request
        request.path.unlink(missing_ok=True)
    return applied


# Public control: take the lease to write, or post a request to the live holder


def cancel_job(
    ws: Workspace, job_id: str, reason: CancelReason = "user"
) -> Literal["cancelled", "requested"]:
    info = ws.get_job(job_id)
    if info.state in FINISHED_STATES:
        raise JobFinishedError(f"Job {job_id} has already finished ({info.state})")
    with writer_lease(ws) as held:
        if held:
            recover_orphans(ws)
            apply_requests(ws)
            if ws.get_job(job_id).state not in FINISHED_STATES:
                finish_cancelled(ws, job_id, reason)
            return "cancelled"
    post_request(ws, "cancel", job_id, reason)
    return "requested"


def _requeue(
    ws: Workspace, job_id: str, kind: RequestKind, direct: Callable[[Workspace, str], int]
) -> Requeue:
    if ws.get_job(job_id).state == "running" and ws.lease_holder() is not None:
        raise JobRunningError(f"Job {job_id} is running")
    with writer_lease(ws) as held:
        if held:
            recover_orphans(ws)
            apply_requests(ws)
            return Requeue(direct(ws, job_id))
    post_request(ws, kind, job_id)
    return Requeue(0, requested=True)


def resume_job(ws: Workspace, job_id: str) -> Requeue:
    """Re-queue `cancelled` and unstarted Documents."""
    return _requeue(ws, job_id, "resume", resume_direct)


def retry_failed(ws: Workspace, job_id: str) -> Requeue:
    """Re-queue the Job's failed Documents under the same manifest."""
    return _requeue(ws, job_id, "retry", retry_direct)


# Summary (spec §11)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def _stalled(ws: Workspace, job_id: str, running: bool, waiting: int, now: float) -> bool:
    """Running with work waiting and no progress for 15 minutes, derived from the marker."""
    if not running or not waiting:
        return False
    try:
        last = progress_marker(ws, job_id).stat().st_mtime
    except FileNotFoundError:
        return False
    return now - last >= STALL_AFTER


def job_summary(
    ws: Workspace,
    job_id: str,
    *,
    live: Mapping[str, Any] | None = None,
    cancel_reason: str | None = None,
    error: str | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """The Job summary: progress, Classifier usage and cost (estimated), extraction, failures
    by reason and flags. `live` adds what only a running runner knows (the effective rate)."""
    now = time.time() if now is None else now
    prices = workspace_prices(ws)
    info = ws.get_job(job_id)
    manifest = load_manifest(ws, job_id)
    with ws.read_job(job_id) as store:
        states = store.states()
        records = [
            (doc_id, rec)
            for doc_id in store.find_documents()
            if (rec := store.get_record(doc_id, include_text=False)) is not None
        ]
    by_state = Counter(states.values())
    finished = sum(by_state[s] for s in ("done", "failed", "cancelled"))
    end = info.finished_at if info.finished_at is not None else now
    throughput = finished / max(end - info.created_at, 1e-9)
    remaining = len(states) - finished
    usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "cache_hits": 0}
    for _, rec in records:
        for name in usage:
            usage[name] += getattr(rec.usage, name)
    ocr_by_engine = Counter(
        p.engine or "unknown" for _, rec in records for p in rec.pages if p.method != "text-layer"
    )
    seconds = [rec.timings.extract_ms / 1000 for _, rec in records if rec.timings.extract_ms]
    failures: dict[str, dict[str, Any]] = {}
    for doc_id, rec in records:
        if rec.status in ("failed", "partial") and rec.errors:
            entry = failures.setdefault(rec.errors[0].code, {"count": 0, "examples": []})
            entry["count"] += 1
            if len(entry["examples"]) < 3:
                entry["examples"].append(doc_id)
    stalled = _stalled(ws, job_id, info.state == "running", remaining, now) or bool(
        (live or {}).get("stalled")
    )
    return {
        "job_id": job_id,
        "kind": info.kind,
        "state": info.state,
        "progress": {
            "documents": len(states),
            "by_state": dict(by_state),
            "throughput_per_s": throughput,
            "eta_seconds": remaining / throughput if throughput > 0 and remaining else None,
        },
        "classifier": {
            "model": manifest.model,
            **usage,
            "estimated_cost_usd": job_spend(ws, job_id),
            "cost_is_estimated": True,
            "priced": is_priced(manifest.model, prices),
            "price_table_version": manifest.price_table_version,
            **(live or {}).get("classifier", {}),
        },
        "extraction": {
            "pages": sum(len(rec.pages) for _, rec in records),
            "pages_ocr": sum(rec.usage.ocr_pages for _, rec in records),
            "pages_ocr_by_engine": dict(ocr_by_engine),
            "worker_seconds": sum(seconds),
            "p50_seconds": _percentile(seconds, 0.5),
            "p95_seconds": _percentile(seconds, 0.95),
        },
        "failures": failures,
        "flags": {"stalled": stalled, "cancel_reason": cancel_reason, "error": error},
        "max_cost_usd": manifest.max_cost_usd,
    }


def finalize_job(
    ws: Workspace,
    job_id: str,
    state: JobState,
    *,
    cancel_reason: CancelReason | None = None,
    live: Mapping[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """End a Job: freeze `summary.json`, record the terminal state, and drop requests that were
    waiting on it, so none outlives its Job."""
    summary = job_summary(ws, job_id, live=live, cancel_reason=cancel_reason, error=error)
    summary["state"] = state
    ws.finish_job(job_id, state, summary)
    discard_requests(ws, job_id, "cancel")
    return summary
