# SPDX-License-Identifier: Apache-2.0
"""Job control over a Workspace: cancel, resume, retry, the spend cap and the Job summary.

Everything here works on stored state only, so a CLI invocation or an API handler can use it
without running a Classifier. A running Job lives in another process (or task); it is told to
stop through a marker file in its directory, which its runner polls.
"""

import json
import math
import os
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from acceleread.models import DocumentRecord, Judgment, ResolvedManifest
from acceleread.pipeline import new_record
from acceleread.pricing import estimate_cost_usd, is_priced
from acceleread.workspace import (
    JobRunningError,
    JobState,
    JobStore,
    Workspace,
)
from acceleread.workspace.catalog import FINISHED_STATES

CANCEL_REASONS = ("user", "crash_rate", "classifier_rejected", "spend_cap")
MUTABLE_MANIFEST_FIELDS = frozenset({"max_cost_usd"})
LIVE_STATES = ("queued", "extracting", "extracted", "classifying")


class JobFinishedError(RuntimeError):
    """The Job has already finished, so there is nothing to cancel."""


class ManifestImmutableError(ValueError):
    """The manifest is immutable once the Job starts, except `max_cost_usd` (spec §7.1)."""


def manifest_path(ws: Workspace, job_id: str) -> Path:
    return ws.job_dir(job_id) / "manifest.json"


def load_manifest(ws: Workspace, job_id: str) -> ResolvedManifest:
    return ResolvedManifest.model_validate_json(manifest_path(ws, job_id).read_text("utf-8"))


def _write_atomic(path: Path, text: str) -> None:
    partial = path.with_name(path.name + ".partial")
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


# Cost


def record_cost(
    record: DocumentRecord, default_model: str, prices: Mapping[str, float] | None = None
) -> float:
    """Estimated USD for what this Record's Judgments used. Cache hits cost nothing."""
    judged = [record.classification, *record.answers.values()]
    model = next((j.classifier.model for j in judged if isinstance(j, Judgment)), default_model)
    return estimate_cost_usd(model, record.usage.input_tokens, record.usage.output_tokens, prices)


# Cancel, resume and retry


def cancel_marker(ws: Workspace, job_id: str) -> Path:
    return ws.job_dir(job_id) / "cancel.request"


def read_cancel_request(ws: Workspace, job_id: str) -> str | None:
    try:
        return cancel_marker(ws, job_id).read_text("utf-8").strip() or "user"
    except FileNotFoundError:
        return None


def cancel_job(
    ws: Workspace, job_id: str, reason: str = "user"
) -> Literal["cancelled", "requested"]:
    """Cancel a Job. A queued Job (or one whose runner died) is cancelled at once; one a live
    runner is executing is asked to stop and finishes cancelling itself."""
    info = ws.get_job(job_id)
    if info.state in FINISHED_STATES:
        raise JobFinishedError(f"Job {job_id} has already finished ({info.state})")
    if info.state == "running" and ws.lease_holder() is not None:
        cancel_marker(ws, job_id).write_text(reason, encoding="utf-8")
        return "requested"
    finish_cancelled(ws, job_id, reason)
    return "cancelled"


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
            ref = store.get_input_ref(doc_id)
            record = new_record(
                job_id,
                manifest.inputs[int(doc_id)],
                ref,
                manifest.extraction_profile,
                manifest.taxonomy,
            )
        record.status = "cancelled"
        store.save_record(doc_id, "cancelled", record)
        swept += 1
    return swept


def finish_cancelled(ws: Workspace, job_id: str, reason: str) -> None:
    manifest = load_manifest(ws, job_id)
    with ws.open_job(job_id) as store:
        sweep_unfinished(ws, job_id, store, manifest)
    finalize_job(ws, job_id, "cancelled", cancel_reason=reason)


def resume_job(ws: Workspace, job_id: str) -> int:
    """Re-queue `cancelled` and unstarted Documents. Returns how many; 0 leaves the Job as is."""
    info = ws.get_job(job_id)
    if info.state == "running" and ws.lease_holder() is not None:
        raise JobRunningError(f"Job {job_id} is running")
    with ws.open_job(job_id) as store:
        waiting = [d for d, s in store.states().items() if s in ("cancelled", *LIVE_STATES)]
        for doc_id in waiting:
            store.set_state(doc_id, "queued")
    if waiting:
        cancel_marker(ws, job_id).unlink(missing_ok=True)
        ws.set_job_state(job_id, "queued")
    return len(waiting)


def retry_failed(ws: Workspace, job_id: str) -> int:
    """Re-queue the Job's failed Documents under the same manifest. Returns how many.

    Each Document's Record stays until the new attempt replaces it, so a Document whose text was
    extracted is not extracted again, and `attempts` counts up.
    """
    info = ws.get_job(job_id)
    if info.state == "running" and ws.lease_holder() is not None:
        raise JobRunningError(f"Job {job_id} is running")
    with ws.open_job(job_id) as store:
        failed_docs = store.documents_in_state("failed")
        for doc_id in failed_docs:
            store.set_state(doc_id, "queued")
    if failed_docs:
        ws.set_job_state(job_id, "queued")
    return len(failed_docs)


# Summary (spec §11)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def job_summary(
    ws: Workspace,
    job_id: str,
    *,
    live: Mapping[str, Any] | None = None,
    cancel_reason: str | None = None,
    prices: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """The Job summary: progress, Classifier usage and cost (estimated), extraction, failures
    by reason and flags. `live` adds what only a running runner knows (the rate, `stalled`)."""
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
    end = info.finished_at if info.finished_at is not None else time.time()
    elapsed = max(end - info.created_at, 1e-9)
    throughput = finished / elapsed
    remaining = len(states) - finished
    cost = sum(record_cost(rec, manifest.model, prices) for _, rec in records)
    usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "cache_hits": 0}
    for _, rec in records:
        for name in usage:
            usage[name] += getattr(rec.usage, name)
    pages = sum(len(rec.pages) for _, rec in records)
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
            "estimated_cost_usd": cost,
            "cost_is_estimated": True,
            "priced": is_priced(manifest.model, prices),
            "price_table_version": manifest.price_table_version,
            **(live or {}).get("classifier", {}),
        },
        "extraction": {
            "pages": pages,
            "pages_ocr": sum(rec.usage.ocr_pages for _, rec in records),
            "pages_ocr_by_engine": dict(ocr_by_engine),
            "worker_seconds": sum(seconds),
            "p50_seconds": _percentile(seconds, 0.5),
            "p95_seconds": _percentile(seconds, 0.95),
        },
        "failures": failures,
        "flags": {
            "stalled": bool((live or {}).get("stalled", False)),
            "cancel_reason": cancel_reason,
        },
        "max_cost_usd": manifest.max_cost_usd,
    }


def finalize_job(
    ws: Workspace,
    job_id: str,
    state: JobState,
    *,
    cancel_reason: str | None = None,
    live: Mapping[str, Any] | None = None,
    prices: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """End a Job: freeze `summary.json`, then record the terminal state."""
    summary = job_summary(ws, job_id, live=live, cancel_reason=cancel_reason, prices=prices)
    summary["state"] = state
    ws.finish_job(job_id, state, summary)
    cancel_marker(ws, job_id).unlink(missing_ok=True)
    return summary
