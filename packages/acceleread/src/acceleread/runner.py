# SPDX-License-Identifier: Apache-2.0
"""The in-library Job runner (docs/spec/v0.md §7.1-§7.3, ADR 0002, ADR 0008, ADR 0010).

One runner per Workspace holds the runner lease and runs one Job at a time, FIFO. A Job goes
queued -> extracting -> `extracted` (its Record, with Pages, Sections and text, persisted at once)
-> classifying -> done or failed; a cancel ends with `cancelled`. Extraction stops while too many
Documents await the Classifier. The Job-level policies live here: a spend cap and a Classifier
422 or a worker crash rate auto-cancel the Job, and a lost lease stops it.

Nothing here logs Document text or Classifier state.
"""

import asyncio
import glob
import os
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

from acceleread.classifier import Classifier
from acceleread.config import Config, load_config
from acceleread.extract import VENDORED_TESSDATA
from acceleread.jev import JEV_RATE_LIMIT, JevClassifier
from acceleread.jobcontrol import (
    cancel_marker,
    finalize_job,
    finish_cancelled,
    load_manifest,
    read_cancel_request,
    record_cost,
    sweep_unfinished,
)
from acceleread.languages import installed_languages, workspace_tessdata
from acceleread.models import DocumentRecord, JobSpec, ResolvedManifest
from acceleread.pipeline import (
    ClassifyResult,
    classify_stage,
    document_format,
    extract_stage,
    failed,
    has_extraction,
    judgments_of,
    make_extract_task,
    new_record,
)
from acceleread.planner import JudgmentSpec
from acceleread.ratelimit import MAX_IN_FLIGHT, RateLimit, RateLimitedClassifier
from acceleread.validate import resolve
from acceleread.workers import WorkerPool, WorkerSettings
from acceleread.workspace import (
    DocumentState,
    InputChangedError,
    JobKind,
    JobState,
    JobStore,
    Workspace,
)
from acceleread.workspace.catalog import DEFAULT_LEASE_TTL, FINISHED_STATES

BACKPRESSURE_LIMIT = 500  # Documents allowed to await the Classifier (spec §7.3)
POLL_INTERVAL = 0.5  # seconds between the monitor's checks (cancel marker, heartbeat, crash rate)
UNKNOWN_CLASSIFIER_CEILING = RateLimit(tokens_per_s=1e9, requests_per_s=1e6)
_GLOB = ("*", "?", "[")


class LeaseLostError(RuntimeError):
    """This runner's lease went stale and another runner took it: stop without finalising."""


def default_ceiling(classifier: Classifier) -> RateLimit:
    """80% of the published limits (spec §7.5). Only Jev's are known so far."""
    if classifier.capabilities.classifier_id == "jev":
        return JEV_RATE_LIMIT
    return UNKNOWN_CLASSIFIER_CEILING


def expand_inputs(spec: JobSpec) -> JobSpec:
    """Replace each glob input with the files it matches, carrying its overrides along."""
    inputs = []
    overrides = dict(spec.overrides)
    for document in spec.inputs:
        source = document.source
        if "://" in source or Path(source).exists() or not any(c in source for c in _GLOB):
            inputs.append(document)
            continue
        matches = sorted(m for m in glob.glob(source, recursive=True) if Path(m).is_file())
        if not matches:
            inputs.append(document)  # reported when this Document fails to extract
            continue
        inputs += [document.model_copy(update={"source": m}) for m in matches]
        if source in overrides:
            override = overrides.pop(source)
            overrides.update({m: override for m in matches})
    return spec.model_copy(update={"inputs": inputs, "overrides": overrides})


class Runner:
    def __init__(
        self,
        workspace: Workspace,
        classifier: Classifier,
        *,
        workers: WorkerSettings | None = None,
        config: Config | None = None,
        holder: str | None = None,
        backpressure: int = BACKPRESSURE_LIMIT,
        classify_concurrency: int = MAX_IN_FLIGHT,
        poll_interval: float = POLL_INTERVAL,
        lease_ttl: float = DEFAULT_LEASE_TTL,
    ) -> None:
        self.workspace = workspace
        self.config = config if config is not None else load_config(workspace.path)
        # One limiter per Classifier per runner (spec §7.5).
        self.classifier: RateLimitedClassifier = (
            classifier
            if isinstance(classifier, RateLimitedClassifier)
            else RateLimitedClassifier(
                classifier, self.config.rate_limit or default_ceiling(classifier)
            )
        )
        self.workers = workers
        self.holder = holder or f"runner-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.backpressure = backpressure
        self.classify_concurrency = classify_concurrency
        self.poll_interval = poll_interval
        self.lease_ttl = lease_ttl
        self.holds_lease = False

    # Submitting

    def submit(self, spec: JobSpec, *, kind: JobKind = "job") -> str:
        """Validate the spec, write its manifest and queue every Document. Raises `SpecError`."""
        ws = self.workspace
        spec = expand_inputs(spec)
        manifest = resolve(
            spec,
            capabilities=self.classifier.capabilities,
            installed_languages=installed_languages(
                [VENDORED_TESSDATA, workspace_tessdata(ws.path)]
            ),
        )
        job = ws.create_job(manifest.model_dump(mode="json"), kind=kind)
        inputs = ws.job_inputs(job.id)
        with ws.open_job(job.id) as store:
            doc_ids = [f"{i:06d}" for i in range(len(manifest.inputs))]
            store.add_documents(doc_ids)
            for doc_id, document in zip(doc_ids, manifest.inputs, strict=True):
                if "://" in document.source:
                    continue  # URL inputs arrive with the API (#41); the Document fails
                with suppress(OSError):
                    store.set_input_ref(doc_id, inputs.reference_path(Path(document.source)))
        return job.id

    # The lease and the queue

    def acquire(self) -> bool:
        """Take the runner lease, and once held, requeue Jobs a dead runner left `running`."""
        self.holds_lease = self.workspace.acquire_lease(self.holder, self.lease_ttl)
        if self.holds_lease:
            self._recover()
        return self.holds_lease

    def release(self) -> None:
        if self.holds_lease:
            self.workspace.release_lease(self.holder)
            self.holds_lease = False

    def _recover(self) -> None:
        """We hold the lease, so no other runner is running a Job: a `running` Job is orphaned."""
        ws = self.workspace
        for job in ws.list_jobs(include_ingest=True):
            if job.state != "running":
                continue
            if (reason := read_cancel_request(ws, job.id)) is not None:
                finish_cancelled(ws, job.id, reason)
            else:
                ws.set_job_state(job.id, "queued")

    async def drain(self, until: str | None = None) -> None:
        """Run queued Jobs in FIFO order until the queue is empty, or `until` has been run."""
        if not (self.holds_lease or self.acquire()):
            raise LeaseLostError("another runner holds the lease")
        try:
            while (job_id := self.workspace.next_queued()) is not None:
                await self.execute(job_id)
                if job_id == until:
                    return
        finally:
            self.release()

    async def serve(self, stop: asyncio.Event) -> None:
        """Run Jobs as they are queued until `stop` is set: the long-lived runner (`serve`).

        It holds the lease while idle too, so a `run` from another process joins its FIFO.
        """
        self.acquire()
        while not stop.is_set():
            job_id = self.workspace.next_queued()
            if job_id is not None and (self.holds_lease or self.acquire()):
                await self.execute(job_id)
                continue
            if self.holds_lease:
                self.holds_lease = self.workspace.heartbeat(self.holder)
            with suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), self.poll_interval)
        self.release()

    # Executing

    async def execute(self, job_id: str) -> JobState:
        """Run one Job to a terminal state (done, failed or cancelled) and freeze its summary."""
        ws = self.workspace
        manifest = load_manifest(ws, job_id)
        store = ws.open_job(job_id)
        ws.set_job_state(job_id, "running")
        run = _JobRun(self, job_id, manifest, store)
        try:
            return await run.run()
        except LeaseLostError:
            raise  # the new holder requeues the Job and carries on
        except asyncio.CancelledError:
            sweep_unfinished(ws, job_id, store, manifest)
            finalize_job(ws, job_id, "cancelled", cancel_reason="user", prices=self.config.prices)
            raise
        except Exception:
            sweep_unfinished(ws, job_id, store, manifest)
            finalize_job(ws, job_id, "failed", prices=self.config.prices)
            raise
        finally:
            store.close()


class _JobRun:
    """The state of one Job while it runs."""

    def __init__(
        self, runner: Runner, job_id: str, manifest: ResolvedManifest, store: JobStore
    ) -> None:
        self.runner = runner
        self.ws = runner.workspace
        self.job_id = job_id
        self.manifest = manifest
        self.store = store
        self.specs: list[JudgmentSpec] = judgments_of(manifest)
        self.cache = self.ws.cache if manifest.cache else None
        self.stop = asyncio.Event()
        self.cancel_reason: str | None = None
        self.lease_lost = False
        self.spent = 0.0
        self.pool: WorkerPool | None = None
        # Documents allowed between "extraction began" and "classification began".
        self.room = asyncio.Semaphore(max(1, runner.backpressure))
        self.queue: asyncio.Queue[str | None] = asyncio.Queue()
        self.model = runner.classifier.capabilities.model

    def trigger_cancel(self, reason: str) -> None:
        if self.cancel_reason is None:
            self.cancel_reason = reason
        self.stop.set()
        if self.pool is not None:
            self.pool.cancel(reason)

    def _cap(self) -> float | None:
        """`max_cost_usd`, re-read each time because it may change while the Job runs."""
        return load_manifest(self.ws, self.job_id).max_cost_usd

    def check_cap(self) -> None:
        cap = self._cap()
        if cap is not None and self.spent >= cap:
            self.trigger_cancel("spend_cap")

    async def run(self) -> JobState:
        runner = self.runner
        for doc_id, state in self.store.states().items():
            if state == "extracting":  # a runner died mid-extraction
                self.store.set_state(doc_id, "queued")
        todo = deque(
            d for d, s in self.store.states().items() if s in ("queued", "extracted", "classifying")
        )
        self.spent = sum(
            record_cost(rec, self.model, runner.config.prices)
            for d in self.store.find_documents()
            if (rec := self.store.get_record(d, include_text=False)) is not None
        )
        self.check_cap()
        if todo and runner.workers is not None and not self.stop.is_set():
            self.pool = runner.workers.pool(self.manifest.extraction_profile)
        pool_size = self.pool.size if self.pool else 1  # in-process extraction runs one at a time
        extractors = [asyncio.create_task(self.extract_loop(todo)) for _ in range(pool_size)]
        consumers = [
            asyncio.create_task(self.classify_loop())
            for _ in range(max(1, runner.classify_concurrency))
        ]

        async def pipeline() -> None:
            await asyncio.gather(*extractors)
            for _ in consumers:
                self.queue.put_nowait(None)
            await asyncio.gather(*consumers)

        main = asyncio.create_task(pipeline())
        monitor = asyncio.create_task(self.monitor())
        stopper = asyncio.create_task(self.stop.wait())
        try:
            if not self.stop.is_set():
                await asyncio.wait({main, stopper}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (*extractors, *consumers, main, monitor, stopper):
                task.cancel()
            await asyncio.gather(
                *extractors, *consumers, main, monitor, stopper, return_exceptions=True
            )
            if self.pool is not None:
                self.pool.close()
        if main.done() and not main.cancelled() and (error := main.exception()) is not None:
            raise error
        if self.lease_lost:
            raise LeaseLostError("the runner lease was lost")
        if self.cancel_reason is not None and sweep_unfinished(
            self.ws, self.job_id, self.store, self.manifest
        ):  # a cancel that arrives after the last Document finished cancels nothing
            finalize_job(
                self.ws,
                self.job_id,
                "cancelled",
                cancel_reason=self.cancel_reason,
                live=self.live(),
                prices=runner.config.prices,
            )
            return "cancelled"
        finalize_job(self.ws, self.job_id, "done", live=self.live(), prices=runner.config.prices)
        return "done"

    def live(self) -> dict[str, Any]:
        """What only the running runner knows, frozen into the summary."""
        limiter = self.runner.classifier
        return {
            "stalled": limiter.stalled,
            "classifier": {
                "effective_rate": {
                    "tokens_per_s": limiter.rate.tokens_per_s,
                    "requests_per_s": limiter.rate.requests_per_s,
                },
                "ceiling": {
                    "tokens_per_s": limiter.ceiling.tokens_per_s,
                    "requests_per_s": limiter.ceiling.requests_per_s,
                },
            },
        }

    async def monitor(self) -> None:
        """Heartbeat the lease, and watch for a cancel request or the pool's crash-rate cancel."""
        runner = self.runner
        beat_every = runner.lease_ttl / 3
        last_beat = time.monotonic()
        while True:
            await asyncio.sleep(runner.poll_interval)
            if runner.holds_lease and time.monotonic() - last_beat >= beat_every:
                last_beat = time.monotonic()
                if not self.ws.heartbeat(runner.holder) and not self.ws.acquire_lease(
                    runner.holder, runner.lease_ttl
                ):
                    self.lease_lost = True
                    self.stop.set()
                    return
            if (reason := read_cancel_request(self.ws, self.job_id)) is not None:
                self.trigger_cancel(reason)
            if self.pool is not None and self.pool.cancel_reason is not None:
                self.trigger_cancel(self.pool.cancel_reason)

    # Extraction

    async def extract_loop(self, todo: deque[str]) -> None:
        while todo and not self.stop.is_set():
            await self.room.acquire()  # backpressure: stop extracting while the Classifier lags
            if self.stop.is_set() or not todo:
                self.room.release()
                return
            doc_id = todo.popleft()
            queued = await self.extract_one(doc_id)
            if queued:
                self.queue.put_nowait(doc_id)
            else:
                self.room.release()

    async def extract_one(self, doc_id: str) -> bool:
        """Extract (or reuse the stored extraction of) one Document. True if it awaits the
        Classifier; False if it ended here, as failed or because the pool was cancelled."""
        store, manifest = self.store, self.manifest
        document = manifest.inputs[int(doc_id)]
        override = manifest.overrides.get(document.source)
        profile = (override and override.extraction_profile) or manifest.extraction_profile
        languages = (override and override.ocr_languages) or manifest.ocr_languages
        prior = store.get_record(doc_id)
        attempts = 1 if prior is None else prior.attempts + (prior.status == "failed")
        if has_extraction(prior):
            assert prior is not None
            prior.attempts = attempts
            store.save_record(doc_id, "extracted", prior)
            return True
        ref = store.get_input_ref(doc_id)
        record = new_record(self.job_id, document, ref, profile, manifest.taxonomy, attempts)
        store.set_state(doc_id, "extracting")
        fmt = document_format(document.source)
        if fmt is None or "://" in document.source:
            failed(record, "extract", "unsupported_input", f"cannot ingest {document.source}")
        elif ref is None:
            failed(record, "extract", "input_unavailable", "the input file is missing")
        else:
            try:
                self.ws.job_inputs(self.job_id).open(ref).close()
            except InputChangedError as exc:
                failed(record, "extract", "input_changed", str(exc))
            except OSError as exc:
                failed(record, "extract", "input_unavailable", str(exc))
            else:
                path = (
                    self.ws.job_dir(self.job_id) / ref.path
                    if ref.kind == "copy"
                    else Path(ref.path)
                )
                task = make_extract_task(path, fmt, languages, self.ws.path)
                record = await extract_stage(record, task, path, self.pool)
        if record.status == "failed":
            if record.errors and record.errors[-1].code == "cancelled":
                if self.pool is not None and self.pool.cancel_reason is not None:
                    self.trigger_cancel(self.pool.cancel_reason)  # e.g. crash_rate
                return False  # the pool was cancelled; the sweep records this Document
            store.save_record(doc_id, "failed", record)
            return False
        store.save_record(doc_id, "extracted", record)  # persisted at once, before classifying
        return True

    # Classification

    async def classify_loop(self) -> None:
        while (doc_id := await self.queue.get()) is not None:
            self.room.release()
            await self.classify_one(doc_id)

    async def classify_one(self, doc_id: str) -> None:
        store = self.store
        record = store.get_record(doc_id)
        if record is None:
            return
        store.set_state(doc_id, "classifying")
        before = record_cost(record, self.model, self.runner.config.prices)
        result: ClassifyResult = await classify_stage(
            record, self.manifest, self.specs, self.runner.classifier, self.cache
        )
        record = await self.after_first_pass(result.record)
        # A Document whose classification failed (or was rejected) still keeps its extraction.
        state: DocumentState = (
            "failed"
            if record.status == "failed"
            else "cancelled"
            if record.status == "cancelled"
            else "done"
        )
        store.save_record(doc_id, state, record)
        self.spent += record_cost(record, self.model, self.runner.config.prices) - before
        if result.rejected:
            self.trigger_cancel("classifier_rejected")
        self.check_cap()

    async def after_first_pass(self, record: DocumentRecord) -> DocumentRecord:
        """SEAM for Escalation (#38): the first pass is done, the Record is not yet saved.

        #38 calls its escalation function here, with the Record's Judgments and the Job manifest,
        and returns the Record with `escalation` set on each Judgment. A Job with Escalation off
        passes through unchanged.
        """
        return record


# Streaming Records


async def follow_records(
    ws: Workspace,
    job_id: str,
    *,
    poll_interval: float = 0.1,
    is_active: Callable[[], bool] = lambda: True,
) -> AsyncIterator[DocumentRecord]:
    """Yield the Job's Records in input order as each Document finishes, until all are out.

    Works on the stored Records, so it follows a Job another runner executes. It stops early if
    the Job reaches a terminal state, or `is_active()` turns False, with Documents unfinished.
    """
    with ws.read_job(job_id) as store:
        doc_ids = list(store.states())
        for doc_id in doc_ids:
            while True:
                state = store.states().get(doc_id)
                if state in FINISHED_STATES and (record := store.get_record(doc_id)) is not None:
                    yield record
                    break
                if ws.get_job(job_id).state in FINISHED_STATES or not is_active():
                    # Final look: a Record may have landed between the two reads.
                    if (record := store.get_record(doc_id)) is not None and (
                        store.states().get(doc_id) in FINISHED_STATES
                    ):
                        yield record
                        break
                    return
                await asyncio.sleep(poll_interval)


async def stream_job(
    ws: Workspace, runner: Runner, job_id: str, *, poll_interval: float = 0.1
) -> AsyncIterator[DocumentRecord]:
    """Run an already-queued Job here when we can take the runner lease, else follow the runner
    (`serve`) that holds it; either way yield its Records in input order.

    If the lease frees up while we wait (the other runner died or finished), we run the Job
    ourselves. Abandoning the iterator cancels a Job that is still running.
    """
    driver: asyncio.Task[None] | None = None

    def start_driver() -> None:
        nonlocal driver
        if (driver is None or driver.done()) and runner.acquire():
            driver = asyncio.create_task(runner.drain(until=job_id))

    async def takeover() -> None:
        while ws.get_job(job_id).state not in FINISHED_STATES:
            start_driver()
            await asyncio.sleep(poll_interval)

    start_driver()
    watcher = asyncio.create_task(takeover())
    try:
        async for record in follow_records(ws, job_id, poll_interval=poll_interval):
            yield record
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
        if driver is not None:
            if ws.get_job(job_id).state not in FINISHED_STATES:  # the consumer stopped early
                cancel_marker(ws, job_id).write_text("user", encoding="utf-8")
            await asyncio.gather(driver, return_exceptions=False)


async def run_job(
    spec: JobSpec,
    classifier: Classifier | None,
    workers: WorkerSettings | None,
    workspace: Workspace | None,
    *,
    poll_interval: float = 0.1,
) -> AsyncIterator[DocumentRecord]:
    """Submit a Job and stream its Records (see `stream_job`)."""
    owned = workspace is None
    ws = workspace if workspace is not None else Workspace.open()
    try:
        runner = Runner(
            ws,
            classifier or JevClassifier(model=spec.model),
            workers=workers,
            poll_interval=poll_interval,
        )
        job_id = runner.submit(spec)
        async for record in stream_job(ws, runner, job_id, poll_interval=poll_interval):
            yield record
    finally:
        if owned:
            ws.close()
