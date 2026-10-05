# SPDX-License-Identifier: Apache-2.0
"""The in-library Job runner (docs/spec/v0.md §7.1-§7.3, ADR 0002, ADR 0008, ADR 0010).

One runner per Workspace holds the runner lease and runs one Job at a time, FIFO. A Job goes
queued -> extracting -> `extracted` (its Record, with Pages, Sections and text, persisted at once)
-> classifying -> done or failed; a cancel ends with `cancelled`. Extraction stops while too many
Documents await the Classifier. The Job-level policies live here: a spend cap, a Classifier 422 or
a worker crash rate auto-cancel the Job, and a lost lease stops it.

The holder is the single SQLite writer: other processes post requests (`jobcontrol`) that the
holder applies. Nothing here logs Document text or Classifier state.
"""

import asyncio
import logging
import os
import sqlite3
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

from acceleread.classifier import (
    Ask,
    Capabilities,
    Classifier,
    ClassifierError,
    ClassifierResponse,
    JSONState,
)
from acceleread.config import load_config
from acceleread.extract import VENDORED_TESSDATA
from acceleread.jev import JEV_RATE_LIMIT, JevClassifier
from acceleread.jobcontrol import (
    AUTO_CANCEL_REASONS,
    CancelReason,
    add_spend,
    apply_requests,
    finalize_job,
    job_spend,
    load_manifest,
    post_request,
    record_cost,
    recover_orphans,
    sweep_unfinished,
    touch_progress,
    workspace_prices,
)
from acceleread.languages import installed_languages, workspace_tessdata
from acceleread.models import DocumentRecord, JobSpec, RecordError, ResolvedManifest, Usage
from acceleread.pipeline import (
    ClassifyResult,
    Extractor,
    classify_stage,
    document_format,
    expand_inputs,
    extract,
    extract_stage,
    failed,
    has_extraction,
    judgments_of,
    make_extract_task,
    new_record,
)
from acceleread.planner import JudgmentSpec
from acceleread.ratelimit import RateLimit, RateLimitedClassifier
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

logger = logging.getLogger(__name__)

BACKPRESSURE_LIMIT = 500  # Documents allowed to await the Classifier (spec §7.3)
POLL_INTERVAL = 0.5  # seconds between the monitor's checks (requests, heartbeat, crash rate)
CANCEL_GRACE = 30.0  # seconds a cancelled Job waits for Classifier calls already in flight
UNKNOWN_CLASSIFIER_CEILING = RateLimit(tokens_per_s=1e9, requests_per_s=1e6)

type Sleep = Callable[[float], Awaitable[None]]


class LeaseLostError(RuntimeError):
    """This runner's lease went stale and another runner took it: stop without finalising."""


class JobStopping(ClassifierError):
    """The Job is stopping: no new Classifier request may start (in-flight ones finish)."""


class _StopGate:
    """The Classifier as the Job's Documents see it: closed to new requests once the Job stops,
    so the grace period only lets calls already sent finish."""

    def __init__(self, inner: Classifier, stop: asyncio.Event) -> None:
        self._inner = inner
        self._stop = stop

    @property
    def capabilities(self) -> Capabilities:
        return self._inner.capabilities

    async def judge(self, state: JSONState, judgments: Mapping[str, Ask]) -> ClassifierResponse:
        if self._stop.is_set():
            raise JobStopping("the Job is stopping")
        return await self._inner.judge(state, judgments)


def default_ceiling(classifier: Classifier) -> RateLimit:
    """80% of the published limits (spec §7.5). Only Jev's are known so far."""
    if classifier.capabilities.classifier_id == "jev":
        return JEV_RATE_LIMIT
    return UNKNOWN_CLASSIFIER_CEILING


class Runner:
    def __init__(
        self,
        workspace: Workspace,
        classifier: Classifier,
        *,
        workers: WorkerSettings | None = None,
        holder: str | None = None,
        backpressure: int = BACKPRESSURE_LIMIT,
        classify_concurrency: int | None = None,
        poll_interval: float = POLL_INTERVAL,
        lease_ttl: float = DEFAULT_LEASE_TTL,
        cancel_grace: float = CANCEL_GRACE,
        extractor: Extractor = extract,
        clock: Callable[[], float] = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.workspace = workspace
        self.config = load_config(workspace.path)
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
        # Documents classified at once: the ceiling's `max_in_flight` unless a test says less.
        self.classify_concurrency = classify_concurrency or self.classifier.ceiling.max_in_flight
        self.poll_interval = poll_interval
        self.lease_ttl = lease_ttl
        self.cancel_grace = cancel_grace
        self.extractor = extractor
        self.clock = clock
        self.sleep = sleep
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
        """Take the runner lease. Once held, requeue Jobs a dead runner left `running` and apply
        requests posted while there was no holder."""
        self.holds_lease = self.workspace.acquire_lease(self.holder, self.lease_ttl)
        if self.holds_lease:
            recover_orphans(self.workspace)
            self.apply_requests()
        return self.holds_lease

    def release(self) -> None:
        if self.holds_lease:
            self.workspace.release_lease(self.holder)
            self.holds_lease = False

    def apply_requests(self) -> int:
        """Apply cancel, resume and retry requests other processes posted (we are the writer).
        Nothing a request can do (a locked database, a bad Job) is allowed to stop the runner."""
        try:
            return apply_requests(self.workspace)
        except Exception as exc:
            logger.error("could not apply requests: %s", type(exc).__name__)
            return 0

    def _queued(self, job_id: str) -> bool:
        try:
            return self.workspace.get_job(job_id).state == "queued"
        except KeyError:
            return False

    async def _run_one(self, job_id: str) -> bool:
        """Execute one Job. A Job that fails is logged and finalised `failed`, never fatal to the
        runner. False means the lease was lost: stop, whoever holds it now recovers the Job."""
        try:
            await self.execute(job_id)
        except LeaseLostError:
            self.holds_lease = False
            return False
        except Exception as exc:  # `execute` has already finalised the Job as failed
            logger.error("Job %s failed: %s", job_id, type(exc).__name__)
        return True

    async def drain(self, until: str | None = None) -> None:
        """Run queued Jobs in FIFO order until the queue is empty, or `until` has run, ended or
        gone: a caller that only wants its own Job never runs the Jobs queued after it.

        Waits while a short-lived control writer holds the lease, like `serve` does.
        """
        while not (self.holds_lease or self.acquire()):
            if until is not None and not self._queued(until):
                return
            await self.sleep(self.poll_interval)
        try:
            while until is None or self._queued(until):
                job_id = self.workspace.next_queued()
                if job_id is None or not await self._run_one(job_id):
                    return
                self.apply_requests()
        finally:
            self.release()

    async def serve(self, stop: asyncio.Event) -> None:
        """Run Jobs as they are queued until `stop` is set: the long-lived runner (`serve`).

        It holds the lease while idle too, so a `run` from another process joins its FIFO. A stop
        during a Job cancels it (with the usual grace) rather than waiting for it to finish.
        """
        ws = self.workspace
        try:
            while not stop.is_set():
                if not self.holds_lease and not self.acquire():
                    await self._idle(stop)
                    continue
                self.apply_requests()
                job_id = ws.next_queued()
                if job_id is not None:
                    await self._run_until_stop(job_id, stop)
                    continue
                self.holds_lease = ws.heartbeat(self.holder)
                await self._idle(stop)
        finally:
            self.release()

    async def _run_until_stop(self, job_id: str, stop: asyncio.Event) -> None:
        running = asyncio.create_task(self._run_one(job_id))
        stopper = asyncio.create_task(stop.wait())
        try:
            await asyncio.wait({running, stopper}, return_when=asyncio.FIRST_COMPLETED)
            if not running.done():
                post_request(self.workspace, "cancel", job_id, "user")
                await running
        finally:
            stopper.cancel()

    async def _idle(self, stop: asyncio.Event) -> None:
        with suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), self.poll_interval)

    # Executing

    async def execute(self, job_id: str) -> JobState:
        """Run one Job to a terminal state (done, failed or cancelled) and freeze its summary."""
        ws = self.workspace
        if not ws.claim_job(job_id):  # a cancel may have got there first
            return ws.get_job(job_id).state
        manifest: ResolvedManifest | None = None
        store: JobStore | None = None
        try:
            manifest = load_manifest(ws, job_id)
            store = ws.open_job(job_id)
            touch_progress(ws, job_id)
            return await _JobRun(self, job_id, manifest, store).run()
        except LeaseLostError:
            raise  # the new holder requeues the Job and carries on
        except asyncio.CancelledError:
            self._wind_up(job_id, manifest, store, "cancelled", cancel_reason="user")
            raise
        except Exception as exc:
            self._wind_up(job_id, manifest, store, "failed", error=type(exc).__name__)
            raise
        finally:
            if store is not None:
                store.close()

    def _wind_up(
        self,
        job_id: str,
        manifest: ResolvedManifest | None,
        store: JobStore | None,
        state: JobState,
        *,
        cancel_reason: CancelReason | None = None,
        error: str | None = None,
    ) -> None:
        """End a Job that stopped abnormally, still giving every Document its Record."""
        ws = self.workspace
        try:
            if manifest is not None and store is not None:
                sweep_unfinished(ws, job_id, store, manifest)
            finalize_job(ws, job_id, state, cancel_reason=cancel_reason, error=error)
        except Exception:
            ws.set_job_state(job_id, state)  # at least leave the catalog truthful


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
        self.prices = workspace_prices(self.ws)
        self.stop = asyncio.Event()
        self.classifier: Classifier = _StopGate(runner.classifier, self.stop)
        self.cancel_reason: CancelReason | None = None
        self.lease_lost = False
        self.last_beat = runner.clock()
        self.last_beat_ok = self.last_beat
        self.spent = 0.0
        self.cancelled_docs = 0  # Documents the stop cut short while they were being judged
        self.pool: WorkerPool | None = None
        # Documents allowed between "extraction began" and "classification began".
        self.room = asyncio.Semaphore(max(1, runner.backpressure))
        self.queue: asyncio.Queue[str | None] = asyncio.Queue()
        self.model = runner.classifier.capabilities.model

    def trigger_cancel(self, reason: CancelReason) -> None:
        if self.cancel_reason is None:
            self.cancel_reason = reason
        self.stop.set()
        if self.pool is not None:
            self.pool.cancel(reason)

    def check_cap(self) -> None:
        """Re-read `max_cost_usd` each time: it may change while the Job runs."""
        cap = load_manifest(self.ws, self.job_id).max_cost_usd
        if cap is not None and self.spent >= cap:
            self.trigger_cancel("spend_cap")

    def beat(self) -> None:
        """Heartbeat the lease if one is due. Called from the monitor and from long synchronous
        stretches (finalising), so a big Job can't go stale. Raises `LeaseLostError`."""
        runner = self.runner
        if not runner.holds_lease:
            return
        now = runner.clock()
        if now - self.last_beat < runner.lease_ttl / 3:
            return
        self.last_beat = now
        if not self.ws.heartbeat(runner.holder):
            # Never take the lease back mid-Job: whoever has it recovers this Job.
            runner.holds_lease = False
            raise LeaseLostError("the runner lease was lost")
        self.last_beat_ok = now

    async def run(self) -> JobState:
        runner = self.runner
        for doc_id, state in self.store.states().items():
            if state == "extracting":  # a runner died mid-extraction
                self.store.set_state(doc_id, "queued")
        todo = deque(
            d for d, s in self.store.states().items() if s in ("queued", "extracted", "classifying")
        )
        self.spent = job_spend(self.ws, self.job_id)  # earlier attempts count too
        self.check_cap()
        if todo and runner.workers is not None and not self.stop.is_set():
            self.pool = runner.workers.pool(self.manifest.extraction_profile)
        pool_size = self.pool.size if self.pool else 1  # in-process extraction runs one at a time
        extractors = [asyncio.create_task(self.extract_loop(todo)) for _ in range(pool_size)]
        consumers = [
            asyncio.create_task(self.classify_loop()) for _ in range(runner.classify_concurrency)
        ]

        async def feed_end() -> None:
            await asyncio.gather(*extractors)
            for _ in consumers:
                self.queue.put_nowait(None)

        feeder = asyncio.create_task(feed_end())
        consumed: asyncio.Future[list[None]] = asyncio.gather(*consumers)
        monitor = asyncio.create_task(self.monitor())
        stopper: asyncio.Future[Any] = asyncio.ensure_future(self.stop.wait())
        try:
            while not consumed.done() and not self.stop.is_set():
                waiting: set[asyncio.Future[Any]] = {
                    f for f in (consumed, feeder, stopper, monitor) if not f.done()
                }
                await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
                if monitor.done() and not self.stop.is_set():
                    # The monitor owns the heartbeat and the requests: without it the Job must
                    # not run on, or another runner will recover it while we still write.
                    logger.error("Job %s: the monitor died; stopping", self.job_id)
                    self.lease_lost = True
                    self.stop.set()
                if feeder.done() and feeder.exception() is not None:
                    break
            if self.stop.is_set():
                # Stop extracting, but let Classifier calls already in flight finish (they are
                # billed either way), up to the grace period.
                for producer in (*extractors, feeder):
                    producer.cancel()
                await asyncio.gather(*extractors, feeder, return_exceptions=True)
                for _ in consumers:
                    self.queue.put_nowait(None)
                # A lost lease gets no grace: whoever holds it now owns the Job and bills for it.
                grace = 0.0 if self.lease_lost else runner.cancel_grace
                await asyncio.wait({consumed}, timeout=grace)
        finally:
            everything: list[asyncio.Future[Any]] = [
                *extractors,
                *consumers,
                feeder,
                monitor,
                stopper,
            ]
            for future in everything:
                future.cancel()
            await asyncio.gather(*everything, return_exceptions=True)
            if self.pool is not None:
                if self.pool.cancel_reason == "crash_rate" and self.cancel_reason is None:
                    self.cancel_reason = "crash_rate"  # reached on the last Document
                self.pool.close()
        for outcome in (feeder, consumed):  # a bug in a stage fails the Job; our own cancels don't
            if outcome.done() and not outcome.cancelled():
                error = outcome.exception()
                if isinstance(error, Exception):
                    raise error
        if self.lease_lost:
            raise LeaseLostError("the runner lease was lost")
        return self.finish()

    def finish(self) -> JobState:
        reason = self.cancel_reason
        swept = 0
        if reason is not None:
            swept = sweep_unfinished(
                self.ws, self.job_id, self.store, self.manifest, tick=self.beat
            )
        if reason is not None and (swept or self.cancelled_docs or reason in AUTO_CANCEL_REASONS):
            finalize_job(
                self.ws,
                self.job_id,
                "cancelled",
                cancel_reason=reason,
                live=self.live(),
                tick=self.beat,
            )
            return "cancelled"
        finalize_job(self.ws, self.job_id, "done", live=self.live(), tick=self.beat)
        return "done"

    def live(self) -> dict[str, Any]:
        """What only the running runner knows, frozen into the summary."""
        limiter = self.runner.classifier
        return {"stalled": limiter.stalled, "classifier": limiter.rate_summary()}

    async def monitor(self) -> None:
        """Heartbeat the lease, apply posted requests, and watch the pool's crash-rate cancel.

        A transient SQLite error is retried on the next tick (until the lease would have gone
        stale anyway); a request that cannot be applied is dropped by `apply_requests`.
        """
        runner = self.runner
        while True:
            await runner.sleep(runner.poll_interval)
            try:
                self.beat()
                if runner.holds_lease:
                    apply_requests(self.ws, running=self.job_id, on_cancel=self.trigger_cancel)
                if self.pool is not None and self.pool.cancel_reason == "crash_rate":
                    self.trigger_cancel("crash_rate")
            except LeaseLostError:
                self.lease_lost = True
                self.stop.set()
                return
            except sqlite3.OperationalError as exc:
                logger.warning("Job %s: monitor will retry after: %s", self.job_id, exc)
                if runner.clock() - self.last_beat_ok > runner.lease_ttl:
                    runner.holds_lease = False  # it would have gone stale: treat it as lost
                    self.lease_lost = True
                    self.stop.set()
                    return

    # Extraction

    async def extract_loop(self, todo: deque[str]) -> None:
        while todo and not self.stop.is_set():
            await self.room.acquire()  # backpressure: stop extracting while the Classifier lags
            doc_id = ""
            queued = False
            try:
                if self.stop.is_set() or not todo:
                    return
                doc_id = todo.popleft()
                queued = await self.extract_one(doc_id)
            finally:
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
        if prior is not None and has_extraction(prior):
            prior.attempts = attempts
            # The Record carries this attempt's usage, so it matches its own Judgments. Earlier
            # attempts' spend stays in the Job's ledger.
            prior.usage = Usage(ocr_pages=prior.usage.ocr_pages)
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
                record = await extract_stage(record, task, path, self.pool, self.runner.extractor)
        if record.status == "failed":
            if record.errors and record.errors[-1].code == "cancelled":
                if self.pool is not None and self.pool.cancel_reason == "crash_rate":
                    self.trigger_cancel("crash_rate")
                return False  # the pool was cancelled; the sweep records this Document
            store.save_record(doc_id, "failed", record)
            return False
        store.save_record(doc_id, "extracted", record)  # persisted at once, before classifying
        return True

    # Classification

    async def classify_loop(self) -> None:
        while (doc_id := await self.queue.get()) is not None:
            self.room.release()
            if self.stop.is_set():
                continue  # cancelled: the Document stays `extracted`, and the sweep records it
            await self.classify_one(doc_id)

    async def classify_one(self, doc_id: str) -> None:
        store = self.store
        record = store.get_record(doc_id)
        if record is None:
            return
        store.set_state(doc_id, "classifying")
        try:
            result: ClassifyResult = await classify_stage(
                record, self.manifest, self.specs, self.classifier, self.cache
            )
        except asyncio.CancelledError:
            # Cut off at the end of the grace period: keep what is known, and say what is not.
            record.errors.append(
                RecordError(
                    stage="classify",
                    code="cancelled_in_flight",
                    message="the Classifier call was cut off when the Job was cancelled; "
                    "its cost is not known",
                )
            )
            record.status = "cancelled"
            store.save_record(doc_id, "cancelled", record)
            self.cancelled_docs += 1
            raise
        record = await self.after_first_pass(result.record)
        if any(e.code == "JobStopping" for e in record.errors):
            # The Job stopped between this Document's request groups: keep the groups that were
            # judged (and billed), and mark the Document cancelled.
            record.errors = [e for e in record.errors if e.code != "JobStopping"]
            record.errors.append(
                RecordError(
                    stage="classify",
                    code="cancelled",
                    message="the Job was cancelled before every request group was judged",
                )
            )
            record.status = "cancelled"
            self.cancelled_docs += 1
        state: DocumentState = (
            "failed"
            if record.status == "failed"
            else "cancelled"
            if record.status == "cancelled"
            else "done"
        )
        cost = record_cost(record, self.model, self.prices)  # this attempt's usage only
        add_spend(self.ws, self.job_id, doc_id, cost)  # before the Record: a crash can't lose it
        store.save_record(doc_id, state, record)
        self.spent += cost
        if record.status in ("ok", "partial"):
            touch_progress(self.ws, self.job_id)
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
    ourselves. Abandoning the iterator asks whoever runs a still-unfinished Job to cancel it.
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
        if ws.get_job(job_id).state not in FINISHED_STATES:  # the consumer stopped early
            post_request(ws, "cancel", job_id, "user")
        if driver is not None:
            await asyncio.gather(driver, return_exceptions=True)


async def run(
    spec: JobSpec,
    classifier: Classifier | None = None,
    workers: WorkerSettings | None = None,
    *,
    workspace: Workspace | None = None,
    poll_interval: float = 0.1,
    cancel_grace: float = CANCEL_GRACE,
) -> AsyncIterator[DocumentRecord]:
    """Run a Job and yield one Document Record per input, in input order (docs/spec/v0.md §7.1).

    The Job lives in a Workspace (`workspace`, else `--workspace`/`$ACCELEREAD_HOME`/
    `~/.acceleread`). It waits its turn in the Workspace's FIFO, and when another runner (`serve`)
    holds the lease its Records are streamed from the shared queue. Documents the Job could not
    process because it was cancelled come out with status `cancelled`. Abandoning the iterator
    cancels the Job.
    """
    owned = workspace is None
    ws = workspace if workspace is not None else Workspace.open()
    try:
        runner = Runner(
            ws,
            classifier or JevClassifier(model=spec.model),
            workers=workers,
            poll_interval=poll_interval,
            cancel_grace=cancel_grace,
        )
        job_id = runner.submit(spec)
        async for record in stream_job(ws, runner, job_id, poll_interval=poll_interval):
            yield record
    finally:
        if owned:
            ws.close()
