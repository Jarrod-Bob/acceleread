# SPDX-License-Identifier: Apache-2.0
"""The extraction worker manager (docs/spec/v0.md §7.4, ADR 0002 amended).

Our own small manager over `multiprocessing` (spawn), not `ProcessPoolExecutor`, so one hung or
oversized worker can be killed without tearing down the pool. Each worker has its own task pipe;
one task is one Document. Workers are generic: a task names a handler by dotted path
(`module:function`) and the worker imports it on first use, so an engine such as Tesseract is only
loaded in workers that need it.

Failure handling:
- a crash or a memory kill is retried once in a fresh worker, then fails;
- a timeout fails at once and is never retried;
- a handler exception, or a result the parent cannot unpickle, fails the task at once;
- crash failures reaching 5% of finished tasks cancel the pool with reason `crash_rate`.

Every task yields exactly one `Outcome`, even when the pool is cancelled or closed. The timeout
clock starts when the worker reports it has begun the task, so spawn and import time don't count.
"""

import contextlib
import importlib
import multiprocessing
import os
import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from pathlib import Path
from typing import Any, Literal

import psutil

from acceleread.extract import PageCountsCallback

Profile = Literal["fast", "quality"]
OutcomeCode = Literal["handler_error", "worker_crash", "memory_limit", "timeout", "cancelled"]
WorkerMessage = Literal["started", "progress", "done", "error"]
Command = Literal["submit", "cancel"]

GB = 1024**3
MB = 1024**2
# Estimates of what one worker needs, used only to size the pool from available RAM
# (about 150 MB for fast, 1.5-2.5 GB for quality).
RAM_ESTIMATE: Mapping[str, int] = {"fast": 150 * MB, "quality": 2 * GB}
# The hard per-worker cap, watched with psutil while a task runs.
MEMORY_CAP: Mapping[str, int] = {"fast": 2 * GB, "quality": 6 * GB}
THREADS: Mapping[str, int] = {"fast": 1, "quality": 4}
CRASH_CODES: frozenset[str] = frozenset({"worker_crash", "memory_limit"})
CRASH_RATE_CANCEL = 0.05
MAX_ATTEMPTS = 2
POLL_SECONDS = 0.05
START_LIMIT_SECONDS = 120.0  # a worker that never begins its task is treated as crashed

Handler = Callable[[Any, PageCountsCallback], Any]
_SPAWN_ENV = threading.Lock()


@dataclass(frozen=True)
class TimeoutPolicy:
    """`base_s + per_page_s * N`, capped at `cap_s`. N is the OCR Pages or all Pages."""

    base_s: float
    per_page_s: float
    cap_s: float = 1800
    per: Literal["ocr_pages", "pages"] = "ocr_pages"

    def seconds(self, pages: int, ocr_pages: int) -> float:
        count = ocr_pages if self.per == "ocr_pages" else pages
        return min(self.cap_s, self.base_s + self.per_page_s * count)


TIMEOUTS: Mapping[str, TimeoutPolicy] = {
    "fast": TimeoutPolicy(base_s=60, per_page_s=20, per="ocr_pages"),
    "quality": TimeoutPolicy(base_s=60, per_page_s=6, per="pages"),
}


def document_timeout(profile: Profile, pages: int, ocr_pages: int) -> float:
    return TIMEOUTS[profile].seconds(pages, ocr_pages)


def worker_count(
    profile: Profile,
    physical_cores: int | None = None,
    available_ram: int | None = None,
    ocr_workers: int | None = None,
) -> int:
    """`fast`: cores - 1. `quality`: cores // 4. Both capped by available RAM / an estimate.

    `ocr_workers` (the `--ocr-workers` flag) overrides the whole calculation.
    """
    if ocr_workers is not None:
        return max(1, ocr_workers)
    cores = physical_cores or psutil.cpu_count(logical=False) or os.cpu_count() or 1
    ram = available_ram if available_ram is not None else psutil.virtual_memory().available
    by_cpu = cores - 1 if profile == "fast" else cores // 4
    return max(1, min(by_cpu, ram // RAM_ESTIMATE[profile]))


@dataclass(frozen=True)
class Outcome:
    ok: bool
    result: Any = None
    code: OutcomeCode | None = None
    message: str | None = None
    attempts: int = 1


@dataclass(frozen=True)
class ExtractTask:
    """One Document for the default extraction handler."""

    path: Path
    format: Literal["pdf", "html"]
    ocr_languages: tuple[str, ...] = ("en",)
    tessdata: tuple[Path, ...] = ()
    """Extra Tesseract pack directories (the Workspace's), searched after the vendored one."""

    def for_pool(self) -> tuple[str, "ExtractTask"]:
        return "acceleread.workers:extract_document", self


def extract_document(task: ExtractTask, report: PageCountsCallback) -> Any:
    """Default handler, run inside a worker: Extract one Document."""
    from acceleread.extract import VENDORED_TESSDATA, extract_html, extract_pdf

    if task.format == "html":
        return extract_html(task.path)
    return extract_pdf(
        task.path,
        ocr_languages=task.ocr_languages,
        tessdata=[VENDORED_TESSDATA, *task.tessdata],
        on_page_counts=report,
    )


def _load_handler(spec: str) -> Handler:
    module, _, name = spec.partition(":")
    handler: Handler = getattr(importlib.import_module(module), name)
    return handler


def _worker_main(conn: Connection) -> None:
    # The parent set OMP_THREAD_LIMIT and friends in the spawn environment, so they are in place
    # before anything here is imported.
    while True:
        try:
            message = conn.recv()
        except EOFError:
            return
        if message is None:
            return
        spec, payload = message
        conn.send(("started", None))

        def report(pages: int, ocr_pages: int) -> None:
            conn.send(("progress", (pages, ocr_pages)))

        try:
            result = _load_handler(spec)(payload, report)
            conn.send(("done", result))
        except Exception as error:
            conn.send(("error", f"{type(error).__name__}: {error}"))


@dataclass
class _Task:
    handler: str
    payload: Any
    future: "Future[Outcome]"
    attempts: int = 0
    running: bool = False


@dataclass
class _Worker:
    process: Any
    conn: Connection
    done: int = 0
    task: _Task | None = None
    dispatched_at: float = 0.0
    started_at: float | None = None
    deadline: float | None = None


@dataclass
class _Stats:
    finished: int = 0
    crash_failures: int = 0
    cancel_reason: str | None = None


def _resolve(future: "Future[Outcome]", outcome: Outcome) -> None:
    """Set a task's result, tolerating a caller that has cancelled its own future."""
    with contextlib.suppress(InvalidStateError):
        future.set_result(outcome)


def _cancelled(attempts: int = 0) -> Outcome:
    return Outcome(False, code="cancelled", attempts=attempts)


@contextlib.contextmanager
def _spawn_environment(env: Mapping[str, str]) -> Iterator[None]:
    """Put `env` in os.environ while a child spawns, so it is set before the child imports."""
    with _SPAWN_ENV:
        saved = {key: os.environ.get(key) for key in env}
        os.environ.update(env)
        try:
            yield
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


class WorkerPool:
    DEFAULT_RECYCLE_AFTER = 200

    def __init__(
        self,
        workers: int,
        timeout: TimeoutPolicy = TIMEOUTS["fast"],
        memory_limit_bytes: int = MEMORY_CAP["fast"],
        recycle_after: int = DEFAULT_RECYCLE_AFTER,
        threads_per_worker: int = 1,
    ) -> None:
        self.size = max(1, workers)
        self.threads_per_worker = threads_per_worker
        self._timeout = timeout
        self._memory_limit = memory_limit_bytes
        self._recycle_after = recycle_after
        self._env = {
            "OMP_THREAD_LIMIT": str(threads_per_worker),
            "OMP_NUM_THREADS": str(threads_per_worker),
        }
        self._context = multiprocessing.get_context("spawn")
        self._commands: queue.SimpleQueue[tuple[Command, Any]] = queue.SimpleQueue()
        self._wake_r, self._wake_w = self._context.Pipe(duplex=False)
        self._pending: deque[_Task] = deque()
        self._workers: list[_Worker] = []
        self._stats = _Stats()
        self._lock = threading.Lock()
        self._closed = False
        self._closing = False
        self._thread = threading.Thread(target=self._loop, name="acceleread-workers", daemon=True)
        self._thread.start()

    @property
    def cancel_reason(self) -> str | None:
        return self._stats.cancel_reason

    def submit(self, handler: str, payload: Any) -> "Future[Outcome]":
        """Queue one Document. The future always resolves to an Outcome, never raises."""
        future: Future[Outcome] = Future()
        with self._lock:
            if self._closed:
                _resolve(future, _cancelled())
                return future
            self._commands.put(("submit", _Task(handler, payload, future)))
        self._wake()
        return future

    def cancel(self, reason: str) -> None:
        """Cancel queued and in-flight tasks. The first reason given is the one kept."""
        self._commands.put(("cancel", reason))
        self._wake()

    def close(self) -> None:
        self._closing = True
        self._wake()
        self._thread.join(timeout=30)

    def __enter__(self) -> "WorkerPool":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _wake(self) -> None:
        with contextlib.suppress(OSError, ValueError):
            self._wake_w.send(None)

    # Everything below runs on the manager thread only.

    def _loop(self) -> None:
        try:
            while not self._closing:
                self._drain_commands()
                self._dispatch()
                waitables: list[Any] = [self._wake_r]
                for worker in self._workers:
                    waitables += [worker.conn, worker.process.sentinel]
                for ready in wait(waitables, timeout=POLL_SECONDS):
                    if ready is self._wake_r:
                        while self._wake_r.poll():
                            self._wake_r.recv()
                for worker in list(self._workers):
                    self._service(worker)
        finally:
            self._shutdown()

    def _drain_commands(self) -> None:
        while True:
            try:
                kind, value = self._commands.get_nowait()
            except queue.Empty:
                return
            if kind == "cancel":
                self._cancel(value)
            elif kind == "submit":
                if self._stats.cancel_reason is not None:
                    _resolve(value.future, _cancelled())
                else:
                    self._pending.append(value)
            else:
                raise ValueError(f"unknown command {kind!r}")

    def _dispatch(self) -> None:
        while self._pending:
            task = self._pending.popleft()
            if task.future.cancelled():
                continue  # the caller cancelled it while it was queued
            worker = next((w for w in self._workers if w.task is None), None)
            if worker is None and len(self._workers) < self.size:
                worker = self._spawn()
            if worker is None:
                self._pending.appendleft(task)  # stays cancellable until a worker takes it
                return
            if not task.running:
                if not task.future.set_running_or_notify_cancel():
                    continue
                task.running = True
            task.attempts += 1
            worker.task = task
            worker.dispatched_at = time.monotonic()
            worker.started_at = None
            worker.deadline = None
            try:
                worker.conn.send((task.handler, task.payload))
            except (OSError, ValueError):
                self._fail_attempt(worker, "worker_crash", "worker pipe closed")
            except Exception as error:  # the payload could not be pickled
                worker.task = None
                message = f"{type(error).__name__}: {error}"
                self._finish(task, Outcome(False, code="handler_error", message=message))

    def _spawn(self) -> _Worker:
        parent, child = self._context.Pipe()
        process = self._context.Process(
            target=_worker_main, args=(child,), name="acceleread-extract", daemon=True
        )
        with _spawn_environment(self._env):
            process.start()
        child.close()
        worker = _Worker(process=process, conn=parent)
        self._workers.append(worker)
        return worker

    def _service(self, worker: _Worker) -> None:
        while worker.conn.poll():
            try:
                kind, value = worker.conn.recv()
            except (EOFError, OSError):
                break
            except Exception as error:  # e.g. a result that cannot be unpickled here
                kind = "error"
                value = f"undeliverable result: {type(error).__name__}: {error}"
            self._handle(worker, kind, value)
            if worker not in self._workers:
                return
        task = worker.task
        if task is None:
            if not worker.process.is_alive():
                self._remove(worker)
            return
        now = time.monotonic()
        if not worker.process.is_alive():
            self._fail_attempt(worker, "worker_crash", "worker process died")
        elif worker.started_at is None and now - worker.dispatched_at > START_LIMIT_SECONDS:
            self._fail_attempt(worker, "worker_crash", "worker never started the task")
        elif worker.deadline is not None and now > worker.deadline:
            self._fail_attempt(worker, "timeout", "Document timed out")
        elif self._rss(worker) > self._memory_limit:
            self._fail_attempt(worker, "memory_limit", "worker exceeded its memory cap")

    def _handle(self, worker: _Worker, kind: WorkerMessage, value: Any) -> None:
        task = worker.task
        if task is None:
            return
        if kind == "started":
            worker.started_at = time.monotonic()
            worker.deadline = worker.started_at + self._timeout.seconds(0, 0)
        elif kind == "progress":
            started = worker.started_at if worker.started_at is not None else time.monotonic()
            pages, ocr_pages = value
            worker.deadline = started + self._timeout.seconds(int(pages), int(ocr_pages))
        elif kind in ("done", "error"):
            worker.task = None
            worker.done += 1
            if kind == "done":
                outcome = Outcome(True, result=value, attempts=task.attempts)
            else:
                outcome = Outcome(
                    False, code="handler_error", message=value, attempts=task.attempts
                )
            self._finish(task, outcome)
            if worker.done >= self._recycle_after:
                self._retire(worker)
        else:
            raise ValueError(f"unknown worker message {kind!r}")

    def _fail_attempt(self, worker: _Worker, code: OutcomeCode, message: str) -> None:
        task = worker.task
        assert task is not None
        worker.task = None
        self._kill(worker)
        if code in CRASH_CODES and task.attempts < MAX_ATTEMPTS:
            self._pending.appendleft(task)
            return
        self._finish(task, Outcome(False, code=code, message=message, attempts=task.attempts))

    def _finish(self, task: _Task, outcome: Outcome) -> None:
        stats = self._stats
        stats.finished += 1
        if outcome.code in CRASH_CODES:
            stats.crash_failures += 1
        _resolve(task.future, outcome)
        if (
            stats.cancel_reason is None
            and stats.crash_failures / stats.finished >= CRASH_RATE_CANCEL
        ):
            self._cancel("crash_rate")

    def _cancel(self, reason: str) -> None:
        if self._stats.cancel_reason is None:
            self._stats.cancel_reason = reason
        while self._pending:
            _resolve(self._pending.popleft().future, _cancelled())
        for worker in list(self._workers):
            if worker.task is not None:
                task, worker.task = worker.task, None
                self._kill(worker)
                _resolve(task.future, _cancelled(task.attempts))

    @staticmethod
    def _rss(worker: _Worker) -> int:
        try:
            process = psutil.Process(worker.process.pid)
            return sum(p.memory_info().rss for p in [process, *process.children(recursive=True)])
        except psutil.Error:
            return 0

    def _retire(self, worker: _Worker) -> None:
        """Recycle: ask the worker to exit after its quota; a fresh one spawns on demand."""
        with contextlib.suppress(OSError, ValueError):
            worker.conn.send(None)
        worker.process.join(timeout=5)
        self._remove(worker)

    def _kill(self, worker: _Worker) -> None:
        """Kill the worker and everything it started (the tree `_rss` measures)."""
        try:
            children = psutil.Process(worker.process.pid).children(recursive=True)
        except psutil.Error:
            children = []
        if worker.process.is_alive():
            worker.process.kill()
        for child in children:
            with contextlib.suppress(psutil.Error):
                child.kill()
        worker.process.join(timeout=5)
        self._remove(worker)

    def _remove(self, worker: _Worker) -> None:
        if worker in self._workers:
            self._workers.remove(worker)
        worker.conn.close()
        worker.process.join(timeout=0)

    def _shutdown(self) -> None:
        with self._lock:
            self._closed = True
        self._drain_commands()
        self._cancel(self._stats.cancel_reason or "closed")
        for worker in list(self._workers):
            if worker.process.is_alive():
                self._retire(worker)
            else:
                self._remove(worker)
        self._wake_r.close()
        self._wake_w.close()


@dataclass(frozen=True)
class WorkerSettings:
    """The `--ocr-workers` and `--threads-per-worker` overrides; None keeps the defaults."""

    ocr_workers: int | None = None
    threads_per_worker: int | None = None

    def pool(self, profile: Profile = "fast") -> WorkerPool:
        return WorkerPool(
            workers=worker_count(profile, ocr_workers=self.ocr_workers),
            timeout=TIMEOUTS[profile],
            memory_limit_bytes=MEMORY_CAP[profile],
            threads_per_worker=self.threads_per_worker or THREADS[profile],
        )
