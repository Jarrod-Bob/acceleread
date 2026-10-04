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
- a handler exception fails the task at once (it would fail the same way again);
- crash failures reaching 5% of finished tasks cancel the pool with reason `crash_rate`.

Every task yields exactly one `Outcome`, even when the pool is cancelled.
"""

import contextlib
import importlib
import multiprocessing
import os
import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from dataclasses import dataclass, field
from multiprocessing.connection import Connection, wait
from pathlib import Path
from typing import Any, Literal

import psutil

Profile = Literal["fast", "quality"]

GB = 1024**3
RAM_PER_WORKER: Mapping[str, int] = {"fast": 2 * GB, "quality": 6 * GB}
CRASH_CODES = frozenset({"worker_crash", "memory_limit"})
CRASH_RATE_CANCEL = 0.05
MAX_ATTEMPTS = 2
POLL_SECONDS = 0.05

Report = Callable[[int], None]
Handler = Callable[[Any, Report], Any]


@dataclass(frozen=True)
class TimeoutPolicy:
    """`base_s + per_page_s * Pages needing OCR`, capped at `cap_s`."""

    base_s: float
    per_page_s: float
    cap_s: float = 1800

    def seconds(self, ocr_pages: int) -> float:
        return min(self.cap_s, self.base_s + self.per_page_s * ocr_pages)


TIMEOUTS: Mapping[str, TimeoutPolicy] = {
    "fast": TimeoutPolicy(base_s=60, per_page_s=20),
    "quality": TimeoutPolicy(base_s=60, per_page_s=6),
}


def document_timeout(profile: Profile, ocr_pages: int) -> float:
    return TIMEOUTS[profile].seconds(ocr_pages)


def worker_count(
    profile: Profile,
    physical_cores: int | None = None,
    available_ram: int | None = None,
    ocr_workers: int | None = None,
) -> int:
    """`fast`: cores - 1. `quality`: cores // 4. Both capped by RAM / the Profile's estimate.

    `ocr_workers` (the `--ocr-workers` flag) overrides the whole calculation.
    """
    if ocr_workers is not None:
        return max(1, ocr_workers)
    cores = physical_cores or psutil.cpu_count(logical=False) or os.cpu_count() or 1
    ram = available_ram if available_ram is not None else psutil.virtual_memory().total
    by_cpu = cores - 1 if profile == "fast" else cores // 4
    return max(1, min(by_cpu, ram // RAM_PER_WORKER[profile]))


@dataclass(frozen=True)
class Outcome:
    ok: bool
    result: Any = None
    code: str | None = None
    message: str | None = None
    attempts: int = 1


@dataclass(frozen=True)
class ExtractTask:
    """One Document for the default extraction handler."""

    path: Path
    format: Literal["pdf", "html"]
    ocr_languages: tuple[str, ...] = ("en",)

    def for_pool(self) -> tuple[str, "ExtractTask"]:
        return "acceleread.workers:extract_document", self


def extract_document(task: ExtractTask, report: Report) -> Any:
    """Default handler, run inside a worker: Extract one Document."""
    from acceleread.extract import extract_html, extract_pdf

    if task.format == "html":
        return extract_html(task.path)
    return extract_pdf(task.path, ocr_languages=task.ocr_languages, on_ocr_pages=report)


def _load_handler(spec: str) -> Handler:
    module, _, name = spec.partition(":")
    handler: Handler = getattr(importlib.import_module(module), name)
    return handler


def _worker_main(conn: Connection, env: dict[str, str]) -> None:
    os.environ.update(env)  # before any engine is imported
    while True:
        try:
            message = conn.recv()
        except EOFError:
            return
        if message is None:
            return
        spec, payload = message

        def report(ocr_pages: int) -> None:
            conn.send(("progress", ocr_pages))

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


@dataclass
class _Worker:
    process: Any
    conn: Connection
    done: int = 0
    task: _Task | None = None
    started: float = 0.0
    deadline: float = 0.0


@dataclass
class _Stats:
    finished: int = 0
    crash_failures: int = 0
    cancel_reason: str | None = None
    pids: list[int] = field(default_factory=list)


class WorkerPool:
    DEFAULT_RECYCLE_AFTER = 200

    def __init__(
        self,
        workers: int,
        timeout: TimeoutPolicy = TIMEOUTS["fast"],
        memory_limit_bytes: int = RAM_PER_WORKER["fast"],
        recycle_after: int = DEFAULT_RECYCLE_AFTER,
        threads_per_worker: int = 1,
    ) -> None:
        self._size = max(1, workers)
        self._timeout = timeout
        self._memory_limit = memory_limit_bytes
        self._recycle_after = recycle_after
        self._env = {
            "OMP_THREAD_LIMIT": str(threads_per_worker),
            "OMP_NUM_THREADS": str(threads_per_worker),
        }
        self._context = multiprocessing.get_context("spawn")
        self._commands: queue.SimpleQueue[tuple[str, Any]] = queue.SimpleQueue()
        self._wake_r, self._wake_w = self._context.Pipe(duplex=False)
        self._pending: deque[_Task] = deque()
        self._workers: list[_Worker] = []
        self._stats = _Stats()
        self._closing = False
        self._thread = threading.Thread(target=self._loop, name="acceleread-workers", daemon=True)
        self._thread.start()

    @property
    def cancel_reason(self) -> str | None:
        return self._stats.cancel_reason

    def submit(self, handler: str, payload: Any) -> "Future[Outcome]":
        """Queue one Document. The future always resolves to an Outcome, never raises."""
        future: Future[Outcome] = Future()
        self._send("submit", _Task(handler, payload, future))
        return future

    def cancel(self, reason: str) -> None:
        """Cancel queued and in-flight tasks. The first reason given is the one kept."""
        self._send("cancel", reason)

    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._wake()
        self._thread.join(timeout=30)

    def __enter__(self) -> "WorkerPool":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _send(self, kind: str, value: Any) -> None:
        self._commands.put((kind, value))
        self._wake()

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
            elif self._stats.cancel_reason is not None:
                value.future.set_result(Outcome(False, code="cancelled", attempts=0))
            else:
                self._pending.append(value)

    def _dispatch(self) -> None:
        while self._pending:
            worker = next((w for w in self._workers if w.task is None), None)
            if worker is None:
                if len(self._workers) >= self._size:
                    return
                worker = self._spawn()
            task = self._pending.popleft()
            task.attempts += 1
            worker.task = task
            worker.started = time.monotonic()
            worker.deadline = worker.started + self._timeout.seconds(0)
            try:
                worker.conn.send((task.handler, task.payload))
            except (OSError, ValueError):
                self._fail_attempt(worker, "worker_crash", "worker pipe closed")
            except Exception as error:  # payload could not be pickled
                worker.task = None
                task.future.set_result(
                    Outcome(False, code="handler_error", message=str(error), attempts=1)
                )

    def _spawn(self) -> _Worker:
        parent, child = self._context.Pipe()
        process = self._context.Process(
            target=_worker_main, args=(child, self._env), name="acceleread-extract", daemon=True
        )
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
            if kind == "progress":
                worker.deadline = worker.started + self._timeout.seconds(int(value))
            elif worker.task is not None:
                self._complete(worker, kind, value)
                if worker not in self._workers:
                    return
        if worker.task is None:
            if not worker.process.is_alive():
                self._remove(worker)
            return
        if not worker.process.is_alive():
            self._fail_attempt(worker, "worker_crash", "worker process died")
        elif time.monotonic() > worker.deadline:
            self._fail_attempt(worker, "timeout", "Document timed out")
        elif self._rss(worker) > self._memory_limit:
            self._fail_attempt(worker, "memory_limit", "worker exceeded its memory cap")

    def _complete(self, worker: _Worker, kind: str, value: Any) -> None:
        task = worker.task
        assert task is not None
        worker.task = None
        worker.done += 1
        if kind == "done":
            outcome = Outcome(True, result=value, attempts=task.attempts)
        else:
            outcome = Outcome(False, code="handler_error", message=value, attempts=task.attempts)
        self._finish(task, outcome)
        if worker.done >= self._recycle_after:
            self._retire(worker)

    def _fail_attempt(self, worker: _Worker, code: str, message: str) -> None:
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
        task.future.set_result(outcome)
        if (
            stats.cancel_reason is None
            and stats.crash_failures / stats.finished >= CRASH_RATE_CANCEL
        ):
            self._cancel("crash_rate")

    def _cancel(self, reason: str) -> None:
        if self._stats.cancel_reason is None:
            self._stats.cancel_reason = reason
        while self._pending:
            self._pending.popleft().future.set_result(Outcome(False, code="cancelled", attempts=0))
        for worker in list(self._workers):
            if worker.task is not None:
                task, worker.task = worker.task, None
                self._kill(worker)
                task.future.set_result(Outcome(False, code="cancelled", attempts=task.attempts))

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
        if worker.process.is_alive():
            worker.process.kill()
        worker.process.join(timeout=5)
        self._remove(worker)

    def _remove(self, worker: _Worker) -> None:
        if worker in self._workers:
            self._workers.remove(worker)
        worker.conn.close()
        worker.process.join(timeout=0)

    def _shutdown(self) -> None:
        self._cancel(self._stats.cancel_reason or "closed")
        for worker in list(self._workers):
            if worker.process.is_alive():
                self._retire(worker)
            else:
                self._remove(worker)
        self._wake_r.close()
        self._wake_w.close()
