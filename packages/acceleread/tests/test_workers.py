# SPDX-License-Identifier: Apache-2.0
"""The extraction worker manager (docs/spec/v0.md §7.4): spawn workers, one task per Document,
kill a single worker, recycle, retry once on crash, never retry a timeout, crash-rate cancel."""

import asyncio
import time
from collections.abc import Iterator
from pathlib import Path

import psutil
import pytest

from acceleread.extract import Extracted
from acceleread.workers import (
    ExtractTask,
    Outcome,
    TimeoutPolicy,
    WorkerPool,
    WorkerSettings,
    document_timeout,
    worker_count,
)

TESTS = Path(__file__).parent
FIXTURES = TESTS / "fixtures"
GB = 1024**3
ECHO = "worker_handlers:echo"
GENEROUS = TimeoutPolicy(base_s=30, per_page_s=1, cap_s=60)


@pytest.fixture(autouse=True)
def handlers_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Spawned workers inherit sys.path, so they can import tests/worker_handlers.py."""
    monkeypatch.syspath_prepend(str(TESTS))


@pytest.fixture
def make_pool() -> Iterator[type[WorkerPool]]:
    pools: list[WorkerPool] = []

    class Tracked(WorkerPool):
        def __init__(self, **kwargs: object) -> None:
            kwargs.setdefault("timeout", GENEROUS)
            kwargs.setdefault("workers", 1)
            super().__init__(**kwargs)  # type: ignore[arg-type]
            pools.append(self)

    yield Tracked
    for pool in pools:
        pool.close()


def run(pool: WorkerPool, handler: str, payload: object = None) -> Outcome:
    return pool.submit(handler, payload).result(timeout=60)


def test_a_task_runs_in_a_worker_and_returns_its_result(make_pool: type[WorkerPool]) -> None:
    pool = make_pool()
    outcome = run(pool, ECHO, {"a": [1, 2]})
    assert outcome.ok is True
    assert outcome.result == {"a": [1, 2]}
    assert outcome.attempts == 1


def test_tasks_spread_over_the_configured_number_of_workers(make_pool: type[WorkerPool]) -> None:
    pool = make_pool(workers=2)
    futures = [pool.submit("worker_handlers:pid", 0.3) for _ in range(4)]
    pids = {f.result(timeout=60).result for f in futures}
    assert len(pids) == 2


def test_a_handler_error_fails_the_task_without_retry_or_losing_the_worker(
    make_pool: type[WorkerPool],
) -> None:
    pool = make_pool()
    first = run(pool, "worker_handlers:pid")
    broken = run(pool, "worker_handlers:fail")
    assert broken.ok is False
    assert broken.code == "handler_error"
    assert "this Document is broken" in (broken.message or "")
    assert broken.attempts == 1
    assert run(pool, "worker_handlers:pid").result == first.result


def test_a_worker_is_recycled_after_its_quota_of_documents(make_pool: type[WorkerPool]) -> None:
    pool = make_pool(recycle_after=2)
    pids = [run(pool, "worker_handlers:pid").result for _ in range(5)]
    assert pids[0] == pids[1] != pids[2] == pids[3] != pids[4]


def test_the_default_recycle_quota_is_two_hundred_documents() -> None:
    assert WorkerPool.DEFAULT_RECYCLE_AFTER == 200


def test_workers_run_engines_on_one_thread_by_default(make_pool: type[WorkerPool]) -> None:
    pool = make_pool()
    assert run(pool, "worker_handlers:omp_thread_limit").result == "1"


def test_threads_per_worker_overrides_the_thread_limit(make_pool: type[WorkerPool]) -> None:
    pool = make_pool(threads_per_worker=4)
    assert run(pool, "worker_handlers:omp_thread_limit").result == "4"


def test_a_crash_is_retried_once_in_a_fresh_worker(
    make_pool: type[WorkerPool], tmp_path: Path
) -> None:
    pool = make_pool()
    outcome = run(pool, "worker_handlers:crash_once", str(tmp_path / "marker"))
    assert outcome.ok is True
    assert outcome.attempts == 2


def test_a_second_crash_fails_the_task(make_pool: type[WorkerPool]) -> None:
    pool = make_pool()
    outcome = run(pool, "worker_handlers:crash")
    assert outcome.ok is False
    assert outcome.code == "worker_crash"
    assert outcome.attempts == 2


def test_the_pool_keeps_working_after_a_crash(make_pool: type[WorkerPool]) -> None:
    pool = make_pool()
    for i in range(30):
        run(pool, ECHO, i)
    run(pool, "worker_handlers:crash")
    assert run(pool, ECHO, "still here").result == "still here"


def test_a_timeout_fails_immediately_and_kills_only_that_worker(
    make_pool: type[WorkerPool],
) -> None:
    pool = make_pool(workers=2, timeout=TimeoutPolicy(base_s=3, per_page_s=1, cap_s=10))
    hung = pool.submit("worker_handlers:hang", None)
    survivor = pool.submit("worker_handlers:pid", 0.5)
    timed_out = hung.result(timeout=60)
    assert timed_out.ok is False
    assert timed_out.code == "timeout"
    assert timed_out.attempts == 1
    assert survivor.result(timeout=60).ok is True


def test_announcing_page_counts_extends_the_deadline(make_pool: type[WorkerPool]) -> None:
    pool = make_pool(timeout=TimeoutPolicy(base_s=2, per_page_s=3, cap_s=30, per="ocr_pages"))
    unannounced = run(pool, "worker_handlers:sleep_after_report", (5, 0, 4.0))
    assert unannounced.code == "timeout"
    announced = run(pool, "worker_handlers:sleep_after_report", (5, 2, 4.0))
    assert announced.ok is True


def test_a_quality_deadline_grows_with_all_pages_not_only_ocr_pages(
    make_pool: type[WorkerPool],
) -> None:
    pool = make_pool(timeout=TimeoutPolicy(base_s=2, per_page_s=3, cap_s=30, per="pages"))
    assert run(pool, "worker_handlers:sleep_after_report", (2, 0, 4.0)).ok is True


def test_spawn_time_does_not_count_against_the_deadline(make_pool: type[WorkerPool]) -> None:
    pool = make_pool(timeout=TimeoutPolicy(base_s=0.01, per_page_s=0, cap_s=1))
    assert run(pool, "worker_handlers:echo", "fresh worker").ok is True


def test_a_worker_over_its_memory_cap_is_killed_and_retried_once(
    make_pool: type[WorkerPool],
) -> None:
    probe = make_pool()
    worker_pid = run(probe, "worker_handlers:pid").result
    baseline = psutil.Process(worker_pid).memory_info().rss
    pool = make_pool(memory_limit_bytes=baseline + 100 * 1024 * 1024)
    outcome = run(pool, "worker_handlers:hog", 400)
    assert outcome.ok is False
    assert outcome.code == "memory_limit"
    assert outcome.attempts == 2


def test_crash_failures_reaching_five_percent_cancel_the_rest(
    make_pool: type[WorkerPool],
) -> None:
    pool = make_pool()
    outcomes = [run(pool, ECHO, i) for i in range(19)]
    assert all(o.ok for o in outcomes)
    assert pool.cancel_reason is None
    crashed = run(pool, "worker_handlers:crash")
    assert crashed.code == "worker_crash"
    assert pool.cancel_reason == "crash_rate"
    after = run(pool, ECHO, "late")
    assert after.ok is False
    assert after.code == "cancelled"


def test_queued_tasks_are_cancelled_when_the_crash_rate_trips(
    make_pool: type[WorkerPool],
) -> None:
    pool = make_pool()
    crash = pool.submit("worker_handlers:crash", None)
    queued = [pool.submit(ECHO, i) for i in range(3)]
    assert crash.result(timeout=60).code == "worker_crash"
    assert [f.result(timeout=60).code for f in queued] == ["cancelled"] * 3


def test_a_crash_rate_just_under_five_percent_does_not_cancel(
    make_pool: type[WorkerPool],
) -> None:
    pool = make_pool()
    for i in range(20):
        run(pool, ECHO, i)
    run(pool, "worker_handlers:crash")
    assert pool.cancel_reason is None
    assert run(pool, ECHO, "carry on").ok is True


def test_timeouts_and_handler_errors_do_not_count_as_crashes(
    make_pool: type[WorkerPool],
) -> None:
    pool = make_pool(timeout=TimeoutPolicy(base_s=2, per_page_s=1, cap_s=10))
    assert run(pool, "worker_handlers:hang").code == "timeout"
    assert run(pool, "worker_handlers:fail").code == "handler_error"
    assert pool.cancel_reason is None


def test_cancel_stops_in_flight_and_queued_tasks(make_pool: type[WorkerPool]) -> None:
    pool = make_pool()
    running = pool.submit("worker_handlers:hang", None)
    queued = pool.submit(ECHO, 1)
    pool.cancel("user")
    assert running.result(timeout=60).code == "cancelled"
    assert queued.result(timeout=60).code == "cancelled"
    assert pool.cancel_reason == "user"


def test_a_caller_cancelling_its_future_does_not_stop_other_tasks(
    make_pool: type[WorkerPool],
) -> None:
    pool = make_pool()
    running = pool.submit("worker_handlers:pid", 0.5)
    doomed = pool.submit(ECHO, "doomed")
    survivor = pool.submit(ECHO, "survivor")
    deadline = time.monotonic() + 30
    while not running.running() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert doomed.cancel() is True  # still queued behind the running task
    assert running.cancel() is False  # already running: it must still resolve
    assert survivor.result(timeout=60).result == "survivor"
    assert running.result(timeout=60).ok is True
    assert run(pool, ECHO, "after").result == "after"


def test_close_resolves_queued_tasks_as_cancelled(make_pool: type[WorkerPool]) -> None:
    pool = make_pool()
    running = pool.submit("worker_handlers:hang", None)
    queued = [pool.submit(ECHO, i) for i in range(3)]
    pool.close()
    assert running.result(timeout=60).code == "cancelled"
    assert [f.result(timeout=60).code for f in queued] == ["cancelled"] * 3


def test_submitting_after_close_is_cancelled_at_once(make_pool: type[WorkerPool]) -> None:
    pool = make_pool()
    pool.close()
    assert pool.submit(ECHO, 1).result(timeout=5).code == "cancelled"


def test_a_result_the_parent_cannot_unpickle_fails_that_task_only(
    make_pool: type[WorkerPool],
) -> None:
    pool = make_pool()
    bad = run(pool, "worker_handlers:unloadable")
    assert bad.ok is False
    assert bad.code == "handler_error"
    assert run(pool, ECHO, "next").result == "next"


def test_a_payload_that_cannot_be_pickled_fails_as_a_handler_error(
    make_pool: type[WorkerPool],
) -> None:
    pool = make_pool()
    bad = run(pool, ECHO, lambda: None)
    assert bad.code == "handler_error"
    assert run(pool, ECHO, "next").result == "next"


def test_killing_a_worker_kills_its_children_too(
    make_pool: type[WorkerPool], tmp_path: Path
) -> None:
    pool = make_pool(timeout=TimeoutPolicy(base_s=2, per_page_s=0, cap_s=2))
    pidfile = tmp_path / "child.pid"
    outcome = run(pool, "worker_handlers:spawn_child_and_hang", str(pidfile))
    assert outcome.code == "timeout"
    child = int(pidfile.read_text())
    deadline = time.monotonic() + 10
    while psutil.pid_exists(child) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not psutil.pid_exists(child)


async def test_async_callers_can_await_a_task(make_pool: type[WorkerPool]) -> None:
    pool = make_pool()
    outcome = await asyncio.wrap_future(pool.submit(ECHO, 42))
    assert outcome.result == 42


def test_extract_task_runs_the_default_extraction_handler(make_pool: type[WorkerPool]) -> None:
    pool = make_pool()
    scanned = run(pool, *ExtractTask(FIXTURES / "scanned.pdf", "pdf").for_pool())
    assert scanned.ok is True
    assert isinstance(scanned.result, Extracted)
    assert "polysilicon" in scanned.result.text
    assert scanned.result.pages[0].method == "ocr-full"
    page = run(pool, *ExtractTask(FIXTURES / "sample.pdf", "pdf").for_pool())
    assert isinstance(page.result, Extracted) and page.result.ocr_pages == 0


def test_extract_task_handles_html(make_pool: type[WorkerPool], tmp_path: Path) -> None:
    source = tmp_path / "a.html"
    source.write_text("<html><body><p>Hello world</p></body></html>")
    outcome = run(make_pool(), *ExtractTask(source, "html").for_pool())
    assert isinstance(outcome.result, Extracted)
    assert outcome.result.text == "Hello world"


def test_extract_task_with_an_unreadable_file_fails_as_a_handler_error(
    make_pool: type[WorkerPool], tmp_path: Path
) -> None:
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf")
    outcome = run(make_pool(), *ExtractTask(broken, "pdf").for_pool())
    assert outcome.ok is False
    assert outcome.code == "handler_error"


class TestWorkerCounts:
    def test_fast_uses_physical_cores_minus_one(self) -> None:
        assert worker_count("fast", physical_cores=8, available_ram=64 * GB) == 7

    def test_a_16_gb_16_core_machine_is_not_held_to_eight_fast_workers(self) -> None:
        assert worker_count("fast", physical_cores=16, available_ram=16 * GB) == 15

    def test_quality_uses_a_quarter_of_the_cores(self) -> None:
        assert worker_count("quality", physical_cores=16, available_ram=128 * GB) == 4

    def test_there_is_always_at_least_one_worker(self) -> None:
        assert worker_count("fast", physical_cores=1, available_ram=64 * GB) == 1
        assert worker_count("quality", physical_cores=2, available_ram=64 * GB) == 1
        assert worker_count("fast", physical_cores=8, available_ram=1) == 1

    def test_available_ram_caps_fast_at_about_150_mb_a_worker(self) -> None:
        assert worker_count("fast", physical_cores=64, available_ram=1 * GB) == 6

    def test_available_ram_caps_quality_at_about_2_gb_a_worker(self) -> None:
        assert worker_count("quality", physical_cores=64, available_ram=5 * GB) == 2

    def test_ocr_workers_overrides_the_default(self) -> None:
        assert worker_count("fast", physical_cores=8, available_ram=64 * GB, ocr_workers=3) == 3
        assert worker_count("fast", physical_cores=8, available_ram=1 * GB, ocr_workers=30) == 30

    def test_the_defaults_come_from_this_machine(self) -> None:
        assert worker_count("fast") >= 1


class TestDocumentTimeout:
    def test_fast_allows_sixty_seconds_plus_twenty_per_ocr_page(self) -> None:
        assert document_timeout("fast", pages=50, ocr_pages=0) == 60
        assert document_timeout("fast", pages=50, ocr_pages=3) == 120

    def test_quality_allows_sixty_seconds_plus_six_per_page_of_the_document(self) -> None:
        assert document_timeout("quality", pages=10, ocr_pages=0) == 120
        assert document_timeout("quality", pages=10, ocr_pages=10) == 120

    def test_it_is_capped_at_thirty_minutes(self) -> None:
        assert document_timeout("fast", pages=2000, ocr_pages=1000) == 1800
        assert document_timeout("quality", pages=2000, ocr_pages=0) == 1800


def test_worker_settings_size_a_pool_and_set_its_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = WorkerSettings(ocr_workers=2, threads_per_worker=3)
    pool = settings.pool("fast")
    try:
        assert pool.size == 2
        assert run(pool, "worker_handlers:omp_thread_limit").result == "3"
    finally:
        pool.close()


def test_worker_settings_default_to_one_thread_for_fast_and_four_for_quality() -> None:
    fast, quality = WorkerSettings().pool("fast"), WorkerSettings(ocr_workers=1).pool("quality")
    try:
        assert fast.threads_per_worker == 1
        assert quality.threads_per_worker == 4
    finally:
        fast.close()
        quality.close()
