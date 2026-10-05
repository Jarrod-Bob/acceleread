# SPDX-License-Identifier: Apache-2.0
"""The runner's concurrency and policy edges (docs/spec/v0.md §7.2, §7.5, §11): the single SQLite
writer, claiming Jobs, a runner that survives a bad Job, auto-cancels, billing and estimates."""

import asyncio
import json
import os
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import pytest

from acceleread import validate
from acceleread.classifier import ClassifierRejected, ClassifierUnavailable
from acceleread.extract import Extracted
from acceleread.jobcontrol import (
    apply_requests,
    cancel_job,
    finalize_job,
    job_summary,
    pending_requests,
    post_request,
    progress_marker,
    resume_job,
)
from acceleread.models import DocumentRecord, JobSpec, Question, Taxonomy
from acceleread.ratelimit import STALL_AFTER
from acceleread.runner import LeaseLostError, Runner, run
from acceleread.workers import Outcome, WorkerSettings
from acceleread.workspace import Workspace

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = FIXTURES / "sample.pdf"
FILING = FIXTURES / "filing10k.pdf"
TAXONOMY = Taxonomy.from_file(FIXTURES / "taxonomy.yaml")
Fake = Callable[..., Any]


@pytest.fixture
def ws(tmp_path: Path, clock: Any) -> Iterator[Workspace]:
    with Workspace.open(tmp_path / "ws", clock=clock.now) as workspace:
        yield workspace


@pytest.fixture
def other(ws: Workspace, clock: Any) -> Iterator[Workspace]:
    """A second handle on the same Workspace: another process."""
    with Workspace.open(ws.path, clock=clock.now) as workspace:
        yield workspace


def spec(tmp_path: Path, count: int = 1, **kw: Any) -> JobSpec:
    paths = []
    for i in range(count):
        target = tmp_path / f"doc{i}.pdf"
        target.write_bytes(SAMPLE.read_bytes())
        paths.append(target)
    return JobSpec(inputs=paths, taxonomy=TAXONOMY, **kw)


def runner_for(ws: Workspace, classifier: Any, clock: Any, **kw: Any) -> Runner:
    kw.setdefault("poll_interval", 0.02)
    kw.setdefault("cancel_grace", 0.05)
    return Runner(ws, classifier, clock=clock.now, **kw)


def records_of(ws: Workspace, job_id: str) -> list[DocumentRecord]:
    with ws.read_job(job_id) as store:
        found = [store.get_record(d) for d in store.states()]
    return [r for r in found if r is not None]


def summary_file(ws: Workspace, job_id: str) -> dict[str, Any]:
    return json.loads((ws.job_dir(job_id) / "summary.json").read_text())


# The single writer (§7.2)


async def test_a_live_holder_applies_cancel_requests_other_processes_only_post(
    ws: Workspace, other: Workspace, tmp_path: Path, fake_classifier: Fake, clock: Any
):
    runner = runner_for(ws, fake_classifier(), clock, holder="holder")
    assert runner.acquire()
    job_id = runner.submit(spec(tmp_path))

    assert cancel_job(other, job_id) == "requested"

    assert other.get_job(job_id).state == "queued"  # the other process wrote nothing
    (request,) = pending_requests(other)
    assert (request.kind, request.job_id, request.reason) == ("cancel", job_id, "user")
    runner.apply_requests()
    assert ws.get_job(job_id).state == "cancelled"
    assert pending_requests(ws) == []


async def test_with_no_live_holder_the_caller_takes_the_lease_to_write(
    ws: Workspace, other: Workspace, tmp_path: Path, fake_classifier: Fake, clock: Any
):
    job_id = runner_for(ws, fake_classifier(), clock).submit(spec(tmp_path))
    assert cancel_job(other, job_id) == "cancelled"
    assert ws.get_job(job_id).state == "cancelled"
    assert ws.lease_holder() is None  # released again


async def test_resume_goes_through_the_holder_too(
    ws: Workspace, other: Workspace, tmp_path: Path, fake_classifier: Fake, clock: Any
):
    runner = runner_for(ws, fake_classifier(), clock, holder="holder")
    job_id = runner.submit(spec(tmp_path))
    cancel_job(ws, job_id)
    assert runner.acquire()

    result = resume_job(other, job_id)

    assert result.requested and result.count == 0
    assert other.get_job(job_id).state == "cancelled"
    runner.apply_requests()
    assert ws.get_job(job_id).state == "queued"


async def test_resume_discards_a_stale_cancel_request(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake, clock: Any
):
    job_id = runner_for(ws, fake_classifier(), clock).submit(spec(tmp_path))
    cancel_job(ws, job_id)
    post_request(ws, "cancel", job_id)  # left over from before

    assert resume_job(ws, job_id).count == 1
    assert pending_requests(ws) == []
    assert ws.get_job(job_id).state == "queued"


async def test_requests_for_finished_or_missing_jobs_are_dropped(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake, clock: Any
):
    runner = runner_for(ws, fake_classifier(), clock)
    job_id = runner.submit(spec(tmp_path))
    await runner.execute(job_id)
    post_request(ws, "cancel", job_id)
    post_request(ws, "cancel", "no-such-job")
    (ws.path / "control" / ".half-written.tmp").write_text("{")  # a writer that has not renamed

    apply_requests(ws)

    assert pending_requests(ws) == []  # nothing outlives its Job, and the temp file is ignored
    assert ws.get_job(job_id).state == "done"


async def test_finalising_a_job_clears_its_cancel_requests(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake, clock: Any
):
    job_id = runner_for(ws, fake_classifier(), clock).submit(spec(tmp_path))
    post_request(ws, "cancel", job_id)
    finalize_job(ws, job_id, "done")
    assert pending_requests(ws) == []


async def test_a_job_is_claimed_once(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake, clock: Any
):
    classifier = fake_classifier()
    runner = runner_for(ws, classifier, clock)
    job_id = runner.submit(spec(tmp_path))
    cancel_job(ws, job_id)  # the cancel wins
    assert await runner.execute(job_id) == "cancelled"  # and the start does not
    assert classifier.calls == []

    second = runner.submit(spec(tmp_path))
    assert ws.claim_job(second) is True
    assert ws.claim_job(second) is False


async def test_serve_survives_a_job_that_fails_and_keeps_the_fifo_going(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake, clock: Any
):
    runner = runner_for(ws, fake_classifier(), clock, holder="serve")
    broken = runner.submit(spec(tmp_path))
    good = runner.submit(spec(tmp_path))
    (ws.job_dir(broken) / "manifest.json").write_text("{ not json")
    stop = asyncio.Event()
    server = asyncio.create_task(runner.serve(stop))

    await until_state(ws, good, "done")
    stop.set()
    await server

    assert ws.get_job(broken).state == "failed"
    assert ws.lease_holder() is None  # released in `finally`


async def until_state(ws: Workspace, job_id: str, state: str, timeout: float = 10.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while ws.get_job(job_id).state != state:
        assert asyncio.get_running_loop().time() < deadline, "state not reached"
        await asyncio.sleep(0.005)


async def test_a_lost_heartbeat_stops_the_job_and_the_lease_is_never_retaken(
    ws: Workspace,
    other: Workspace,
    tmp_path: Path,
    fake_classifier: Fake,
    clock: Any,
    until: Fake,
):
    classifier = fake_classifier(gate=asyncio.Event())
    runner = runner_for(ws, classifier, clock, holder="old", sleep=clock.sleep)
    assert runner.acquire()
    job_id = runner.submit(spec(tmp_path))
    task = asyncio.create_task(runner.execute(job_id))
    await until(lambda: classifier.calls)

    clock.advance(31)  # the lease goes stale, and another runner takes it before we beat
    assert other.acquire_lease("new")

    with pytest.raises(LeaseLostError):
        await task
    assert other.lease_holder() == "new"  # we did not take it back
    assert ws.get_job(job_id).state == "running"  # left for the holder to recover


async def test_the_lease_is_kept_alive_while_a_job_runs(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake, clock: Any, until: Fake
):
    gate = asyncio.Event()
    classifier = fake_classifier(gate=gate)
    runner = runner_for(ws, classifier, clock, holder="beat", sleep=clock.sleep, poll_interval=1.0)
    assert runner.acquire()
    started = clock.now()
    job_id = runner.submit(spec(tmp_path))
    task = asyncio.create_task(runner.execute(job_id))

    await until(lambda: clock.now() > started + 100)  # several lease lifetimes of fake time

    assert ws.lease_holder() == "beat"
    gate.set()
    assert await task == "done"


async def test_abandoning_a_stream_asks_the_runner_that_holds_the_job_to_cancel_it(
    ws: Workspace,
    other: Workspace,
    tmp_path: Path,
    fake_classifier: Fake,
    clock: Any,
    until: Fake,
):
    served = fake_classifier(gate=asyncio.Event())
    server = runner_for(other, served, clock, holder="serve", cancel_grace=0.05)
    stop = asyncio.Event()
    serving = asyncio.create_task(server.serve(stop))
    await until(lambda: ws.lease_holder() == "serve")

    stream = run(spec(tmp_path), fake_classifier(), workspace=ws, poll_interval=0.02)
    first = asyncio.ensure_future(stream.__anext__())
    await until(lambda: served.calls)
    first.cancel()
    await asyncio.gather(first, return_exceptions=True)
    await stream.aclose()  # the consumer walked away

    (job,) = ws.list_jobs()
    await until_state(ws, job.id, "cancelled")
    stop.set()
    await serving


# Auto-cancels end the Job cancelled with their reason, even on the last Document (§7.5)


async def test_a_rejection_on_the_last_document_cancels_the_job(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake, clock: Any
):
    runner = runner_for(ws, fake_classifier(raises=lambda _: ClassifierRejected("422")), clock)
    job_id = runner.submit(spec(tmp_path))
    assert await runner.execute(job_id) == "cancelled"
    assert summary_file(ws, job_id)["flags"]["cancel_reason"] == "classifier_rejected"


async def test_reaching_the_spend_cap_on_the_last_document_cancels_the_job(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake, clock: Any
):
    classifier = fake_classifier(model="jev-fake", input_tokens=1_000_000)
    runner = runner_for(ws, classifier, clock)
    job_id = runner.submit(spec(tmp_path, max_cost_usd=0.01))
    assert await runner.execute(job_id) == "cancelled"
    assert [r.status for r in records_of(ws, job_id)] == ["ok"]  # it did finish its Document
    assert summary_file(ws, job_id)["flags"]["cancel_reason"] == "spend_cap"


class FinishingPool:
    """A worker pool whose one task succeeds, but whose crash rate has reached the limit."""

    size = 1
    cancel_reason: str | None = None

    def submit(self, handler: str, payload: Any) -> "Future[Outcome]":
        self.cancel_reason = "crash_rate"
        future: Future[Outcome] = Future()
        future.set_result(
            Outcome(True, result=Extracted(text="Hello world.", pages=[], title=None))
        )
        return future

    def cancel(self, reason: str) -> None: ...

    def close(self) -> None: ...


class FinishingSettings(WorkerSettings):
    def pool(self, profile: str = "fast") -> Any:  # type: ignore[override]
        return FinishingPool()


async def test_a_crash_rate_on_the_last_document_cancels_the_job(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake, clock: Any
):
    runner = runner_for(ws, fake_classifier(), clock, workers=FinishingSettings())
    job_id = runner.submit(spec(tmp_path))
    assert await runner.execute(job_id) == "cancelled"
    assert summary_file(ws, job_id)["flags"]["cancel_reason"] == "crash_rate"


# Billing and pricing


async def test_the_summary_always_prices_from_the_workspace_config(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake, clock: Any
):
    (ws.path / "config.yaml").write_text("prices:\n  jev: 1.0\n")
    classifier = fake_classifier(model="jev-fake", input_tokens=1_000_000)
    runner = runner_for(ws, classifier, clock)
    job_id = runner.submit(spec(tmp_path))
    await runner.execute(job_id)
    assert summary_file(ws, job_id)["classifier"]["estimated_cost_usd"] == pytest.approx(1.0)
    assert job_summary(ws, job_id)["classifier"]["estimated_cost_usd"] == pytest.approx(1.0)


async def test_a_call_in_flight_when_the_job_is_cancelled_is_still_billed(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake, clock: Any, until: Fake
):
    gate = asyncio.Event()
    classifier = fake_classifier(model="jev-fake", input_tokens=1_000_000, gate=gate)
    runner = runner_for(ws, classifier, clock, classify_concurrency=1, cancel_grace=30)
    assert runner.acquire()
    job_id = runner.submit(spec(tmp_path, 2, cache=False))
    task = asyncio.create_task(runner.execute(job_id))
    await until(lambda: classifier.calls)

    cancel_job(ws, job_id)
    await until(lambda: pending_requests(ws) == [])  # the runner has seen it
    gate.set()  # the response arrives after the cancel

    assert await task == "cancelled"
    first, second = records_of(ws, job_id)
    assert first.classification is not None and first.usage.input_tokens == 1_000_000
    assert second.status == "cancelled"
    assert summary_file(ws, job_id)["classifier"]["estimated_cost_usd"] == pytest.approx(0.042)


async def test_a_retried_record_carries_this_attempts_usage_and_the_job_keeps_the_spend(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake, clock: Any
):
    # Two request groups: the whole Document, and its `business` Section.
    spec_ = JobSpec(
        inputs=[FILING],
        cache=False,
        questions=[
            Question(name="a", kind="noul", instructions="Is `document` short?"),
            Question(
                name="b", kind="noul", instructions="Is `document` solar?", reads=["business"]
            ),
        ],
    )
    flaky = fake_classifier(
        model="jev-fake",
        input_tokens=1_000_000,
        raises=lambda state: (
            ClassifierUnavailable("503") if "sections" in state["document"] else None
        ),  # type: ignore[call-overload, operator]
    )
    from acceleread.ratelimit import RateLimit, RateLimitedClassifier

    first = Runner(
        ws,
        RateLimitedClassifier(flaky, RateLimit(1e9, 1e6), max_unavailable_tries=1),
        clock=clock.now,
    )
    job_id = first.submit(spec_)
    await first.execute(job_id)
    (failed,) = records_of(ws, job_id)
    assert failed.status == "failed" and failed.usage.input_tokens == 1_000_000  # group `a` billed

    from acceleread.jobcontrol import retry_failed

    assert retry_failed(ws, job_id).count == 1
    healthy = runner_for(ws, fake_classifier(model="jev-fake", input_tokens=1_000_000), clock)
    await healthy.execute(job_id)

    (record,) = records_of(ws, job_id)
    assert record.attempts == 2 and record.status == "ok"
    assert record.usage.input_tokens == 2_000_000  # this attempt only: it matches its Judgments
    # ...but the Job paid for all of it: 1M for the first attempt, 2M for the second.
    assert job_summary(ws, job_id)["classifier"]["estimated_cost_usd"] == pytest.approx(0.126)


# Stalled, concurrency limits and estimates


async def test_stalled_is_derived_when_the_summary_is_read(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake, clock: Any
):
    job_id = runner_for(ws, fake_classifier(), clock).submit(spec(tmp_path))
    assert ws.claim_job(job_id)  # running, with its Document still waiting
    marker = progress_marker(ws, job_id)
    marker.write_text("")
    now = 2_000_000.0

    os.utime(marker, (now - 60, now - 60))
    assert job_summary(ws, job_id, now=now)["flags"]["stalled"] is False
    os.utime(marker, (now - STALL_AFTER - 1, now - STALL_AFTER - 1))
    assert job_summary(ws, job_id, now=now)["flags"]["stalled"] is True


async def test_the_configured_max_in_flight_sets_how_many_documents_classify_at_once(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake, clock: Any, until: Fake
):
    (ws.path / "config.yaml").write_text(
        "classifier:\n  rate_limit:\n    tokens_per_s: 1000000\n    requests_per_s: 1000\n"
        "    max_in_flight: 2\n"
    )
    gate = asyncio.Event()
    classifier = fake_classifier(gate=gate)
    runner = runner_for(ws, classifier, clock)
    job_id = runner.submit(spec(tmp_path, 5, cache=False))
    task = asyncio.create_task(runner.execute(job_id))
    await until(lambda: len(classifier.calls) >= 2)
    for _ in range(100):  # give a third call every chance to start
        await asyncio.sleep(0)
    assert classifier.in_flight == 2

    gate.set()
    assert await task == "done"
    assert classifier.max_in_flight == 2


def test_validate_estimates_cost_and_duration_from_the_planner_and_the_price_table(
    tmp_path: Path,
):
    one = validate(JobSpec(inputs=[SAMPLE], taxonomy=TAXONOMY)).estimate
    two = validate(JobSpec(inputs=[SAMPLE, SAMPLE], taxonomy=TAXONOMY)).estimate
    assert not one.stubbed and one.documents == 1
    assert one.cost_usd is not None and 0 < one.cost_usd < 0.01  # a 2-page Document at $0.042/1M
    assert one.duration_seconds is not None and one.duration_seconds > 0
    assert two.cost_usd == pytest.approx(2 * one.cost_usd)
