# SPDX-License-Identifier: Apache-2.0
"""A runner that keeps its lease and its promises when things go wrong (docs/spec/v0.md §7.2,
§7.5): a monitor that survives bad requests, heartbeats through finalising, a `run` that never
runs other people's Jobs, an honest cancel grace, and requests that are not silently lost."""

import asyncio
import json
import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from acceleread.jobcli import describe_request
from acceleread.jobcontrol import (
    add_spend,
    cancel_job,
    finalize_job,
    job_spend,
    pending_requests,
    post_request,
    resume_job,
    retry_failed,
)
from acceleread.models import DocumentRecord, JobSpec, Question, Taxonomy
from acceleread.runner import LeaseLostError, Runner
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
    with Workspace.open(ws.path, clock=clock.now) as workspace:
        yield workspace


def spec(tmp_path: Path, count: int = 1, tag: str = "", **kw: Any) -> JobSpec:
    paths = []
    for i in range(count):
        target = tmp_path / f"{tag}doc{i}.pdf"
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


def corrupt(ws: Workspace, job_id: str) -> None:
    (ws.job_dir(job_id) / "manifest.json").write_text("{ not json")


# The monitor


async def test_the_monitor_survives_a_request_it_cannot_apply_and_keeps_heartbeating(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake, clock: Any, until: Fake
):
    gate = asyncio.Event()
    classifier = fake_classifier(gate=gate)
    runner = runner_for(ws, classifier, clock, holder="beat", sleep=clock.sleep, poll_interval=1.0)
    assert runner.acquire()
    running = runner.submit(spec(tmp_path))
    broken = runner.submit(spec(tmp_path, tag="b"))
    corrupt(ws, broken)  # cancelling it fails with a validation error
    task = asyncio.create_task(runner.execute(running))
    await until(lambda: classifier.calls)

    post_request(ws, "cancel", broken)
    post_request(ws, "bogus", broken)  # type: ignore[arg-type]
    await until(lambda: pending_requests(ws) == [])  # both handled and dropped, not fatal
    started = clock.t
    await until(lambda: clock.t > started + 100)  # lease lifetimes later, still beating

    assert ws.lease_holder() == "beat"
    gate.set()
    assert await task == "done"


async def test_a_transient_sqlite_error_in_the_monitor_is_retried(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake, clock: Any, until: Fake
):
    gate = asyncio.Event()
    classifier = fake_classifier(gate=gate)
    runner = runner_for(ws, classifier, clock, holder="beat", sleep=clock.sleep, poll_interval=1.0)
    assert runner.acquire()
    job_id = runner.submit(spec(tmp_path))
    task = asyncio.create_task(runner.execute(job_id))
    await until(lambda: classifier.calls)

    clock.fail_once = sqlite3.OperationalError("database is locked")
    started = clock.t
    await until(lambda: clock.t > started + 100)

    assert ws.lease_holder() == "beat" and clock.fail_once is None  # it hit the error, and went on
    gate.set()
    assert await task == "done"


async def test_a_monitor_that_dies_stops_the_job(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake, clock: Any, until: Fake
):
    classifier = fake_classifier(gate=asyncio.Event())
    runner = runner_for(ws, classifier, clock, holder="old", sleep=clock.sleep)
    assert runner.acquire()
    job_id = runner.submit(spec(tmp_path))
    task = asyncio.create_task(runner.execute(job_id))
    await until(lambda: classifier.calls)

    clock.fail_once = RuntimeError("the monitor has a bug")

    with pytest.raises(LeaseLostError):  # no heartbeats, so it must not run on unwatched
        await task
    assert ws.get_job(job_id).state == "running"  # left for whoever holds the lease to recover


class BeatCounting(Workspace):
    beats = 0
    beats_at_finish = -1

    def heartbeat(self, holder: str) -> bool:
        type(self).beats += 1
        return super().heartbeat(holder)

    def finish_job(self, *args: Any, **kwargs: Any) -> None:
        type(self).beats_at_finish = type(self).beats
        super().finish_job(*args, **kwargs)


async def test_the_lease_is_heartbeated_while_a_job_is_finalised(
    tmp_path: Path, fake_classifier: Fake, clock: Any
):
    BeatCounting.beats = 0
    with BeatCounting.open(tmp_path / "ws", clock=clock.now) as ws:
        # The monitor never ticks (its sleep is endless), so only finalising can beat.
        runner = Runner(
            ws,
            fake_classifier(),
            clock=clock.now,
            lease_ttl=0.0,
            poll_interval=10_000,
            holder="beat",
        )
        assert runner.acquire()
        job_id = runner.submit(spec(tmp_path, 3))
        before = BeatCounting.beats
        assert await runner.execute(job_id) == "done"
        assert BeatCounting.beats_at_finish > before


def test_finalising_ticks_once_per_document_so_a_big_job_can_beat(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake, clock: Any
):
    job_id = runner_for(ws, fake_classifier(), clock).submit(spec(tmp_path, 3))
    ticks: list[int] = []
    finalize_job(ws, job_id, "done", tick=lambda: ticks.append(1))
    assert len(ticks) >= 3


async def test_serve_is_not_killed_by_a_request_it_cannot_apply(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake, clock: Any, until: Fake
):
    runner = runner_for(ws, fake_classifier(), clock, holder="serve")
    broken = runner.submit(spec(tmp_path, tag="b"))
    corrupt(ws, broken)
    good = runner.submit(spec(tmp_path, tag="g"))
    post_request(ws, "cancel", broken)
    stop = asyncio.Event()
    serving = asyncio.create_task(runner.serve(stop))

    await until(lambda: ws.get_job(good).state == "done")
    assert not serving.done()
    stop.set()
    await serving


# drain


async def test_drain_stops_when_its_job_is_cancelled_while_queued_and_never_runs_the_rest(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake, clock: Any
):
    classifier = fake_classifier()
    runner = runner_for(ws, classifier, clock)
    mine = runner.submit(spec(tmp_path, tag="m"))
    cancel_job(ws, mine)
    theirs = runner.submit(spec(tmp_path, tag="t"))

    await runner.drain(until=mine)
    await runner.drain(until="no-such-job")

    assert ws.get_job(theirs).state == "queued" and classifier.calls == []


async def test_drain_waits_for_a_control_writer_to_let_go_of_the_lease(
    ws: Workspace, other: Workspace, tmp_path: Path, fake_classifier: Fake, clock: Any, until: Fake
):
    runner = runner_for(ws, fake_classifier(), clock, holder="runner", sleep=clock.sleep)
    job_id = runner.submit(spec(tmp_path))
    assert other.acquire_lease("control-writer")
    draining = asyncio.create_task(runner.drain(until=job_id))
    await asyncio.sleep(0.05)
    assert not draining.done()  # waiting, not raising

    other.release_lease("control-writer")
    await draining
    assert ws.get_job(job_id).state == "done"


# Spend ledger


def test_a_torn_last_ledger_line_is_ignored_not_fatal(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake, clock: Any
):
    job_id = runner_for(ws, fake_classifier(), clock).submit(spec(tmp_path))
    add_spend(ws, job_id, "000000", 0.5)
    with (ws.job_dir(job_id) / "spend.log").open("a") as ledger:
        ledger.write("000001 0.")  # the machine died mid-append
    assert job_spend(ws, job_id) == 0.5


# Requests


async def test_a_cancel_posted_after_a_resume_is_not_discarded_by_it(
    ws: Workspace, other: Workspace, tmp_path: Path, fake_classifier: Fake, clock: Any
):
    runner = runner_for(ws, fake_classifier(), clock, holder="holder")
    job_id = runner.submit(spec(tmp_path))
    cancel_job(ws, job_id)
    post_request(ws, "cancel", job_id)  # stale: older than the resume
    assert runner.acquire()
    assert resume_job(other, job_id).requested
    post_request(ws, "cancel", job_id)  # newer than the resume: the user meant it

    runner.apply_requests()

    assert ws.get_job(job_id).state == "cancelled"


async def test_a_retry_posted_for_a_running_job_waits_and_is_applied_after_it(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake, clock: Any, until: Fake
):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf")
    gate = asyncio.Event()
    classifier = fake_classifier(gate=gate)
    runner = runner_for(ws, classifier, clock, holder="holder", sleep=clock.sleep)
    assert runner.acquire()
    job_id = runner.submit(JobSpec(inputs=[broken, *spec(tmp_path).inputs], taxonomy=TAXONOMY))
    task = asyncio.create_task(runner.execute(job_id))
    await until(lambda: classifier.calls)

    post_request(ws, "retry", job_id)  # the Job was claimed before the request was applied
    started = clock.t
    await until(lambda: clock.t > started + 5)
    assert [r.kind for r in pending_requests(ws)] == ["retry"]  # kept, not silently dropped

    gate.set()
    assert await task == "done"
    runner.apply_requests()
    assert pending_requests(ws) == []
    assert ws.get_job(job_id).state == "queued"  # the failed Document is queued again


def test_the_cli_says_who_will_apply_a_posted_request():
    assert "running" in describe_request("resume", live_holder=True)
    assert "next runner" in describe_request("resume", live_holder=False)


# The cancel grace


def two_group_spec() -> JobSpec:
    return JobSpec(
        inputs=[FILING],
        cache=False,
        questions=[
            Question(name="a", kind="noul", instructions="Is `document` short?"),
            Question(
                name="b", kind="noul", instructions="Is `document` solar?", reads=["business"]
            ),
        ],
    )


async def test_during_the_grace_period_no_new_classifier_call_starts(
    ws: Workspace, fake_classifier: Fake, clock: Any, until: Fake
):
    gate = asyncio.Event()
    classifier = fake_classifier(model="jev-fake", input_tokens=1_000_000, gate=gate)
    runner = runner_for(ws, classifier, clock, holder="holder", cancel_grace=30)
    assert runner.acquire()
    job_id = runner.submit(two_group_spec())
    task = asyncio.create_task(runner.execute(job_id))
    await until(lambda: classifier.calls)

    post_request(ws, "cancel", job_id)
    await until(lambda: pending_requests(ws) == [])
    gate.set()  # the call already sent returns

    assert await task == "cancelled"
    assert len(classifier.calls) == 1  # the Document's second request group never went out
    (record,) = records_of(ws, job_id)
    assert record.status == "cancelled" and record.usage.input_tokens == 1_000_000  # billed
    assert len(record.answers) == 1  # what was known is kept


async def test_a_call_cut_off_at_the_grace_deadline_is_noted_on_its_record(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake, clock: Any, until: Fake
):
    classifier = fake_classifier(gate=asyncio.Event())  # never answers
    runner = runner_for(ws, classifier, clock, holder="holder", cancel_grace=0.05)
    assert runner.acquire()
    job_id = runner.submit(spec(tmp_path))
    task = asyncio.create_task(runner.execute(job_id))
    await until(lambda: classifier.calls)

    post_request(ws, "cancel", job_id)
    assert await task == "cancelled"

    (record,) = records_of(ws, job_id)
    assert record.status == "cancelled" and record.text
    assert record.errors[0].code == "cancelled_in_flight"
    assert "cost" in record.errors[0].message


async def test_serve_stops_a_running_job_when_asked_to_shut_down(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake, clock: Any, until: Fake
):
    classifier = fake_classifier(gate=asyncio.Event())
    runner = runner_for(ws, classifier, clock, holder="serve")
    job_id = runner.submit(spec(tmp_path))
    stop = asyncio.Event()
    serving = asyncio.create_task(runner.serve(stop))
    await until(lambda: classifier.calls)

    stop.set()
    await asyncio.wait_for(serving, 10)  # it did not wait for the whole Job

    assert ws.get_job(job_id).state == "cancelled"
    assert (
        json.loads((ws.job_dir(job_id) / "summary.json").read_text())["flags"]["cancel_reason"]
        == "user"
    )
    assert ws.lease_holder() is None


async def test_retry_and_resume_still_refuse_a_job_that_is_running_right_now(
    ws: Workspace, other: Workspace, tmp_path: Path, fake_classifier: Fake, clock: Any
):
    runner = runner_for(ws, fake_classifier(), clock, holder="holder")
    job_id = runner.submit(spec(tmp_path))
    assert runner.acquire() and ws.claim_job(job_id)
    from acceleread.workspace import JobRunningError

    with pytest.raises(JobRunningError):
        retry_failed(other, job_id)
