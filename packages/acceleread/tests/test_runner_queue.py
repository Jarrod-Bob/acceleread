# SPDX-License-Identifier: Apache-2.0
"""One running Job per runner, FIFO, the runner lease and `acceleread.run` (spec §7.2, ADR 0008)."""

import asyncio
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from acceleread import run
from acceleread.jobcontrol import read_cancel_request
from acceleread.models import DocumentRecord, JobSpec, Taxonomy
from acceleread.runner import LeaseLostError, Runner, run_job
from acceleread.workspace import Workspace

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = FIXTURES / "sample.pdf"
TAXONOMY = Taxonomy.from_file(FIXTURES / "taxonomy.yaml")
Fake = Callable[..., Any]


@pytest.fixture
def ws(tmp_path: Path) -> Iterator[Workspace]:
    with Workspace.open(tmp_path / "ws") as workspace:
        yield workspace


def spec(tmp_path: Path, count: int = 1, tag: str = "") -> JobSpec:
    paths = []
    for i in range(count):
        target = tmp_path / f"{tag}doc{i}.pdf"
        target.write_bytes(SAMPLE.read_bytes())
        paths.append(target)
    return JobSpec(inputs=paths, taxonomy=TAXONOMY)


def runner_for(ws: Workspace, classifier: Any, **kw: Any) -> Runner:
    kw.setdefault("poll_interval", 0.02)
    return Runner(ws, classifier, **kw)


async def test_jobs_run_one_at_a_time_in_fifo_order(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake
):
    gate = asyncio.Event()
    classifier = fake_classifier(gate=gate)
    runner = runner_for(ws, classifier)
    first = runner.submit(spec(tmp_path, 1, "a"))
    second = runner.submit(spec(tmp_path, 1, "b"))
    drain = asyncio.create_task(runner.drain())
    while not classifier.calls:
        await asyncio.sleep(0.02)

    assert ws.get_job(first).state == "running"
    assert ws.get_job(second).state == "queued"  # waits its turn
    gate.set()
    await drain

    assert [ws.get_job(j).state for j in (first, second)] == ["done", "done"]
    done = [ws.get_job(j).finished_at for j in (first, second)]
    assert done[0] is not None and done[1] is not None and done[0] <= done[1]
    assert ws.lease_holder() is None  # the lease is released when the queue is empty


async def test_only_one_runner_holds_the_lease(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake
):
    with Workspace.open(ws.path) as other:
        first = runner_for(ws, fake_classifier(), holder="one")
        second = runner_for(other, fake_classifier(), holder="two")
        assert first.acquire() and not second.acquire()
        first.release()
        assert second.acquire()


async def test_the_lease_is_heartbeated_while_a_job_runs(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake
):
    gate = asyncio.Event()
    classifier = fake_classifier(gate=gate)
    runner = runner_for(ws, classifier, holder="beat", lease_ttl=0.3)
    job_id = runner.submit(spec(tmp_path))
    drain = asyncio.create_task(runner.drain())
    await asyncio.sleep(0.9)  # three lease lifetimes: it would be stale without heartbeats
    assert ws.lease_holder() == "beat"
    assert ws.get_job(job_id).state == "running"
    gate.set()
    await drain


async def test_a_runner_that_loses_its_lease_stops_without_finalising(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake, monkeypatch: pytest.MonkeyPatch
):
    gate = asyncio.Event()
    classifier = fake_classifier(gate=gate)
    runner = runner_for(ws, classifier, holder="old", lease_ttl=0.3)
    job_id = runner.submit(spec(tmp_path))
    assert runner.acquire()
    monkeypatch.setattr(ws, "heartbeat", lambda holder: False)
    monkeypatch.setattr(ws, "acquire_lease", lambda holder, ttl=30.0: False)  # someone else has it

    with pytest.raises(LeaseLostError):
        await runner.execute(job_id)
    assert ws.get_job(job_id).state == "running"  # left for the new holder to recover


async def test_a_new_runner_requeues_a_job_a_dead_runner_left_running(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake
):
    runner = runner_for(ws, fake_classifier())
    job_id = runner.submit(spec(tmp_path, 2))
    ws.set_job_state(job_id, "running")  # the old runner died here
    with ws.open_job(job_id) as store:
        store.set_state("000000", "extracting")

    await runner.drain()

    assert ws.get_job(job_id).state == "done"
    with ws.read_job(job_id) as store:
        assert set(store.states().values()) == {"done"}


async def test_run_streams_records_in_input_order_and_leaves_a_done_job(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake
):
    classifier = fake_classifier()
    records: list[DocumentRecord] = [
        r async for r in run(spec(tmp_path, 3), classifier, workspace=ws)
    ]
    assert [r.source.filename for r in records] == ["doc0.pdf", "doc1.pdf", "doc2.pdf"]
    (job,) = ws.list_jobs()
    assert job.state == "done"


async def test_run_joins_the_fifo_when_serve_holds_the_lease(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake
):
    with Workspace.open(ws.path) as server_ws:
        serving = runner_for(server_ws, fake_classifier(), holder="serve")
        stop = asyncio.Event()
        server = asyncio.create_task(serving.serve(stop))
        await asyncio.sleep(0.1)
        assert ws.lease_holder() == "serve"

        mine = fake_classifier()
        records = [r async for r in run_job(spec(tmp_path, 2), mine, None, ws, poll_interval=0.02)]

        assert [r.status for r in records] == ["ok", "ok"]
        assert mine.calls == []  # serve's runner did the work, run only streamed its Records
        stop.set()
        await server


async def test_run_takes_over_when_the_lease_holder_goes_away(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake
):
    assert ws.acquire_lease("gone", ttl=0.2)  # a runner that never heartbeats again
    records = [
        r async for r in run_job(spec(tmp_path, 1), fake_classifier(), None, ws, poll_interval=0.02)
    ]
    assert [r.status for r in records] == ["ok"]


async def test_abandoning_the_iterator_cancels_the_job(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake
):
    gate = asyncio.Event()
    classifier = fake_classifier(gate=gate)
    stream = run_job(spec(tmp_path, 2), classifier, None, ws, poll_interval=0.02)
    next_record = asyncio.ensure_future(stream.__anext__())
    while not classifier.calls:
        await asyncio.sleep(0.02)
    next_record.cancel()
    await asyncio.gather(next_record, return_exceptions=True)
    await stream.aclose()

    (job,) = ws.list_jobs()
    assert job.state == "cancelled"
    assert read_cancel_request(ws, job.id) is None  # the marker is consumed
