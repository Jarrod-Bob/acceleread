# SPDX-License-Identifier: Apache-2.0
"""Catalog (Jobs, FIFO, runner lease), Job directories, inputs, retention (spec §8, §7.2)."""

import json
import os
from datetime import timedelta
from pathlib import Path

import pytest

from acceleread.workspace import (
    InputChangedError,
    JobRunningError,
    Workspace,
)


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, **kwargs: float) -> None:
        self.now += timedelta(**kwargs).total_seconds()


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def ws(tmp_path: Path, clock: Clock):
    with Workspace.open(tmp_path / "ws", clock=clock, filesystem_type=lambda _p: "apfs") as w:
        yield w


def test_job_gets_its_directory_and_manifest(ws: Workspace):
    job = ws.create_job({"taxonomy": "t", "model": "m"})
    directory = ws.job_dir(job.id)
    assert directory == ws.path / "jobs" / job.id
    assert json.loads((directory / "manifest.json").read_text()) == {"taxonomy": "t", "model": "m"}
    assert (directory / "job.sqlite").is_file()
    assert (directory / "inputs").is_dir()
    assert job.state == "queued"
    assert job.kind == "job"


def test_jobs_run_in_fifo_order(ws: Workspace):
    first = ws.create_job({})
    second = ws.create_job({})
    third = ws.create_job({})
    assert ws.next_queued() == first.id
    ws.set_job_state(first.id, "running")
    ws.set_job_state(first.id, "done")
    assert ws.next_queued() == second.id
    assert [j.id for j in ws.list_jobs()] == [first.id, second.id, third.id]


def test_next_queued_is_none_when_nothing_waits(ws: Workspace):
    assert ws.next_queued() is None


def test_ingest_jobs_are_hidden_from_lists_by_default(ws: Workspace):
    job = ws.create_job({})
    ingest = ws.create_job({}, kind="ingest")
    assert [j.id for j in ws.list_jobs()] == [job.id]
    assert {j.id for j in ws.list_jobs(include_ingest=True)} == {job.id, ingest.id}


def test_summary_is_written_at_job_end(ws: Workspace):
    job = ws.create_job({})
    ws.finish_job(job.id, "done", summary={"documents": 3})
    assert json.loads((ws.job_dir(job.id) / "summary.json").read_text()) == {"documents": 3}
    assert ws.get_job(job.id).state == "done"


def test_get_unknown_job_raises(ws: Workspace):
    with pytest.raises(KeyError):
        ws.get_job("nope")


def test_catalog_survives_reopening(tmp_path: Path):
    with Workspace.open(tmp_path) as first:
        job = first.create_job({"a": 1})
    with Workspace.open(tmp_path) as again:
        assert [j.id for j in again.list_jobs()] == [job.id]


# Runner lease


def test_one_holder_at_a_time(ws: Workspace):
    assert ws.acquire_lease("runner-a")
    assert not ws.acquire_lease("runner-b")
    assert ws.lease_holder() == "runner-a"


def test_holder_can_reacquire_and_heartbeat(ws: Workspace, clock: Clock):
    assert ws.acquire_lease("a", ttl=30)
    clock.advance(seconds=20)
    assert ws.heartbeat("a")
    clock.advance(seconds=20)  # 40s since acquire, but only 20s since the heartbeat
    assert not ws.acquire_lease("b", ttl=30)


def test_stale_lease_is_taken_over(ws: Workspace, clock: Clock):
    ws.acquire_lease("a", ttl=30)
    clock.advance(seconds=31)
    assert ws.lease_holder() is None
    assert ws.acquire_lease("b", ttl=30)
    assert not ws.heartbeat("a")  # the old holder learns it lost the lease
    assert ws.lease_holder() == "b"


def test_release_frees_the_lease(ws: Workspace):
    ws.acquire_lease("a")
    ws.release_lease("a")
    assert ws.acquire_lease("b")


def test_only_the_holder_can_release(ws: Workspace):
    ws.acquire_lease("a")
    ws.release_lease("b")
    assert ws.lease_holder() == "a"


def test_lease_is_shared_across_connections(tmp_path: Path):
    with Workspace.open(tmp_path) as one, Workspace.open(tmp_path) as two:
        assert one.acquire_lease("a")
        assert not two.acquire_lease("b")


# Inputs


def test_uploads_are_copied_by_content_hash(ws: Workspace):
    job = ws.create_job({})
    ref = ws.job_inputs(job.id).add_bytes(b"%PDF-hello")
    assert ref.kind == "copy"
    assert ref.bytes == 10
    assert Path(ref.path).parent == ws.job_dir(job.id) / "inputs"
    assert Path(ref.path).name == ref.sha256
    assert Path(ref.path).read_bytes() == b"%PDF-hello"


def test_identical_uploads_are_stored_once(ws: Workspace):
    inputs = ws.job_inputs(ws.create_job({}).id)
    a = inputs.add_bytes(b"same")
    b = inputs.add_bytes(b"same")
    assert a == b
    assert len(list(Path(a.path).parent.iterdir())) == 1


def test_local_paths_are_referenced_not_copied(ws: Workspace, tmp_path: Path):
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"content")
    job = ws.create_job({})
    ref = ws.job_inputs(job.id).reference_path(source)
    assert ref.kind == "path"
    assert (ref.path, ref.bytes) == (str(source), 7)
    assert list((ws.job_dir(job.id) / "inputs").iterdir()) == []
    assert ws.job_inputs(job.id).open(ref).read() == b"content"


def test_changed_local_file_is_detected(ws: Workspace, tmp_path: Path):
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"content")
    inputs = ws.job_inputs(ws.create_job({}).id)
    ref = inputs.reference_path(source)
    source.write_bytes(b"CONTENT")  # same size, different bytes
    with pytest.raises(InputChangedError, match="changed"):
        inputs.open(ref)


def test_missing_local_file_is_detected(ws: Workspace, tmp_path: Path):
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"content")
    inputs = ws.job_inputs(ws.create_job({}).id)
    ref = inputs.reference_path(source)
    source.unlink()
    with pytest.raises(InputChangedError, match="missing"):
        inputs.open(ref)


def test_input_refs_are_remembered_per_document(ws: Workspace):
    job = ws.create_job({})
    ref = ws.job_inputs(job.id).add_bytes(b"x")
    with ws.open_job(job.id) as store:
        store.add_documents(["d1"])
        store.set_input_ref("d1", ref)
        assert store.get_input_ref("d1") == ref
        assert store.get_input_ref("missing") is None


# Retention


def test_delete_removes_directory_and_catalog_row(ws: Workspace):
    job = ws.create_job({})
    ws.finish_job(job.id, "done")
    ws.delete_job(job.id)
    assert not ws.job_dir(job.id).exists()
    assert ws.list_jobs() == []


def test_delete_is_refused_while_running(ws: Workspace):
    job = ws.create_job({})
    ws.set_job_state(job.id, "running")
    with pytest.raises(JobRunningError):
        ws.delete_job(job.id)
    assert ws.job_dir(job.id).exists()


def _age(ws: Workspace, clock: Clock, job_id: str, days: int) -> None:
    ws.finish_job(job_id, "done")
    clock.advance(days=days)


def test_prune_removes_only_finished_jobs_older_than_cutoff(ws: Workspace, clock: Clock):
    old = ws.create_job({})
    _age(ws, clock, old.id, 10)
    recent = ws.create_job({})
    _age(ws, clock, recent.id, 1)
    running = ws.create_job({})
    ws.set_job_state(running.id, "running")
    queued = ws.create_job({})
    removed = ws.prune_jobs(older_than=timedelta(days=5))
    assert removed == [old.id]
    assert {j.id for j in ws.list_jobs()} == {recent.id, running.id, queued.id}


def test_prune_can_target_ingest_jobs_only(ws: Workspace, clock: Clock):
    job = ws.create_job({})
    ingest = ws.create_job({}, kind="ingest")
    _age(ws, clock, job.id, 0)
    _age(ws, clock, ingest.id, 10)
    assert ws.prune_jobs(older_than=timedelta(days=1), kind="ingest") == [ingest.id]
    assert [j.id for j in ws.list_jobs()] == [job.id]


def test_prune_keep_records_frees_inputs_but_keeps_the_job(ws: Workspace, clock: Clock):
    job = ws.create_job({})
    ref = ws.job_inputs(job.id).add_bytes(b"big upload")
    _age(ws, clock, job.id, 10)
    assert ws.prune_jobs(older_than=timedelta(days=5), keep_records=True) == [job.id]
    assert not Path(ref.path).exists()
    assert ws.get_job(job.id).pruned
    with ws.read_job(job.id):  # still exportable
        pass
    with pytest.raises(PermissionError, match="pruned"):
        ws.open_job(job.id)  # but not retryable


def test_prune_leaves_no_stray_files(ws: Workspace, clock: Clock):
    job = ws.create_job({})
    _age(ws, clock, job.id, 10)
    ws.prune_jobs(older_than=timedelta(days=5))
    assert os.listdir(ws.path / "jobs") == []


# Judgment cache


def test_cache_returns_what_was_stored(ws: Workspace):
    assert ws.cache.get("k") is None
    ws.cache.put("k", {"value": "tech", "p": 0.9})
    assert ws.cache.get("k") == {"value": "tech", "p": 0.9}


def test_cache_is_shared_across_jobs_and_reopens(tmp_path: Path):
    with Workspace.open(tmp_path) as first:
        first.cache.put("k", {"v": 1})
    with Workspace.open(tmp_path) as again:
        assert again.cache.get("k") == {"v": 1}


def test_cache_put_overwrites(ws: Workspace):
    ws.cache.put("k", {"v": 1})
    ws.cache.put("k", {"v": 2})
    assert ws.cache.get("k") == {"v": 2}


def test_cache_clear_empties_it(ws: Workspace):
    ws.cache.put("a", {})
    ws.cache.put("b", {})
    assert ws.cache.clear() == 2
    assert ws.cache.get("a") is None


def test_cache_prune_drops_entries_not_used_recently(ws: Workspace, clock: Clock):
    ws.cache.put("stale", {})
    ws.cache.put("used", {})
    clock.advance(days=10)
    assert ws.cache.get("used") == {}  # a hit refreshes the entry
    assert ws.cache.prune(older_than=timedelta(days=5)) == 1
    assert ws.cache.get("stale") is None
    assert ws.cache.get("used") == {}


def test_cache_with_another_format_version_is_discarded_not_misread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from acceleread.workspace import cache

    with Workspace.open(tmp_path) as first:
        first.cache.put("k", {"v": 1})
    monkeypatch.setattr(cache, "CACHE_FORMAT_VERSION", cache.CACHE_FORMAT_VERSION + 1)
    with Workspace.open(tmp_path) as again:
        assert again.cache.get("k") is None
        again.cache.put("k", {"v": 2})
        assert again.cache.get("k") == {"v": 2}
