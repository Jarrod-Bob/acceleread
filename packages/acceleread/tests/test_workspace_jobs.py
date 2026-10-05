# SPDX-License-Identifier: Apache-2.0
"""Catalog (Jobs, FIFO, runner lease), Job directories, inputs, retention (spec §8, §7.2)."""

import json
from datetime import timedelta
from pathlib import Path

import pytest

from acceleread.models import DocumentRecord, Source
from acceleread.workspace import (
    InputChangedError,
    InputRef,
    JobPrunedError,
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


def test_heartbeat_cannot_revive_an_expired_lease(ws: Workspace, clock: Clock):
    ws.acquire_lease("a", ttl=30)
    clock.advance(seconds=31)
    assert not ws.heartbeat("a")
    assert ws.lease_holder() is None
    assert ws.acquire_lease("a", ttl=30)  # the holder has to re-acquire


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
    assert ref.path == f"inputs/{ref.sha256}"  # relative to the Job directory
    assert (ws.job_dir(job.id) / ref.path).read_bytes() == b"%PDF-hello"
    assert ws.job_inputs(job.id).open(ref).read() == b"%PDF-hello"


def test_identical_uploads_are_stored_once(ws: Workspace):
    inputs = ws.job_inputs(ws.create_job({}).id)
    a = inputs.add_bytes(b"same")
    b = inputs.add_bytes(b"same")
    assert a == b
    assert len(list(inputs.directory.iterdir())) == 1


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


def _record(doc: str, text: str | None = "some filing text") -> DocumentRecord:
    return DocumentRecord(
        record_id=f"rec-{doc}",
        job_id="j",
        status="ok",
        source=Source(filename=f"{doc}.pdf", format="pdf", sha256="ab" * 32, bytes=3),
        text=text,
    )


def _job_with_input_and_record(ws: Workspace) -> tuple[str, InputRef]:
    job = ws.create_job({})
    ref = ws.job_inputs(job.id).add_bytes(b"big upload")
    with ws.open_job(job.id) as store:
        store.add_documents(["d1"])
        store.set_input_ref("d1", ref)
        store.save_record("d1", "done", _record("d1"))
    return job.id, ref


def test_prune_marks_old_finished_jobs_and_leaves_others_alone(ws: Workspace, clock: Clock):
    old = ws.create_job({})
    _age(ws, clock, old.id, 10)
    recent = ws.create_job({})
    _age(ws, clock, recent.id, 1)
    running = ws.create_job({})
    ws.set_job_state(running.id, "running")
    queued = ws.create_job({})
    assert ws.prune_jobs(older_than=timedelta(days=5)) == [old.id]
    assert {j.id for j in ws.list_jobs() if j.pruned} == {old.id}
    assert {j.id for j in ws.list_jobs()} == {old.id, recent.id, running.id, queued.id}


def test_prune_frees_inputs_and_record_text_but_keeps_the_job_exportable(
    ws: Workspace, clock: Clock
):
    job_id, ref = _job_with_input_and_record(ws)
    _age(ws, clock, job_id, 10)
    assert ws.prune_jobs(older_than=timedelta(days=5)) == [job_id]
    assert not (ws.job_dir(job_id) / ref.path).exists()
    assert ws.get_job(job_id).pruned
    with ws.read_job(job_id) as reader:  # still exportable, without text
        record = reader.get_record("d1")
        assert record is not None
        assert record.text is None
        assert record.status == "ok"
    with pytest.raises(JobPrunedError, match="pruned"):
        ws.open_job(job_id)  # but not retryable


def test_prune_can_target_ingest_jobs_only(ws: Workspace, clock: Clock):
    job = ws.create_job({})
    ingest = ws.create_job({}, kind="ingest")
    _age(ws, clock, job.id, 0)
    _age(ws, clock, ingest.id, 10)
    assert ws.prune_jobs(older_than=timedelta(days=1), kind="ingest") == [ingest.id]
    assert [j.id for j in ws.list_jobs() if j.pruned] == []
    assert ws.get_job(ingest.id).pruned


def test_prune_keep_records_frees_inputs_only(ws: Workspace, clock: Clock):
    job_id, ref = _job_with_input_and_record(ws)
    _age(ws, clock, job_id, 10)
    assert ws.prune_jobs(older_than=timedelta(days=5), keep_records=True) == [job_id]
    assert not (ws.job_dir(job_id) / ref.path).exists()
    assert ws.get_job(job_id).pruned
    with ws.read_job(job_id) as reader:
        record = reader.get_record("d1")
        assert record is not None
        assert record.text == "some filing text"
    with pytest.raises(JobPrunedError):
        ws.open_job(job_id)


def test_prune_marks_the_job_before_removing_anything(
    ws: Workspace, clock: Clock, monkeypatch: pytest.MonkeyPatch
):
    job_id, _ref = _job_with_input_and_record(ws)
    _age(ws, clock, job_id, 10)

    def crash(*_a: object, **_k: object) -> None:
        raise OSError("disk went away")

    monkeypatch.setattr("acceleread.workspace.shutil.rmtree", crash)
    with pytest.raises(OSError, match="disk"):
        ws.prune_jobs(older_than=timedelta(days=5))
    assert ws.get_job(job_id).pruned


def test_pruning_twice_reports_nothing_new(ws: Workspace, clock: Clock):
    job_id, _ref = _job_with_input_and_record(ws)
    _age(ws, clock, job_id, 10)
    ws.prune_jobs(older_than=timedelta(days=5))
    assert ws.prune_jobs(older_than=timedelta(days=5)) == []


def test_delete_failure_keeps_the_catalog_row(ws: Workspace, monkeypatch: pytest.MonkeyPatch):
    job = ws.create_job({})
    ws.finish_job(job.id, "done")

    def crash(*_a: object, **_k: object) -> None:
        raise OSError("busy")

    monkeypatch.setattr("acceleread.workspace.shutil.rmtree", crash)
    with pytest.raises(OSError, match="busy"):
        ws.delete_job(job.id)
    assert [j.id for j in ws.list_jobs()] == [job.id]


def test_missing_job_database_is_an_error_not_an_empty_store(ws: Workspace):
    job = ws.create_job({})
    (ws.job_dir(job.id) / "job.sqlite").unlink()
    for name in ("job.sqlite-wal", "job.sqlite-shm"):
        (ws.job_dir(job.id) / name).unlink(missing_ok=True)
    with pytest.raises(FileNotFoundError):
        ws.read_job(job.id)
    with pytest.raises(FileNotFoundError):
        ws.open_job(job.id)
    assert not (ws.job_dir(job.id) / "job.sqlite").exists()


def test_job_ids_come_from_the_injected_clock(ws: Workspace, clock: Clock):
    assert ws.create_job({}).id.startswith(f"{int(clock.now):x}-")


def test_inputs_survive_moving_the_workspace(tmp_path: Path):
    with Workspace.open(tmp_path / "old") as ws:
        job = ws.create_job({})
        ref = ws.job_inputs(job.id).add_bytes(b"portable")
    (tmp_path / "old").rename(tmp_path / "new")
    with Workspace.open(tmp_path / "new") as moved:
        assert moved.job_inputs(job.id).open(ref).read() == b"portable"


def test_failed_copy_leaves_no_temp_file(ws: Workspace, monkeypatch: pytest.MonkeyPatch):
    inputs = ws.job_inputs(ws.create_job({}).id)

    def crash(*_a: object, **_k: object) -> None:
        raise OSError("rename failed")

    monkeypatch.setattr("acceleread.workspace.inputs.os.replace", crash)
    with pytest.raises(OSError, match="rename"):
        inputs.add_bytes(b"data")
    assert list(inputs.directory.iterdir()) == []


def test_pruned_job_error_is_a_domain_error():
    assert not issubclass(JobPrunedError, PermissionError)


def test_opening_a_fresh_workspace_concurrently_is_safe(tmp_path: Path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    barrier = Barrier(6)

    def open_it(_: int) -> int:
        barrier.wait()
        with Workspace.open(tmp_path / "ws") as w:
            w.cache.put("k", {})
            return len(w.list_jobs())

    with ThreadPoolExecutor(6) as pool:
        assert list(pool.map(open_it, range(6))) == [0] * 6


def test_failed_cache_open_does_not_leak_the_catalog_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import sqlite3

    from acceleread.workspace import _db

    opened: list[sqlite3.Connection] = []
    real = _db.connect

    def tracking(path: Path) -> sqlite3.Connection:
        db = real(path)
        opened.append(db)
        return db

    def broken(*_a: object, **_k: object) -> None:
        raise RuntimeError("cache broken")

    monkeypatch.setattr("acceleread.workspace._db.connect", tracking)
    monkeypatch.setattr("acceleread.workspace.JudgmentCache", broken)
    with pytest.raises(RuntimeError, match="cache broken"):
        Workspace.open(tmp_path)
    assert opened
    for db in opened:
        with pytest.raises(sqlite3.ProgrammingError):
            db.execute("SELECT 1")


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
