# SPDX-License-Identifier: Apache-2.0
"""Per-Job storage: Document state, Records, filters, read-only readers, versions (spec §8)."""

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from acceleread.models import (
    ClassifierInfo,
    Coverage,
    DocumentRecord,
    Escalation,
    Judgment,
    Source,
    TaxonomyRef,
)
from acceleread.workspace import (
    JobStore,
    StorageVersionError,
    jobstore,
)

TEXT = "Risk factors. " * 5000


def make_record(
    doc: str = "d1",
    *,
    status: str = "ok",
    category: str | None = "tech",
    confidence: float = 0.9,
    escalation: str = "none",
    text: str | None = TEXT,
) -> DocumentRecord:
    classification = None
    if category is not None:
        classification = Judgment(
            kind="choice",
            value=category,
            probabilities={category: confidence},
            confidence=confidence,
            classifier=ClassifierInfo(id="jev", model="jev-1", version="1"),
            coverage=Coverage(est_tokens=10),
            escalation=Escalation(status=escalation),  # type: ignore[arg-type]
        )
    return DocumentRecord.model_validate(
        {
            "record_id": f"rec-{doc}",
            "job_id": "job1",
            "status": status,
            "source": Source(filename=f"{doc}.pdf", format="pdf", sha256="ab" * 32, bytes=3),
            "text": text,
            "taxonomy": TaxonomyRef(name="t", hash="sha256:x"),
            "classification": classification,
        }
    )


@pytest.fixture
def store(tmp_path: Path):
    with JobStore.create(tmp_path / "job1", job_id="job1") as s:
        yield s


def test_record_round_trips_with_text(store: JobStore):
    store.add_documents(["d1"])
    store.save_record("d1", "done", make_record("d1"))
    assert store.get_record("d1") == make_record("d1")


def test_text_can_be_left_out(store: JobStore):
    store.add_documents(["d1"])
    store.save_record("d1", "done", make_record("d1"))
    assert store.get_record("d1", include_text=False).text is None


def test_text_is_compressed_on_disk(store: JobStore, tmp_path: Path):
    store.add_documents(["d1"])
    store.save_record("d1", "done", make_record("d1"))
    with sqlite3.connect(tmp_path / "job1" / "job.sqlite") as raw:
        blobs = [v for row in raw.execute("SELECT * FROM documents") for v in row]
    assert not any(isinstance(v, bytes | str) and TEXT[:50] in str(v) for v in blobs)
    assert (tmp_path / "job1" / "job.sqlite").stat().st_size < len(TEXT) / 2


def test_record_without_text_round_trips(store: JobStore):
    store.add_documents(["d1"])
    store.save_record("d1", "failed", make_record("d1", status="failed", text=None, category=None))
    assert store.get_record("d1") == make_record("d1", status="failed", text=None, category=None)


def test_new_documents_are_queued_and_unknown_record_is_none(store: JobStore):
    store.add_documents(["d1", "d2"])
    assert store.states() == {"d1": "queued", "d2": "queued"}
    assert store.get_record("d1") is None


def test_state_and_record_are_written_together(store: JobStore):
    store.add_documents(["d1"])
    with pytest.raises(sqlite3.IntegrityError):
        store.save_record("d1", "not-a-state", make_record("d1"))  # type: ignore[arg-type]
    assert store.states() == {"d1": "queued"}
    assert store.get_record("d1") is None


def test_saving_for_unknown_document_is_an_error(store: JobStore):
    with pytest.raises(KeyError):
        store.save_record("nope", "done", make_record("nope"))


def test_set_state_and_documents_in_state_keep_order(store: JobStore):
    store.add_documents(["a", "b", "c"])
    store.set_state("b", "extracting")
    assert store.documents_in_state("queued") == ["a", "c"]
    assert store.documents_in_state("extracting") == ["b"]


def test_saving_again_replaces_the_record(store: JobStore):
    store.add_documents(["d1"])
    store.save_record("d1", "failed", make_record("d1", status="failed", category=None))
    store.save_record("d1", "done", make_record("d1"))
    assert store.get_record("d1") == make_record("d1")
    assert store.find_documents(status="failed") == []


def test_filter_by_status_escalation_category_and_confidence(store: JobStore):
    store.add_documents(["a", "b", "c", "d"])
    store.save_record("a", "done", make_record("a", category="tech", confidence=0.95))
    store.save_record("b", "done", make_record("b", category="bank", confidence=0.4))
    store.save_record(
        "c", "done", make_record("c", category="tech", confidence=0.6, escalation="flagged")
    )
    store.save_record("d", "failed", make_record("d", status="failed", category=None))
    assert store.find_documents(status="failed") == ["d"]
    assert store.find_documents(category="tech") == ["a", "c"]
    assert store.find_documents(escalation_status="flagged") == ["c"]
    assert store.find_documents(max_confidence=0.5) == ["b"]
    assert store.find_documents(min_confidence=0.9) == ["a"]
    assert store.find_documents(category="tech", min_confidence=0.7) == ["a"]
    assert store.find_documents() == ["a", "b", "c", "d"]


def test_filter_by_answer_value(tmp_path: Path):
    class WithAnswers(DocumentRecord):
        answers: dict[str, Any] = {}  # noqa: RUF012  (Answers land with the models issue)

    base = make_record("a").model_dump()
    with JobStore.create(tmp_path / "j", job_id="j") as s:
        s.add_documents(["a", "b"])
        s.save_record(
            "a", "done", WithAnswers(**base, answers={"has_going_concern": {"value": "yes"}})
        )
        s.save_record(
            "b", "done", WithAnswers(**base, answers={"has_going_concern": {"value": "no"}})
        )
        assert s.find_documents(answer=("has_going_concern", "yes")) == ["a"]


def test_reader_sees_committed_records_and_cannot_write(tmp_path: Path):
    with JobStore.create(tmp_path / "j", job_id="j") as writer:
        writer.add_documents(["d1"])
        writer.save_record("d1", "done", make_record("d1"))
        with JobStore.reader(tmp_path / "j") as reader:
            assert reader.get_record("d1") == make_record("d1")
            with pytest.raises(sqlite3.OperationalError):
                reader.set_state("d1", "queued")
            writer.add_documents(["d2"])  # a writer keeps working while a reader is open
            assert reader.states() == {"d1": "done", "d2": "queued"}


def test_job_database_uses_wal(store: JobStore, tmp_path: Path):
    with sqlite3.connect(tmp_path / "job1" / "job.sqlite") as raw:
        assert raw.execute("PRAGMA journal_mode").fetchone() == ("wal",)


def test_new_store_carries_the_current_storage_version(store: JobStore, tmp_path: Path):
    with sqlite3.connect(tmp_path / "job1" / "job.sqlite") as raw:
        version = raw.execute("PRAGMA user_version").fetchone()[0]
    assert version == jobstore.STORAGE_VERSION


def test_older_minor_version_is_migrated_in_place(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    with JobStore.create(tmp_path / "j", job_id="j") as s:
        s.add_documents(["d1"])
        s.save_record("d1", "done", make_record("d1"))
    newer = [
        *jobstore.MIGRATIONS,
        (jobstore.STORAGE_VERSION + 1, "ALTER TABLE documents ADD COLUMN note TEXT"),
    ]
    monkeypatch.setattr(jobstore, "MIGRATIONS", newer)
    monkeypatch.setattr(jobstore, "STORAGE_VERSION", jobstore.STORAGE_VERSION + 1)
    with JobStore.reader(tmp_path / "j") as reader:
        assert reader.get_record("d1") == make_record("d1")
    with sqlite3.connect(tmp_path / "j" / "job.sqlite") as raw:
        assert raw.execute("PRAGMA user_version").fetchone()[0] == jobstore.STORAGE_VERSION
        assert "note" in [r[1] for r in raw.execute("PRAGMA table_info(documents)")]


def _bump_major(monkeypatch: pytest.MonkeyPatch) -> None:
    major = jobstore.STORAGE_VERSION // 1000 + 1
    monkeypatch.setattr(jobstore, "STORAGE_VERSION", major * 1000)
    monkeypatch.setattr(jobstore, "MIGRATIONS", [*jobstore.MIGRATIONS, (major * 1000, "SELECT 1")])


def test_resume_is_refused_across_a_major_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    with JobStore.create(tmp_path / "j", job_id="j") as s:
        s.add_documents(["d1"])
        s.save_record("d1", "done", make_record("d1"))
    _bump_major(monkeypatch)
    with JobStore.reader(tmp_path / "j") as reader:  # still readable and exportable
        assert reader.get_record("d1") == make_record("d1")
    with pytest.raises(StorageVersionError, match="major"):
        JobStore.open_for_run(tmp_path / "j")


def test_open_for_run_works_within_a_major_version(tmp_path: Path):
    JobStore.create(tmp_path / "j", job_id="j").close()
    with JobStore.open_for_run(tmp_path / "j") as s:
        s.add_documents(["x"])


def test_a_job_from_a_newer_major_version_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _bump_major(monkeypatch)
    JobStore.create(tmp_path / "j", job_id="j").close()
    monkeypatch.undo()
    with pytest.raises(StorageVersionError, match="newer"):
        JobStore.reader(tmp_path / "j")
