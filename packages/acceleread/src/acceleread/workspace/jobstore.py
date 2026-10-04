# SPDX-License-Identifier: Apache-2.0
"""One Job's `job.sqlite`: Document state and queue plus Records (spec §8, ADR 0010).

A Document's state and its Record are written in one transaction, so crash-resume never sees
a half-written Record. Text is zlib-compressed in its own column; the rest of the Record is
JSON, with indexed columns for the filters the UI and API use.
"""

import sqlite3
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self

from acceleread.models import DocumentRecord
from acceleread.workspace import _db
from acceleread.workspace.inputs import InputRef

DocumentState = Literal[
    "queued", "extracting", "extracted", "classifying", "done", "failed", "cancelled"
]

# user_version = major * 1000 + minor. Each step runs once, in order, in one transaction.
_SCHEMA_V1 = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE documents (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL UNIQUE,
    state TEXT NOT NULL DEFAULT 'queued' CHECK (state IN
        ('queued','extracting','extracted','classifying','done','failed','cancelled')),
    input_ref TEXT,
    record TEXT,
    has_text INTEGER NOT NULL DEFAULT 0,
    text BLOB,
    status TEXT,
    escalation_status TEXT,
    category TEXT,
    confidence REAL
);
CREATE INDEX documents_state ON documents (state);
CREATE INDEX documents_status ON documents (status);
CREATE INDEX documents_escalation ON documents (escalation_status);
CREATE INDEX documents_category ON documents (category);
CREATE INDEX documents_confidence ON documents (confidence);
CREATE TABLE answer_values (
    doc_id TEXT NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    value TEXT,
    PRIMARY KEY (doc_id, name)
);
CREATE INDEX answer_values_lookup ON answer_values (name, value);
"""

MIGRATIONS: list[tuple[int, str]] = [(1000, _SCHEMA_V1)]
STORAGE_VERSION = 1000


class StorageVersionError(RuntimeError):
    """A Job's storage version cannot be used for the requested operation."""


def _major(version: int) -> int:
    return version // 1000


def _migrate(db: sqlite3.Connection) -> None:
    current = int(db.execute("PRAGMA user_version").fetchone()[0])
    for version, sql in MIGRATIONS:
        if version > current:
            body = sql.strip().rstrip(";")
            db.executescript(f"BEGIN;\n{body};\nPRAGMA user_version = {version};\nCOMMIT;")


def _stored_version(path: Path) -> int:
    with sqlite3.connect(path) as raw:
        return int(raw.execute("PRAGMA user_version").fetchone()[0])


def _index_columns(record: dict[str, Any]) -> dict[str, Any]:
    classification = record.get("classification") or {}
    escalation = classification.get("escalation") or {}
    return {
        "status": record.get("status"),
        "escalation_status": escalation.get("status"),
        "category": classification.get("value"),
        "confidence": classification.get("confidence"),
    }


class JobStore:
    def __init__(self, db: sqlite3.Connection, path: Path) -> None:
        self._db = db
        self.path = path

    @classmethod
    def create(cls, job_dir: Path, *, job_id: str) -> Self:
        job_dir.mkdir(parents=True, exist_ok=True)
        db = _db.connect(job_dir / "job.sqlite")
        _migrate(db)
        db.execute("INSERT OR IGNORE INTO meta VALUES ('job_id', ?)", (job_id,))
        db.execute(
            "INSERT OR IGNORE INTO meta VALUES ('created_major', ?)",
            (str(_major(STORAGE_VERSION)),),
        )
        return cls(db, job_dir)

    @classmethod
    def open_for_run(cls, job_dir: Path) -> Self:
        """Open for resume or retry. Refused if the Job was created under another major version."""
        path = job_dir / "job.sqlite"
        stored = _stored_version(path)
        if _major(stored) > _major(STORAGE_VERSION):
            raise StorageVersionError(f"{job_dir.name} was written by a newer major version")
        db = _db.connect(path)
        row = db.execute("SELECT value FROM meta WHERE key='created_major'").fetchone()
        if row is None or int(row[0]) != _major(STORAGE_VERSION):
            db.close()
            raise StorageVersionError(
                f"{job_dir.name} was created under storage major version {row and row[0]}; "
                f"resume and retry are refused across a major version "
                f"(current: {_major(STORAGE_VERSION)}). It can still be read and exported."
            )
        _migrate(db)
        return cls(db, job_dir)

    @classmethod
    def reader(cls, job_dir: Path) -> Self:
        """A read-only connection, migrating the Job's storage in place first if it is older."""
        path = job_dir / "job.sqlite"
        stored = _stored_version(path)
        if _major(stored) > _major(STORAGE_VERSION):
            raise StorageVersionError(
                f"{job_dir.name} was written by a newer version of acceleread (storage {stored})"
            )
        if stored < STORAGE_VERSION:
            migrator = _db.connect(path)
            try:
                _migrate(migrator)
            finally:
                migrator.close()
        return cls(_db.connect_readonly(path), job_dir)

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        self._db.execute("BEGIN IMMEDIATE")
        try:
            yield self._db
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        self._db.execute("COMMIT")

    def add_documents(self, doc_ids: list[str]) -> None:
        with self._transaction() as db:
            db.executemany(
                "INSERT OR IGNORE INTO documents (doc_id) VALUES (?)", [(d,) for d in doc_ids]
            )

    def states(self) -> dict[str, DocumentState]:
        rows = self._db.execute("SELECT doc_id, state FROM documents ORDER BY seq")
        return dict(rows.fetchall())

    def documents_in_state(self, state: DocumentState) -> list[str]:
        rows = self._db.execute("SELECT doc_id FROM documents WHERE state=? ORDER BY seq", (state,))
        return [r[0] for r in rows]

    def set_state(self, doc_id: str, state: DocumentState) -> None:
        with self._transaction() as db:
            cursor = db.execute("UPDATE documents SET state=? WHERE doc_id=?", (state, doc_id))
            if cursor.rowcount == 0:
                raise KeyError(doc_id)

    def set_input_ref(self, doc_id: str, ref: InputRef) -> None:
        with self._transaction() as db:
            cursor = db.execute(
                "UPDATE documents SET input_ref=? WHERE doc_id=?", (ref.model_dump_json(), doc_id)
            )
            if cursor.rowcount == 0:
                raise KeyError(doc_id)

    def get_input_ref(self, doc_id: str) -> InputRef | None:
        row = self._db.execute("SELECT input_ref FROM documents WHERE doc_id=?", (doc_id,))
        found = row.fetchone()
        return None if found is None or found[0] is None else InputRef.model_validate_json(found[0])

    def save_record(self, doc_id: str, state: DocumentState, record: DocumentRecord) -> None:
        """Write the Document's state and its Record in one transaction."""
        dumped = record.model_dump(mode="json")
        text = record.text
        body = record.model_dump_json(exclude={"text"})
        index = _index_columns(dumped)
        answers = dumped.get("answers") or {}
        with self._transaction() as db:
            cursor = db.execute(
                "UPDATE documents SET state=?, record=?, has_text=?, text=?, status=?,"
                " escalation_status=?, category=?, confidence=? WHERE doc_id=?",
                (
                    state,
                    body,
                    text is not None,
                    None if text is None else zlib.compress(text.encode("utf-8")),
                    index["status"],
                    index["escalation_status"],
                    index["category"],
                    index["confidence"],
                    doc_id,
                ),
            )
            if cursor.rowcount == 0:
                raise KeyError(doc_id)
            db.execute("DELETE FROM answer_values WHERE doc_id=?", (doc_id,))
            db.executemany(
                "INSERT INTO answer_values VALUES (?, ?, ?)",
                [
                    (doc_id, name, None if (v := _answer_value(a)) is None else str(v))
                    for name, a in answers.items()
                ],
            )

    def get_record(self, doc_id: str, *, include_text: bool = True) -> DocumentRecord | None:
        row = self._db.execute(
            "SELECT record, has_text, text FROM documents WHERE doc_id=?", (doc_id,)
        ).fetchone()
        if row is None or row[0] is None:
            return None
        record = DocumentRecord.model_validate_json(row[0])
        if include_text and row[1]:
            record = record.model_copy(update={"text": zlib.decompress(row[2]).decode("utf-8")})
        return record

    def find_documents(
        self,
        *,
        status: str | None = None,
        escalation_status: str | None = None,
        category: str | None = None,
        min_confidence: float | None = None,
        max_confidence: float | None = None,
        answer: tuple[str, str] | None = None,
    ) -> list[str]:
        """Doc ids whose Records match every given filter, in submission order."""
        clauses: list[str] = ["record IS NOT NULL"]
        params: list[Any] = []
        for column, value in (
            ("status", status),
            ("escalation_status", escalation_status),
            ("category", category),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        if min_confidence is not None:
            clauses.append("confidence >= ?")
            params.append(min_confidence)
        if max_confidence is not None:
            clauses.append("confidence <= ?")
            params.append(max_confidence)
        if answer is not None:
            clauses.append(
                "doc_id IN (SELECT doc_id FROM answer_values WHERE name = ? AND value = ?)"
            )
            params.extend(answer)
        sql = f"SELECT doc_id FROM documents WHERE {' AND '.join(clauses)} ORDER BY seq"
        return [r[0] for r in self._db.execute(sql, params)]


def _answer_value(answer: Any) -> Any:
    return answer.get("value") if isinstance(answer, dict) else None
