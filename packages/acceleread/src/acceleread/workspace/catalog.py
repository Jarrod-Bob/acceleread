# SPDX-License-Identifier: Apache-2.0
"""The Workspace catalog, `catalog.sqlite`: Job list, FIFO order and the runner lease (§7.2, §8)."""

import sqlite3
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel

JobState = Literal["queued", "running", "done", "failed", "cancelled"]
JobKind = Literal["job", "ingest"]
FINISHED_STATES = ("done", "failed", "cancelled")

DEFAULT_LEASE_TTL = 30.0

_SCHEMA = """
CREATE TABLE jobs (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL DEFAULT 'job',
    state TEXT NOT NULL DEFAULT 'queued',
    created_at REAL NOT NULL,
    finished_at REAL,
    pruned INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE lease (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    holder TEXT NOT NULL,
    heartbeat_at REAL NOT NULL,
    ttl REAL NOT NULL
);
"""
CATALOG_VERSION = 1000


class JobInfo(BaseModel):
    id: str
    kind: JobKind
    state: JobState
    created_at: float
    finished_at: float | None
    pruned: bool


class JobRunningError(RuntimeError):
    """The Job is running, so it cannot be deleted or pruned."""


class Catalog:
    def __init__(self, db: sqlite3.Connection, clock: Callable[[], float]) -> None:
        self._db = db
        self._clock = clock
        if int(db.execute("PRAGMA user_version").fetchone()[0]) < CATALOG_VERSION:
            db.executescript(
                f"BEGIN;\n{_SCHEMA}\nPRAGMA user_version = {CATALOG_VERSION};\nCOMMIT;"
            )

    # Jobs

    def add_job(self, job_id: str, kind: JobKind) -> JobInfo:
        self._db.execute(
            "INSERT INTO jobs (id, kind, created_at) VALUES (?, ?, ?)",
            (job_id, kind, self._clock()),
        )
        return self.get(job_id)

    def get(self, job_id: str) -> JobInfo:
        rows = self._select("WHERE id=?", (job_id,))
        if not rows:
            raise KeyError(job_id)
        return rows[0]

    def list_jobs(self, *, include_ingest: bool = False) -> list[JobInfo]:
        return self._select("" if include_ingest else "WHERE kind != 'ingest'", ())

    def next_queued(self) -> str | None:
        row = self._db.execute(
            "SELECT id FROM jobs WHERE state='queued' ORDER BY seq LIMIT 1"
        ).fetchone()
        return None if row is None else str(row[0])

    def set_state(self, job_id: str, state: JobState) -> None:
        finished = self._clock() if state in FINISHED_STATES else None
        cursor = self._db.execute(
            "UPDATE jobs SET state=?, finished_at=? WHERE id=?", (state, finished, job_id)
        )
        if cursor.rowcount == 0:
            raise KeyError(job_id)

    def now(self) -> float:
        return self._clock()

    def close(self) -> None:
        self._db.close()

    def mark_pruned(self, job_id: str) -> None:
        self._db.execute("UPDATE jobs SET pruned=1 WHERE id=?", (job_id,))

    def remove(self, job_id: str) -> None:
        self._db.execute("DELETE FROM jobs WHERE id=?", (job_id,))

    def _select(self, where: str, params: tuple[str, ...]) -> list[JobInfo]:
        rows = self._db.execute(
            "SELECT id, kind, state, created_at, finished_at, pruned FROM jobs "
            f"{where} ORDER BY seq",
            params,
        )
        return [
            JobInfo(
                id=r[0], kind=r[1], state=r[2], created_at=r[3], finished_at=r[4], pruned=bool(r[5])
            )
            for r in rows
        ]

    # Runner lease

    def acquire_lease(self, holder: str, ttl: float = DEFAULT_LEASE_TTL) -> bool:
        """Take the lease if it is free, stale or already ours. Only the holder writes."""
        now = self._clock()
        self._db.execute("BEGIN IMMEDIATE")
        try:
            row = self._db.execute("SELECT holder, heartbeat_at, ttl FROM lease").fetchone()
            free = row is None or row[0] == holder or now - row[1] > row[2]
            if free:
                self._db.execute(
                    "INSERT INTO lease (id, holder, heartbeat_at, ttl) VALUES (1, ?, ?, ?)"
                    " ON CONFLICT (id) DO UPDATE SET holder=excluded.holder,"
                    " heartbeat_at=excluded.heartbeat_at, ttl=excluded.ttl",
                    (holder, now, ttl),
                )
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        self._db.execute("COMMIT")
        return free

    def heartbeat(self, holder: str) -> bool:
        """Refresh the lease. False means it was lost to another holder."""
        cursor = self._db.execute(
            "UPDATE lease SET heartbeat_at=? WHERE holder=?", (self._clock(), holder)
        )
        return cursor.rowcount == 1

    def release_lease(self, holder: str) -> None:
        self._db.execute("DELETE FROM lease WHERE holder=?", (holder,))

    def lease_holder(self) -> str | None:
        row = self._db.execute("SELECT holder, heartbeat_at, ttl FROM lease").fetchone()
        if row is None or self._clock() - row[1] > row[2]:
            return None
        return str(row[0])
