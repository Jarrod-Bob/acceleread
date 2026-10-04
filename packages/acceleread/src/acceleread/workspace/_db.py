# SPDX-License-Identifier: Apache-2.0
"""SQLite connection helpers shared by the Workspace databases (spec §8)."""

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path


def connect(path: Path) -> sqlite3.Connection:
    """Open a read-write connection in WAL mode. Transactions are explicit (BEGIN/COMMIT)."""
    db = sqlite3.connect(path, isolation_level=None, timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("PRAGMA foreign_keys=ON")
    return db


def connect_readonly(path: Path) -> sqlite3.Connection:
    """Open a read-only connection; any write on it raises sqlite3.OperationalError."""
    db = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=30)
    db.execute("PRAGMA query_only=ON")
    return db


@contextmanager
def transaction(db: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """BEGIN IMMEDIATE ... COMMIT, rolling back if the body raises."""
    db.execute("BEGIN IMMEDIATE")
    try:
        yield db
    except BaseException:
        db.execute("ROLLBACK")
        raise
    db.execute("COMMIT")


def _statements(script: str) -> Iterator[str]:
    pending = ""
    for line in script.splitlines(keepends=True):
        pending += line
        if sqlite3.complete_statement(pending):
            yield pending
            pending = ""
    if pending.strip():
        yield pending if pending.rstrip().endswith(";") else pending + ";"


def migrate(
    db: sqlite3.Connection,
    steps: list[tuple[int, str]],
    *,
    on_apply: Callable[[sqlite3.Connection], None] | None = None,
) -> None:
    """Apply every step newer than `user_version`, in one transaction that also checks it.

    `on_apply` runs inside that transaction when anything was applied, so the caller's own
    bookkeeping rows commit atomically with the schema. Safe against concurrent first opens.
    """
    with transaction(db):
        current = int(db.execute("PRAGMA user_version").fetchone()[0])
        applied = False
        for version, sql in steps:
            if version > current:
                for statement in _statements(sql):
                    db.execute(statement)
                db.execute(f"PRAGMA user_version = {version}")
                applied = True
        if applied and on_apply is not None:
            on_apply(db)
