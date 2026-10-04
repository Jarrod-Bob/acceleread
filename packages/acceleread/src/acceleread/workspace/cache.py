# SPDX-License-Identifier: Apache-2.0
"""The Workspace-wide Judgment cache, `cache.sqlite` (ADR 0010).

Keys are opaque to this module: the Classifier seam builds them, and includes the model
version. The file's own format is versioned; a different format is discarded, never misread.
"""

import json
import sqlite3
from collections.abc import Callable
from datetime import timedelta
from typing import Any

from acceleread.workspace import _db

CACHE_FORMAT_VERSION = 1


class JudgmentCache:
    def __init__(self, db: sqlite3.Connection, clock: Callable[[], float]) -> None:
        self._db = db
        self._clock = clock
        with _db.transaction(db):
            if int(db.execute("PRAGMA user_version").fetchone()[0]) != CACHE_FORMAT_VERSION:
                db.execute("DROP TABLE IF EXISTS judgments")
                db.execute(
                    "CREATE TABLE judgments (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
                    " last_used REAL NOT NULL)"
                )
                db.execute(f"PRAGMA user_version = {CACHE_FORMAT_VERSION}")

    def get(self, key: str) -> dict[str, Any] | None:
        row = self._db.execute("SELECT value FROM judgments WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        self._db.execute("UPDATE judgments SET last_used=? WHERE key=?", (self._clock(), key))
        value: dict[str, Any] = json.loads(row[0])
        return value

    def put(self, key: str, value: dict[str, Any]) -> None:
        self._db.execute(
            "INSERT INTO judgments (key, value, last_used) VALUES (?, ?, ?)"
            " ON CONFLICT (key) DO UPDATE SET value=excluded.value, last_used=excluded.last_used",
            (key, json.dumps(value), self._clock()),
        )

    def prune(self, *, older_than: timedelta) -> int:
        """Drop entries not used within `older_than`. Returns how many were dropped."""
        cutoff = self._clock() - older_than.total_seconds()
        return self._db.execute("DELETE FROM judgments WHERE last_used < ?", (cutoff,)).rowcount

    def close(self) -> None:
        self._db.close()

    def clear(self) -> int:
        return self._db.execute("DELETE FROM judgments").rowcount
