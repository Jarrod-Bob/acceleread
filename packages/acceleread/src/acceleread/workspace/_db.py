# SPDX-License-Identifier: Apache-2.0
"""SQLite connection helpers shared by the Workspace databases (spec §8)."""

import sqlite3
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
