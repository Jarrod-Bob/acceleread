# SPDX-License-Identifier: Apache-2.0
"""The Workspace: where one machine keeps its Jobs, Records, inputs and Judgment cache (§8)."""

import json
import os
import shutil
import sqlite3
import time
import uuid
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from acceleread.workspace import _db
from acceleread.workspace._fs import detect_filesystem_type, is_network_fs
from acceleread.workspace.cache import JudgmentCache
from acceleread.workspace.catalog import (
    DEFAULT_LEASE_TTL,
    FINISHED_STATES,
    Catalog,
    JobInfo,
    JobKind,
    JobPrunedError,
    JobRunningError,
    JobState,
)
from acceleread.workspace.inputs import InputChangedError, InputRef, JobInputs
from acceleread.workspace.jobstore import DocumentState, JobStore, StorageVersionError

__all__ = [
    "DocumentState",
    "InputChangedError",
    "InputRef",
    "JobInfo",
    "JobInputs",
    "JobKind",
    "JobPrunedError",
    "JobRunningError",
    "JobState",
    "JobStore",
    "JudgmentCache",
    "NetworkFilesystemError",
    "StorageVersionError",
    "Workspace",
    "resolve_workspace_path",
]


class NetworkFilesystemError(RuntimeError):
    """The Workspace sits on NFS or SMB, where SQLite WAL is unsafe."""


def resolve_workspace_path(explicit: Path | None = None) -> Path:
    """`--workspace`, else `$ACCELEREAD_HOME`, else `~/.acceleread`."""
    if explicit is not None:
        return explicit
    if env := os.environ.get("ACCELEREAD_HOME"):
        return Path(env)
    return Path.home() / ".acceleread"


class Workspace:
    def __init__(self, path: Path, *, clock: Callable[[], float] = time.time) -> None:
        self.path = path
        self.jobs_dir = path / "jobs"
        opened: list[sqlite3.Connection] = []
        try:
            opened.append(_db.connect(path / "catalog.sqlite"))
            self._catalog = Catalog(opened[0], clock)
            opened.append(_db.connect(path / "cache.sqlite"))
            self.cache = JudgmentCache(opened[1], clock)
        except BaseException:
            for db in opened:
                db.close()
            raise

    @classmethod
    def open(
        cls,
        path: Path | None = None,
        *,
        allow_network_fs: bool = False,
        filesystem_type: Callable[[Path], str] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> Self:
        root = resolve_workspace_path(path)
        probe = root
        while not probe.exists():  # check the nearest existing ancestor before creating anything
            probe = probe.parent
        fs_type = (filesystem_type or detect_filesystem_type)(probe)
        if is_network_fs(fs_type) and not allow_network_fs:
            raise NetworkFilesystemError(
                f"Workspace {root} is on a network filesystem ({fs_type}), where SQLite WAL is "
                "unsafe. Pass --allow-network-fs to use it anyway."
            )
        (root / "jobs").mkdir(parents=True, exist_ok=True)
        (root / "models").mkdir(exist_ok=True)
        return cls(root, clock=clock)

    def close(self) -> None:
        self._catalog.close()
        self.cache.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # Jobs

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def create_job(self, manifest: dict[str, Any], *, kind: JobKind = "job") -> JobInfo:
        """Create a queued Job: its directory, `job.sqlite`, `manifest.json` and `inputs/`."""
        job_id = f"{int(self._catalog.now()):x}-{uuid.uuid4().hex[:10]}"
        directory = self.job_dir(job_id)
        (directory / "inputs").mkdir(parents=True)
        JobStore.create(directory, job_id=job_id).close()
        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        return self._catalog.add_job(job_id, kind)

    def get_job(self, job_id: str) -> JobInfo:
        return self._catalog.get(job_id)

    def list_jobs(self, *, include_ingest: bool = False) -> list[JobInfo]:
        return self._catalog.list_jobs(include_ingest=include_ingest)

    def next_queued(self) -> str | None:
        """The oldest queued Job (FIFO)."""
        return self._catalog.next_queued()

    def set_job_state(self, job_id: str, state: JobState) -> None:
        self._catalog.set_state(job_id, state)

    def finish_job(
        self, job_id: str, state: JobState, summary: dict[str, Any] | None = None
    ) -> None:
        """End a Job: freeze `summary.json` (if given), then record the terminal state."""
        if summary is not None:
            (self.job_dir(job_id) / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
            )
        self._catalog.set_state(job_id, state)

    def job_inputs(self, job_id: str) -> JobInputs:
        return JobInputs(self.job_dir(job_id) / "inputs")

    def open_job(self, job_id: str) -> JobStore:
        """The Job's store for running, resuming or retrying. Refused once pruned."""
        if self._catalog.get(job_id).pruned:
            raise JobPrunedError(f"Job {job_id} was pruned: it can be exported but not retried")
        return JobStore.open_for_run(self.job_dir(job_id))

    def read_job(self, job_id: str) -> JobStore:
        """A read-only view of the Job, for export and the UI."""
        self._catalog.get(job_id)
        return JobStore.reader(self.job_dir(job_id))

    # Runner lease

    def acquire_lease(self, holder: str, ttl: float = DEFAULT_LEASE_TTL) -> bool:
        return self._catalog.acquire_lease(holder, ttl)

    def heartbeat(self, holder: str) -> bool:
        return self._catalog.heartbeat(holder)

    def release_lease(self, holder: str) -> None:
        self._catalog.release_lease(holder)

    def lease_holder(self) -> str | None:
        return self._catalog.lease_holder()

    # Retention

    def delete_job(self, job_id: str) -> None:
        """Remove the Job's directory and catalog entry. Refused while it runs.

        The running check, the file removal and the row removal share one transaction; the row
        goes only once the directory has.
        """
        directory = self.job_dir(job_id)

        def remove_files() -> None:
            if directory.exists():
                shutil.rmtree(directory)

        self._catalog.remove(job_id, before=remove_files)

    def prune_jobs(
        self,
        *,
        older_than: timedelta,
        keep_records: bool = False,
        kind: JobKind | None = None,
    ) -> list[str]:
        """Free disk from finished Jobs older than `older_than`; returns the Job ids pruned.

        Pruning removes `inputs/` and, unless `keep_records`, the Records' text, then leaves
        the Job exportable but no longer retryable. The Job is marked pruned first, so a crash
        part-way never leaves an unmarked Job without inputs.
        """
        cutoff = self._catalog.now() - older_than.total_seconds()
        pruned: list[str] = []
        for job in self._catalog.list_jobs(include_ingest=True):
            if job.state not in FINISHED_STATES or job.finished_at is None:
                continue
            if job.finished_at > cutoff or (kind is not None and job.kind != kind):
                continue
            changed = self._catalog.mark_pruned(job.id)
            directory = self.job_dir(job.id)
            if (directory / "inputs").exists():
                shutil.rmtree(directory / "inputs")
                changed = True
            if not keep_records:
                with JobStore.open_for_maintenance(directory) as store:
                    changed = store.drop_text() > 0 or changed
            if changed:
                pruned.append(job.id)
        return pruned
