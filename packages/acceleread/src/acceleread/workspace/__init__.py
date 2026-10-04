# SPDX-License-Identifier: Apache-2.0
"""The Workspace: where one machine keeps its Jobs, Records, inputs and Judgment cache (§8)."""

import json
import os
import shutil
import time
import uuid
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from acceleread.workspace import _db, jobstore
from acceleread.workspace._fs import detect_filesystem_type, is_network_fs
from acceleread.workspace.cache import JudgmentCache
from acceleread.workspace.catalog import (
    DEFAULT_LEASE_TTL,
    FINISHED_STATES,
    Catalog,
    JobInfo,
    JobKind,
    JobRunningError,
    JobState,
)
from acceleread.workspace.inputs import InputChangedError, InputRef, JobInputs
from acceleread.workspace.jobstore import JobStore, StorageVersionError

__all__ = [
    "InputChangedError",
    "InputRef",
    "JobInfo",
    "JobInputs",
    "JobRunningError",
    "JobStore",
    "JudgmentCache",
    "NetworkFilesystemError",
    "StorageVersionError",
    "Workspace",
    "jobstore",
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
        self._catalog = Catalog(_db.connect(path / "catalog.sqlite"), clock)
        self.cache = JudgmentCache(_db.connect(path / "cache.sqlite"), clock)

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
        job_id = f"{int(time.time()):x}-{uuid.uuid4().hex[:10]}"
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
            raise PermissionError(f"Job {job_id} was pruned: it can be exported but not retried")
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
        """Remove the Job's directory and catalog entry. Refused while it runs."""
        if self._catalog.get(job_id).state == "running":
            raise JobRunningError(f"Job {job_id} is running")
        shutil.rmtree(self.job_dir(job_id), ignore_errors=True)
        self._catalog.remove(job_id)

    def prune_jobs(
        self,
        *,
        older_than: timedelta,
        keep_records: bool = False,
        kind: JobKind | None = None,
    ) -> list[str]:
        """Free disk from finished Jobs older than `older_than`; returns the Job ids pruned.

        By default the whole Job goes. With `keep_records`, only `inputs/` is removed: the Job
        stays listable and exportable but can no longer be retried.
        """
        cutoff = self._catalog.now() - older_than.total_seconds()
        pruned: list[str] = []
        for job in self._catalog.list_jobs(include_ingest=True):
            if job.state not in FINISHED_STATES or job.finished_at is None:
                continue
            if job.finished_at > cutoff or (kind is not None and job.kind != kind):
                continue
            if keep_records:
                if job.pruned:
                    continue
                shutil.rmtree(self.job_dir(job.id) / "inputs", ignore_errors=True)
                self._catalog.mark_pruned(job.id)
            else:
                self.delete_job(job.id)
            pruned.append(job.id)
        return pruned
