# SPDX-License-Identifier: Apache-2.0
"""Job control (docs/spec/v0.md §7.1, §7.3, §11): cancel, resume, retry, attempts, the spend
cap, the immutable manifest and the Job summary."""

import asyncio
import json
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import pytest

from acceleread.classifier import ClassifierUnavailable
from acceleread.jobcontrol import (
    JobFinishedError,
    ManifestImmutableError,
    cancel_job,
    job_summary,
    load_manifest,
    resume_job,
    retry_failed,
    update_manifest,
)
from acceleread.models import DocumentRecord, JobSpec, Taxonomy
from acceleread.pipeline import extract
from acceleread.ratelimit import RateLimit, RateLimitedClassifier
from acceleread.runner import Runner
from acceleread.workers import Outcome, WorkerSettings
from acceleread.workspace import JobRunningError, Workspace

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = FIXTURES / "sample.pdf"
TAXONOMY = Taxonomy.from_file(FIXTURES / "taxonomy.yaml")
Fake = Callable[..., Any]


@pytest.fixture
def ws(tmp_path: Path) -> Iterator[Workspace]:
    with Workspace.open(tmp_path / "ws") as workspace:
        yield workspace


def copies(tmp_path: Path, count: int) -> list[Path]:
    paths = []
    for i in range(count):
        target = tmp_path / f"doc{i}.pdf"
        target.write_bytes(SAMPLE.read_bytes())
        paths.append(target)
    return paths


def records_of(ws: Workspace, job_id: str) -> list[DocumentRecord]:
    with ws.read_job(job_id) as store:
        found = [store.get_record(d) for d in store.states()]
    return [r for r in found if r is not None]


def states_of(ws: Workspace, job_id: str) -> list[str]:
    with ws.read_job(job_id) as store:
        return list(store.states().values())


def runner_for(ws: Workspace, classifier: Any, **kw: Any) -> Runner:
    kw.setdefault("poll_interval", 0.02)
    return Runner(ws, classifier, **kw)


class Extractions:
    """Counts the files extracted, so a test can prove OCR was not redone. Passed to the Runner
    as its `extractor`: a public seam, no patching."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    async def __call__(self, task: Any, pool: Any) -> Any:
        self.seen.append(task.path.name)
        return await extract(task, pool)

    def __len__(self) -> int:
        return len(self.seen)


@pytest.fixture
def extractions() -> Extractions:
    return Extractions()


def spec(paths: list[Path], **kw: Any) -> JobSpec:
    return JobSpec(inputs=paths, taxonomy=TAXONOMY, **kw)


# Cancel


async def test_cancelling_a_queued_job_gives_every_document_a_cancelled_record(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake
):
    job_id = runner_for(ws, fake_classifier()).submit(spec(copies(tmp_path, 2)))

    assert cancel_job(ws, job_id) == "cancelled"

    assert ws.get_job(job_id).state == "cancelled"
    assert [r.status for r in records_of(ws, job_id)] == ["cancelled", "cancelled"]
    assert states_of(ws, job_id) == ["cancelled", "cancelled"]
    summary = json.loads((ws.job_dir(job_id) / "summary.json").read_text())
    assert summary["flags"]["cancel_reason"] == "user"


async def test_cancelling_a_running_job_stops_it_and_keeps_extracted_work(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake
):
    gate = asyncio.Event()
    classifier = fake_classifier(gate=gate)
    runner = runner_for(ws, classifier, classify_concurrency=1, cancel_grace=0.05)
    assert runner.acquire()  # a live runner holds the lease
    job_id = runner.submit(spec(copies(tmp_path, 3)))
    task = asyncio.create_task(runner.execute(job_id))
    while not classifier.calls:
        await asyncio.sleep(0.02)

    assert cancel_job(ws, job_id) == "requested"  # a live runner finishes the cancel itself
    assert await task == "cancelled"

    records = records_of(ws, job_id)
    assert [r.status for r in records] == ["cancelled"] * 3
    assert records[0].text  # extracted before the cancel, and kept
    assert ws.get_job(job_id).state == "cancelled"


async def test_cancelling_a_finished_job_is_refused(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake
):
    runner = runner_for(ws, fake_classifier())
    job_id = runner.submit(spec(copies(tmp_path, 1)))
    await runner.execute(job_id)
    with pytest.raises(JobFinishedError):
        cancel_job(ws, job_id)


# Resume


async def test_resume_requeues_cancelled_documents_without_redoing_extraction(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake, extractions: Extractions
):
    gate = asyncio.Event()
    classifier = fake_classifier(gate=gate)
    runner = runner_for(
        ws, classifier, classify_concurrency=1, cancel_grace=0.05, extractor=extractions
    )
    assert runner.acquire()  # a live runner holds the lease
    job_id = runner.submit(spec(copies(tmp_path, 3)))
    task = asyncio.create_task(runner.execute(job_id))
    while not classifier.calls:
        await asyncio.sleep(0.02)
    cancel_job(ws, job_id)
    await task
    runner.release()  # nobody holds the lease now, so resume writes directly

    assert resume_job(ws, job_id).count == 3
    assert ws.get_job(job_id).state == "queued"
    gate.set()
    assert await runner.execute(job_id) == "done"

    assert [r.status for r in records_of(ws, job_id)] == ["ok"] * 3
    # Only Documents never extracted were extracted again; the extracted ones were not.
    assert len(extractions) == 3
    assert len(set(extractions.seen)) == 3


async def test_resume_leaves_a_finished_job_alone(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake
):
    runner = runner_for(ws, fake_classifier())
    job_id = runner.submit(spec(copies(tmp_path, 1)))
    await runner.execute(job_id)
    assert resume_job(ws, job_id).count == 0
    assert ws.get_job(job_id).state == "done"


async def test_resume_is_refused_while_the_job_runs(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake
):
    gate = asyncio.Event()
    classifier = fake_classifier(gate=gate)
    runner = runner_for(ws, classifier)
    assert runner.acquire()
    job_id = runner.submit(spec(copies(tmp_path, 1)))
    task = asyncio.create_task(runner.execute(job_id))
    while not classifier.calls:
        await asyncio.sleep(0.02)
    with pytest.raises(JobRunningError):
        resume_job(ws, job_id)
    gate.set()
    await task


# Retry


async def test_retry_failed_replaces_the_record_and_counts_attempts(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake, extractions: Extractions
):
    broken = fake_classifier(raises=lambda _: ClassifierUnavailable("503"))
    runner = runner_for(
        ws,
        RateLimitedClassifier(broken, RateLimit(1e9, 1e6), max_unavailable_tries=1),
        extractor=extractions,
    )
    job_id = runner.submit(spec(copies(tmp_path, 2)))
    await runner.execute(job_id)
    assert states_of(ws, job_id) == ["failed", "failed"]
    assert [r.attempts for r in records_of(ws, job_id)] == [1, 1]

    assert retry_failed(ws, job_id).count == 2
    assert ws.get_job(job_id).state == "queued"
    healthy = runner_for(ws, fake_classifier(), extractor=extractions)
    assert await healthy.execute(job_id) == "done"

    records = records_of(ws, job_id)
    assert [r.status for r in records] == ["ok", "ok"]
    assert [r.attempts for r in records] == [2, 2]
    assert [r.errors for r in records] == [[], []]
    assert len(extractions) == 2  # the retry reused the stored text: no OCR again


async def test_retry_re_extracts_a_document_whose_input_was_missing(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake
):
    (path,) = copies(tmp_path, 1)
    original = path.read_bytes()
    runner = runner_for(ws, fake_classifier())
    job_id = runner.submit(spec([path]))
    path.unlink()
    await runner.execute(job_id)
    assert states_of(ws, job_id) == ["failed"]

    path.write_bytes(original)  # the same file is back, so the recorded hash matches again
    retry_failed(ws, job_id)
    await runner.execute(job_id)

    (record,) = records_of(ws, job_id)
    assert record.status == "ok" and record.attempts == 2 and record.text


# Spend cap and the manifest


async def test_reaching_the_spend_cap_auto_cancels_and_a_raised_cap_lets_resume_finish(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake
):
    # 1M input tokens at Jev's $0.042 per 1M is $0.042 a Document.
    classifier = fake_classifier(model="jev-fake", input_tokens=1_000_000)
    runner = runner_for(ws, classifier, classify_concurrency=1, cancel_grace=0.05)
    job_id = runner.submit(spec(copies(tmp_path, 5), max_cost_usd=0.05, cache=False))

    assert await runner.execute(job_id) == "cancelled"

    assert sorted(r.status for r in records_of(ws, job_id)) == ["cancelled"] * 3 + ["ok"] * 2
    summary = json.loads((ws.job_dir(job_id) / "summary.json").read_text())
    assert summary["flags"]["cancel_reason"] == "spend_cap"
    assert summary["classifier"]["estimated_cost_usd"] == pytest.approx(0.084)

    update_manifest(ws, job_id, {"max_cost_usd": 1.0})
    assert resume_job(ws, job_id).count == 3
    assert await runner.execute(job_id) == "done"
    assert [r.status for r in records_of(ws, job_id)] == ["ok"] * 5


async def test_the_manifest_is_immutable_except_max_cost_usd(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake
):
    job_id = runner_for(ws, fake_classifier()).submit(spec(copies(tmp_path, 1)))
    with pytest.raises(ManifestImmutableError, match="new Job"):
        update_manifest(ws, job_id, {"model": "other"})
    with pytest.raises(ManifestImmutableError):
        update_manifest(ws, job_id, {"max_cost_usd": 2.0, "cache": False})
    assert load_manifest(ws, job_id).max_cost_usd is None

    update_manifest(ws, job_id, {"max_cost_usd": 2.0})
    assert load_manifest(ws, job_id).max_cost_usd == 2.0
    update_manifest(ws, job_id, {"max_cost_usd": None})
    assert load_manifest(ws, job_id).max_cost_usd is None


# Summary


async def test_the_summary_reports_progress_usage_extraction_and_failures(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake
):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf")
    runner = runner_for(ws, fake_classifier(model="jev-fake", input_tokens=500_000))
    job_id = runner.submit(spec([broken, *copies(tmp_path, 2)]))
    await runner.execute(job_id)

    summary = job_summary(ws, job_id)
    assert summary["progress"]["documents"] == 3
    assert summary["progress"]["by_state"] == {"done": 2, "failed": 1}
    # The two copies are identical, so the second is a cache hit and costs nothing.
    assert summary["classifier"]["requests"] == 1
    assert summary["classifier"]["cache_hits"] == 1
    assert summary["classifier"]["input_tokens"] == 500_000
    assert summary["classifier"]["estimated_cost_usd"] == pytest.approx(0.021)
    assert summary["classifier"]["cost_is_estimated"] is True
    assert summary["extraction"]["pages"] == 4
    failure = summary["failures"]["PdfiumError"]
    assert failure["count"] == 1 and failure["examples"] == ["000000"]
    assert summary["flags"] == {"stalled": False, "cancel_reason": None, "error": None}
    frozen = json.loads((ws.job_dir(job_id) / "summary.json").read_text())
    assert frozen["classifier"]["ceiling"]["requests_per_s"] > 0
    assert frozen["state"] == "done"


class CrashingPool:
    """A stand-in worker pool whose workers keep crashing, so it cancels itself `crash_rate`."""

    size = 1
    cancel_reason: str | None = None

    def submit(self, handler: str, payload: Any) -> "Future[Outcome]":
        self.cancel_reason = "crash_rate"
        future: Future[Outcome] = Future()
        future.set_result(Outcome(False, code="cancelled"))
        return future

    def cancel(self, reason: str) -> None: ...

    def close(self) -> None: ...


class CrashingSettings(WorkerSettings):
    def pool(self, profile: str = "fast") -> Any:  # type: ignore[override]
        return CrashingPool()


async def test_a_pool_crash_rate_cancel_cancels_the_job_with_that_reason(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake
):
    runner = runner_for(ws, fake_classifier(), workers=CrashingSettings())
    job_id = runner.submit(spec(copies(tmp_path, 3)))

    assert await runner.execute(job_id) == "cancelled"

    assert [r.status for r in records_of(ws, job_id)] == ["cancelled"] * 3
    summary = json.loads((ws.job_dir(job_id) / "summary.json").read_text())
    assert summary["flags"]["cancel_reason"] == "crash_rate"
