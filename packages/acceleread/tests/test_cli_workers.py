# SPDX-License-Identifier: Apache-2.0
"""`acceleread run --ocr-workers/--threads-per-worker` (docs/spec/v0.md §7.4): extraction goes
through the worker pool these flags size."""

import json
from pathlib import Path

import pytest
import typesafe_sdk as ts

from acceleread import JobSpec, Taxonomy, run
from acceleread.cli import main
from acceleread.jev import JevClassifier
from acceleread.workers import WorkerSettings

FIXTURES = Path(__file__).parent / "fixtures"


def idle_classifier() -> JevClassifier:
    return JevClassifier(client=ts.AsyncTypeSafeClient(api_key="unused"))


def test_run_accepts_the_worker_flags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[WorkerSettings | None] = []

    async def fake_run(
        spec: JobSpec, classifier: object = None, workers: WorkerSettings | None = None
    ):  # type: ignore[no-untyped-def]
        seen.append(workers)
        return
        yield

    monkeypatch.setattr("acceleread.cli.run", fake_run)
    monkeypatch.setattr("acceleread.cli.make_classifier", lambda model: idle_classifier())
    out = tmp_path / "out.jsonl"
    args = ["run", str(FIXTURES / "sample.pdf"), "--taxonomy", str(FIXTURES / "taxonomy.yaml")]
    main([*args, "--ocr-workers", "3", "--threads-per-worker", "2", "-o", str(out)])
    main([*args, "-o", str(out)])
    assert seen == [WorkerSettings(ocr_workers=3, threads_per_worker=2), WorkerSettings()]


async def test_documents_are_extracted_in_the_pool_when_settings_are_given(
    tmp_path: Path,
) -> None:
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf")
    spec = JobSpec(inputs=[broken], taxonomy=Taxonomy.from_file(FIXTURES / "taxonomy.yaml"))
    (record,) = [
        r async for r in run(spec, idle_classifier(), workers=WorkerSettings(ocr_workers=1))
    ]
    assert record.status == "failed"
    assert record.errors[0].stage == "extract"
    assert record.errors[0].code == "handler_error"
    json.loads(record.model_dump_json())
