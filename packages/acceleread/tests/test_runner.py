# SPDX-License-Identifier: Apache-2.0
"""The runner pipeline (docs/spec/v0.md §7.3, §7.5): extract, persist `extracted`, classify, and
the wiring of the Workspace, the rate limiter, the Judgment cache and Section detection."""

import asyncio
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from acceleread.classifier import ClassifierRejected, ClassifierUnavailable
from acceleread.languages import workspace_tessdata
from acceleread.models import DocumentRecord, JobSpec, Question, Taxonomy
from acceleread.pipeline import extract, make_extract_task
from acceleread.ratelimit import RateLimit, RateLimitedClassifier
from acceleread.runner import Runner
from acceleread.validate import SpecError
from acceleread.workers import WorkerSettings
from acceleread.workspace import Workspace

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = FIXTURES / "sample.pdf"
FILING = FIXTURES / "filing10k.pdf"
TAXONOMY = Taxonomy.from_file(FIXTURES / "taxonomy.yaml")
Fake = Callable[..., Any]


@pytest.fixture
def ws(tmp_path: Path) -> Iterator[Workspace]:
    with Workspace.open(tmp_path / "ws") as workspace:
        yield workspace


def copies(tmp_path: Path, count: int, source: Path = SAMPLE) -> list[Path]:
    paths = []
    for i in range(count):
        target = tmp_path / f"doc{i}.pdf"
        shutil.copy(source, target)
        paths.append(target)
    return paths


def records_of(ws: Workspace, job_id: str) -> list[DocumentRecord]:
    with ws.read_job(job_id) as store:
        found = [store.get_record(d) for d in store.states()]
    return [r for r in found if r is not None]


def runner_for(ws: Workspace, classifier: Any, **kw: Any) -> Runner:
    kw.setdefault("poll_interval", 0.02)
    return Runner(ws, classifier, **kw)


async def settle(seconds: float = 0.3) -> None:
    await asyncio.sleep(seconds)


async def test_a_job_runs_from_queued_to_done(ws: Workspace, tmp_path: Path, fake_classifier: Fake):
    runner = runner_for(ws, fake_classifier())
    job_id = runner.submit(JobSpec(inputs=copies(tmp_path, 2), taxonomy=TAXONOMY))
    assert ws.get_job(job_id).state == "queued"

    assert await runner.execute(job_id) == "done"

    assert ws.get_job(job_id).state == "done"
    with ws.read_job(job_id) as store:
        assert set(store.states().values()) == {"done"}
    first, second = records_of(ws, job_id)
    assert first.status == "ok" and second.status == "ok"
    assert first.classification is not None and first.classification.value == "energy"
    assert first.text and first.pages and first.source.filename == "doc0.pdf"
    assert (ws.job_dir(job_id) / "manifest.json").is_file()
    assert (ws.job_dir(job_id) / "summary.json").is_file()


async def test_the_extracted_record_is_persisted_before_the_classifier_answers(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake
):
    gate = asyncio.Event()
    classifier = fake_classifier(gate=gate)
    runner = runner_for(ws, classifier)
    job_id = runner.submit(JobSpec(inputs=copies(tmp_path, 1), taxonomy=TAXONOMY))
    task = asyncio.create_task(runner.execute(job_id))
    while not classifier.calls:
        await settle(0.02)

    with ws.read_job(job_id) as store:
        assert list(store.states().values()) == ["classifying"]
        (record,) = [store.get_record(d) for d in store.states()]
    assert record is not None and record.text and record.pages
    assert record.classification is None  # extraction is saved, the Judgment is not yet

    gate.set()
    assert await task == "done"


async def test_the_classifier_is_wrapped_in_one_rate_limiter_with_the_configured_ceiling(
    ws: Workspace, fake_classifier: Fake
):
    (ws.path / "config.yaml").write_text(
        "classifier:\n  rate_limit:\n    tokens_per_s: 1234\n    requests_per_s: 5\n"
    )
    runner = Runner(ws, fake_classifier())
    assert isinstance(runner.classifier, RateLimitedClassifier)
    assert runner.classifier.ceiling == RateLimit(1234, 5)


async def test_jev_defaults_to_eighty_percent_of_the_published_limits(ws: Workspace):
    from acceleread.jev import JevClassifier

    runner = Runner(ws, JevClassifier())
    assert runner.classifier.ceiling == RateLimit(64_000, 51.2)


async def test_the_judgment_cache_is_shared_across_jobs_and_no_cache_bypasses_it(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake
):
    classifier = fake_classifier()
    runner = runner_for(ws, classifier)
    paths = copies(tmp_path, 1)

    first = runner.submit(JobSpec(inputs=paths, taxonomy=TAXONOMY))
    await runner.execute(first)
    assert len(classifier.calls) == 1

    second = runner.submit(JobSpec(inputs=paths, taxonomy=TAXONOMY))
    await runner.execute(second)
    assert len(classifier.calls) == 1  # answered from the Workspace cache
    (record,) = records_of(ws, second)
    assert record.usage.cache_hits == 1 and record.usage.requests == 0

    third = runner.submit(JobSpec(inputs=paths, taxonomy=TAXONOMY, cache=False))
    await runner.execute(third)
    assert len(classifier.calls) == 2


async def test_questions_that_read_sections_get_only_those_sections(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake
):
    classifier = fake_classifier()
    runner = runner_for(ws, classifier)
    spec = JobSpec(
        inputs=[FILING, SAMPLE],
        questions=[
            Question(
                name="supply",
                kind="noul",
                instructions="Does `document` warn of supply problems?",
                reads=["risk_factors"],
            )
        ],
    )
    job_id = runner.submit(spec)
    await runner.execute(job_id)

    filing, sample = records_of(ws, job_id)
    assert {s.keys[0] for s in filing.sections} == {"business", "risk_factors", "mdna"}
    (state, names) = classifier.calls[0]
    assert names == ["q_supply"]
    sections = state["document"]["sections"]  # type: ignore[index, call-overload]
    assert list(sections) == ["risk_factors"] and "Polysilicon" in sections["risk_factors"]
    answer = filing.answers["supply"]
    assert answer.coverage.sections == ["risk_factors"]  # type: ignore[union-attr]
    # sample.pdf has no cover page naming a form, so no Sections: the Question is skipped.
    assert sample.sections == []
    assert sample.answers["supply"].value is None  # type: ignore[union-attr]
    assert len(classifier.calls) == 1  # nothing was sent for the skipped Document


async def test_a_job_may_hold_only_questions(ws: Workspace, tmp_path: Path, fake_classifier: Fake):
    runner = runner_for(ws, fake_classifier())
    spec = JobSpec(
        inputs=copies(tmp_path, 1),
        questions=[Question(name="q", kind="noul", instructions="Is `document` short?")],
    )
    job_id = runner.submit(spec)
    await runner.execute(job_id)
    (record,) = records_of(ws, job_id)
    assert record.status == "ok" and record.classification is None and "q" in record.answers


async def test_html_documents_are_ingested_too(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake
):
    page = tmp_path / "report.html"
    page.write_text(
        "<html><head><title>Annual</title></head><body><p>Solar panels.</p></body></html>"
    )
    runner = runner_for(ws, fake_classifier())
    job_id = runner.submit(JobSpec(inputs=[page], taxonomy=TAXONOMY))
    await runner.execute(job_id)
    (record,) = records_of(ws, job_id)
    assert record.source.format == "html" and "Solar panels." in (record.text or "")


async def test_globs_expand_to_one_document_per_file(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake
):
    copies(tmp_path, 3)
    runner = runner_for(ws, fake_classifier())
    job_id = runner.submit(JobSpec(inputs=[str(tmp_path / "doc*.pdf")], taxonomy=TAXONOMY))
    await runner.execute(job_id)
    assert [r.source.filename for r in records_of(ws, job_id)] == [
        "doc0.pdf",
        "doc1.pdf",
        "doc2.pdf",
    ]


# Language packs


def test_the_extract_task_carries_the_workspace_language_packs_and_the_documents_languages(
    tmp_path: Path,
):
    task = make_extract_task(tmp_path / "a.pdf", "pdf", ["en", "de"], tmp_path / "ws")
    assert task.ocr_languages == ("en", "de")
    assert task.tessdata == (workspace_tessdata(tmp_path / "ws"),)


async def test_add_language_makes_the_language_valid_for_jobs(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake
):
    runner = runner_for(ws, fake_classifier())
    spec = JobSpec(inputs=copies(tmp_path, 1), taxonomy=TAXONOMY, ocr_languages=["en", "de"])
    with pytest.raises(SpecError, match="add-language de"):
        runner.submit(spec)

    pack = workspace_tessdata(ws.path)
    pack.mkdir(parents=True)
    (pack / "deu.traineddata").write_bytes(b"pack")
    assert runner.submit(spec)


# Failure policy


async def test_a_classifier_rejection_auto_cancels_the_job(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake
):
    classifier = fake_classifier(raises=lambda _: ClassifierRejected("422"))
    runner = runner_for(ws, classifier, classify_concurrency=1)
    job_id = runner.submit(JobSpec(inputs=copies(tmp_path, 3), taxonomy=TAXONOMY))

    assert await runner.execute(job_id) == "cancelled"

    assert ws.get_job(job_id).state == "cancelled"
    assert len(classifier.calls) == 1  # the other Documents were never sent
    records = records_of(ws, job_id)
    assert [r.status for r in records] == ["cancelled"] * 3  # one Record per Document
    assert records[0].errors[0].code == "classifier_rejected"
    assert records[0].text  # extraction work is kept
    import json

    summary = json.loads((ws.job_dir(job_id) / "summary.json").read_text())
    assert summary["flags"]["cancel_reason"] == "classifier_rejected"


def fail_fast(classifier: Any) -> RateLimitedClassifier:
    """A rate limiter that gives up on 5xx at once instead of backing off for seconds."""
    return RateLimitedClassifier(classifier, RateLimit(1e9, 1e6), max_unavailable_tries=1)


async def test_a_persistently_unavailable_classifier_fails_the_document_not_the_job(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake
):
    classifier = fake_classifier(raises=lambda _: ClassifierUnavailable("503"))
    runner = runner_for(ws, fail_fast(classifier))
    job_id = runner.submit(JobSpec(inputs=copies(tmp_path, 2), taxonomy=TAXONOMY))

    assert await runner.execute(job_id) == "done"

    with ws.read_job(job_id) as store:
        assert set(store.states().values()) == {"failed"}
    (record, _) = records_of(ws, job_id)
    assert record.status == "failed"
    assert record.errors[0].code == "classifier_unavailable"
    assert record.errors[0].message == "classifier unavailable"
    assert record.text  # the Record keeps its extraction


async def test_a_document_that_cannot_be_extracted_fails_with_a_record(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake
):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf")
    runner = runner_for(ws, fake_classifier())
    job_id = runner.submit(JobSpec(inputs=[broken, *copies(tmp_path, 1)], taxonomy=TAXONOMY))
    assert await runner.execute(job_id) == "done"
    bad, good = records_of(ws, job_id)
    assert bad.status == "failed" and bad.errors[0].stage == "extract"
    assert good.status == "ok"


async def test_a_missing_or_changed_input_fails_that_document(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake
):
    (changing, stable) = copies(tmp_path, 2)
    runner = runner_for(ws, fake_classifier())
    job_id = runner.submit(
        JobSpec(inputs=[changing, stable, tmp_path / "gone.pdf"], taxonomy=TAXONOMY)
    )
    changing.write_bytes(changing.read_bytes() + b"\n%edited")

    await runner.execute(job_id)

    changed, ok, gone = records_of(ws, job_id)
    assert changed.errors[0].code == "input_changed" and changed.status == "failed"
    assert ok.status == "ok"
    assert gone.errors[0].code == "input_unavailable" and gone.status == "failed"


async def test_a_url_input_fails_until_the_api_fetches_urls(ws: Workspace, fake_classifier: Fake):
    runner = runner_for(ws, fake_classifier())
    job_id = runner.submit(JobSpec(inputs=["https://example.com/a.pdf"], taxonomy=TAXONOMY))
    await runner.execute(job_id)
    (record,) = records_of(ws, job_id)
    assert record.status == "failed" and record.errors[0].code == "unsupported_input"


# Backpressure and the worker pool


async def test_extraction_stops_while_too_many_documents_await_the_classifier(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake
):
    gate = asyncio.Event()
    classifier = fake_classifier(gate=gate)
    extracted: list[str] = []

    async def counting(task: Any, pool: Any) -> Any:
        extracted.append(task.path.name)
        return await extract(task, pool)

    runner = runner_for(ws, classifier, backpressure=2, classify_concurrency=1, extractor=counting)
    job_id = runner.submit(JobSpec(inputs=copies(tmp_path, 6), taxonomy=TAXONOMY))
    task = asyncio.create_task(runner.execute(job_id))
    while len(extracted) < 3 or not classifier.calls:  # 1 classifying + 2 awaiting, the limit
        await asyncio.sleep(0.005)
    for _ in range(200):  # give extraction every chance to run ahead, if it were allowed to
        await asyncio.sleep(0)

    with ws.read_job(job_id) as store:
        states = list(store.states().values())
    assert states.count("classifying") == 1
    assert states.count("extracted") == 2  # the limit
    assert states.count("queued") == 3  # not extracted yet
    assert len(extracted) == 3

    gate.set()
    assert await task == "done"
    with ws.read_job(job_id) as store:
        assert set(store.states().values()) == {"done"}


async def test_extraction_runs_in_the_worker_pool(
    ws: Workspace, tmp_path: Path, fake_classifier: Fake
):
    runner = runner_for(ws, fake_classifier(), workers=WorkerSettings(ocr_workers=1))
    job_id = runner.submit(JobSpec(inputs=copies(tmp_path, 2), taxonomy=TAXONOMY))
    assert await runner.execute(job_id) == "done"
    assert [r.status for r in records_of(ws, job_id)] == ["ok", "ok"]
