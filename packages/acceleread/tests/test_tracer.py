# SPDX-License-Identifier: Apache-2.0
"""Tracer: PDF → Record through Jev, replayed from a recorded cassette (no network)."""

import json
import os
from pathlib import Path

import httpx2
import pytest
import typesafe_sdk as ts

from acceleread import JobSpec, Taxonomy, run
from acceleread.classifier import Capabilities
from acceleread.cli import main
from acceleread.extract import extract_pdf
from acceleread.jev import JevClassifier
from acceleread.planner import DocumentView, judgment_specs, plan_requests

TESTS = Path(__file__).parent
SAMPLE = TESTS / "fixtures" / "sample.pdf"
TAXONOMY = TESTS / "fixtures" / "taxonomy.yaml"
CASSETTE = json.loads((TESTS / "cassettes" / "jev_sector.json").read_text())


def replay_classifier(seen: list[dict[str, object]]) -> JevClassifier:
    """A JevClassifier whose HTTP calls are answered from the cassette, in order."""
    exchanges = iter(CASSETTE)

    def handler(request: httpx2.Request) -> httpx2.Response:
        exchange = next(exchanges)
        seen.append(json.loads(request.content))
        assert request.url.path == exchange["request"]["path"]
        response = exchange["response"]
        return httpx2.Response(response["status"], json=response["json"])

    client = ts.AsyncTypeSafeClient(api_key="test-key", transport=httpx2.MockTransport(handler))
    return JevClassifier(client=client)


def test_extract_pages_are_offsets_into_text() -> None:
    doc = extract_pdf(SAMPLE)
    assert [p.number for p in doc.pages] == [1, 2]
    assert doc.text[doc.pages[0].start : doc.pages[0].end].startswith("NORTHWIND SOLAR")
    assert doc.text[doc.pages[1].start : doc.pages[1].end].startswith("Item 7.")
    assert doc.title == "Northwind Solar Annual Report"


def test_taxonomy_adds_other_and_hashes_stably() -> None:
    taxonomy = Taxonomy.from_file(TAXONOMY).with_other()
    assert [c.name for c in taxonomy.categories][-1] == "other"
    assert taxonomy.with_other() == taxonomy
    assert taxonomy.hash == Taxonomy.from_file(TAXONOMY).with_other().hash


def test_plan_cuts_head_and_tail_when_over_budget() -> None:
    doc = extract_pdf(SAMPLE)
    tiny = Capabilities(
        kinds=frozenset({"choice"}), max_choice_options=255, token_budget=150, chars_per_token=3.0
    )
    specs = judgment_specs(Taxonomy.from_file(TAXONOMY).with_other(), [])
    view = DocumentView(text=doc.text, pages=doc.pages, title="t")
    (request,), _ = plan_requests(view, specs, tiny)
    text = request.state["document"]["text"]  # type: ignore[index]
    assert request.coverage.truncated
    assert text.startswith("NORTHWIND") and text.endswith("next year.")
    assert "[…]" in text and len(text) < len(doc.text)


async def test_run_produces_record_from_cassette() -> None:
    seen: list[dict[str, object]] = []
    spec = JobSpec(inputs=[SAMPLE], taxonomy=Taxonomy.from_file(TAXONOMY))
    records = [r async for r in run(spec, replay_classifier(seen))]

    assert seen == [CASSETTE[0]["request"]["json"]]  # the request we send hasn't drifted
    (record,) = records
    assert record.status == "ok" and not record.errors
    assert record.source.format == "pdf" and record.source.bytes == SAMPLE.stat().st_size
    assert record.classification is not None
    assert record.classification.value == "energy"
    assert record.classification.confidence == 1.0
    assert record.classification.coverage.page_ranges == [(1, 2)]
    assert record.classification.coverage.input_tokens == record.usage.input_tokens > 0
    assert record.taxonomy is not None and record.taxonomy.name == "sector"


def test_cli_run_writes_jsonl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("acceleread.cli.make_classifier", lambda model: replay_classifier([]))
    out = tmp_path / "out.jsonl"
    assert main(["run", str(SAMPLE), "--taxonomy", str(TAXONOMY), "-o", str(out)]) == 0
    (line,) = out.read_text().splitlines()
    assert json.loads(line)["classification"]["value"] == "energy"


async def test_a_classifier_422_cancels_the_job_and_the_record_keeps_its_text() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(422, json={"error": {"message": "bad question"}})

    client = ts.AsyncTypeSafeClient(
        api_key="test-key",
        transport=httpx2.MockTransport(handler),
        retry=ts.RetryPolicy(max_retries=0),
    )
    spec = JobSpec(inputs=[SAMPLE], taxonomy=Taxonomy.from_file(TAXONOMY))
    (record,) = [r async for r in run(spec, JevClassifier(client=client))]
    assert record.status == "cancelled"  # a 422 auto-cancels the Job (spec §7.5)
    assert record.text and record.classification is None
    assert record.errors[0].stage == "classify"
    assert record.errors[0].code == "classifier_rejected"


async def test_unreadable_file_yields_failed_record(tmp_path: Path) -> None:
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"not a pdf")
    spec = JobSpec(inputs=[bad], taxonomy=Taxonomy.from_file(TAXONOMY))
    (record,) = [r async for r in run(spec, replay_classifier([]))]
    assert record.status == "failed" and record.errors[0].stage == "extract"


@pytest.mark.skipif(not os.environ.get("ACCELEREAD_LIVE_JEV"), reason="set ACCELEREAD_LIVE_JEV=1")
async def test_live_jev() -> None:
    spec = JobSpec(inputs=[SAMPLE], taxonomy=Taxonomy.from_file(TAXONOMY))
    (record,) = [r async for r in run(spec)]
    assert record.status == "ok"
    assert record.classification is not None and record.classification.value == "energy"
