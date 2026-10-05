# SPDX-License-Identifier: Apache-2.0
"""The Planner: grouping, state, budget, Coverage and cache (docs/spec/v0.md §5.2-§5.3)."""

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import httpx2
import pytest
import typesafe_sdk as ts

from acceleread.classifier import (
    Ask,
    Capabilities,
    ClassifierResponse,
    ClassifierTokensExceeded,
    JSONState,
    JudgmentResult,
    Noul,
    estimate_tokens,
)
from acceleread.extract import extract_pdf
from acceleread.jev import JEV_CAPABILITIES, JevClassifier
from acceleread.models import (
    ClassifierInfo,
    Judgment,
    Page,
    Question,
    Section,
    SkippedAnswer,
    Span,
    Taxonomy,
    Verification,
)
from acceleread.planner import (
    DocumentView,
    judge_document,
    judgment_specs,
    plan_requests,
)
from acceleread.workspace import Workspace

CAPS = Capabilities(
    kinds=frozenset({"noul", "score", "choice"}),
    max_choice_options=255,
    token_budget=1_000,
    chars_per_token=3.0,
    model="m-1",
    classifier_id="fake",
)
TESTS = Path(__file__).parent
SAMPLE = TESTS / "fixtures" / "sample.pdf"
TAXONOMY_FILE = TESTS / "fixtures" / "taxonomy.yaml"
CASSETTE = json.loads((TESTS / "cassettes" / "jev_sector.json").read_text())
INFO = ClassifierInfo(id="fake", model="m-1", version="1")


def q(name: str, reads: list[str] | None = None, **kw: object) -> Question:
    return Question(
        name=name, kind="noul", instructions=f"Is {name} true of `document`?", reads=reads, **kw
    )  # type: ignore[arg-type]


TAXONOMY = Taxonomy(name="t", categories=[{"name": "a"}, {"name": "b"}]).with_other()  # type: ignore[list-item]


def section(key: str, start: int, end: int, **kw: object) -> Section:
    return Section(keys=[key], label=key, spans=[Span(start=start, end=end)], method="regex", **kw)  # type: ignore[arg-type]


class FakeClassifier:
    """Answers every Judgment with `yes`/its first choice; records every call."""

    def __init__(self, caps: Capabilities = CAPS, fail_sizes_over: int | None = None) -> None:
        self._caps = caps
        self.calls: list[tuple[JSONState, Mapping[str, Ask]]] = []
        self.fail_sizes_over = fail_sizes_over

    @property
    def capabilities(self) -> Capabilities:
        return self._caps

    async def judge(self, state: JSONState, judgments: Mapping[str, Ask]) -> ClassifierResponse:
        self.calls.append((state, judgments))
        doc = state["document"]
        assert isinstance(doc, dict)
        size = len(str(doc.get("text") or doc.get("sections")))
        if self.fail_sizes_over is not None and size > self.fail_sizes_over:
            raise ClassifierTokensExceeded("too big")
        return ClassifierResponse(
            results={
                name: JudgmentResult(kind="noul", value="yes")
                if isinstance(ask, Noul)
                else JudgmentResult(
                    kind="choice", value="a", probabilities={"a": 1.0}, confidence=1.0
                )
                for name, ask in judgments.items()
            },
            info=INFO,
            input_tokens=100,
            output_tokens=1,
        )


def test_a_short_documents_estimate_is_close_to_what_jev_billed() -> None:
    """Recorded: the 2-page fixture was billed 506 input tokens (it was estimated at 286)."""
    doc = extract_pdf(SAMPLE)
    specs = judgment_specs(Taxonomy.from_file(TAXONOMY_FILE).with_other(), [])
    view = DocumentView(text=doc.text, pages=doc.pages, title=doc.title or "sample")
    (request,), _ = plan_requests(view, specs, JEV_CAPABILITIES)
    billed = CASSETTE[0]["response"]["json"]["usage"]["input_tokens"]
    assert abs(request.coverage.est_tokens - billed) <= 0.15 * billed


# --- grouping -------------------------------------------------------------------------------

BIG = DocumentView(
    text="A" * 100 + "B" * 100 + "C" * 100,
    sections=[
        section("business", 0, 100),
        section("mdna", 100, 200),
        section("controls", 200, 300),
    ],
    title="Acme 10-K",
)


def test_one_request_per_distinct_reads_set_and_the_taxonomy_joins_its_group() -> None:
    specs = judgment_specs(
        TAXONOMY.model_copy(update={"reads": ["business"]}),
        [q("sector", ["business"]), q("outlook", ["mdna"]), q("whole")],
    )
    requests, skipped = plan_requests(BIG, specs, CAPS)
    groups = {r.reads: [s.name for s in r.specs] for r in requests}
    assert groups == {
        ("business",): ["taxonomy", "sector"],
        ("mdna",): ["outlook"],
        None: ["whole"],
    }
    assert not skipped


def test_reads_order_does_not_split_a_group() -> None:
    specs = judgment_specs(None, [q("a", ["mdna", "business"]), q("b", ["business", "mdna"])])
    requests, _ = plan_requests(BIG, specs, CAPS)
    assert len(requests) == 1


async def test_a_question_named_taxonomy_cannot_collide_with_the_taxonomy() -> None:
    specs = judgment_specs(TAXONOMY, [q("taxonomy")])
    fake = FakeClassifier()
    outcome = await judge_document(BIG, specs, fake)
    assert len(fake.calls) == 1
    assert len(fake.calls[0][1]) == 2  # both reached the Classifier, under different names
    assert outcome.classification is not None
    assert isinstance(outcome.answers["taxonomy"], Judgment)
    assert outcome.classification.kind == "choice" and outcome.answers["taxonomy"].kind == "noul"


# --- state ----------------------------------------------------------------------------------


def test_state_is_sections_by_key_or_whole_text_with_known_metadata_only() -> None:
    view = replace(BIG, company="Acme", form="10-K")
    requests, _ = plan_requests(view, judgment_specs(None, [q("s", ["mdna"]), q("w")]), CAPS)
    by_reads = {r.reads: r.state for r in requests}
    assert by_reads[("mdna",)] == {
        "document": {
            "title": "Acme 10-K",
            "company": "Acme",
            "form": "10-K",
            "sections": {"mdna": "B" * 100},
        }
    }
    assert by_reads[None] == {
        "document": {"title": "Acme 10-K", "company": "Acme", "form": "10-K", "text": BIG.text}
    }


def test_several_sections_with_one_key_are_joined() -> None:
    view = DocumentView(text="aaabbbccc", sections=[section("mdna", 0, 3), section("mdna", 6, 9)])
    (request,), _ = plan_requests(view, judgment_specs(None, [q("s", ["mdna"])]), CAPS)
    assert request.state["document"]["sections"] == {"mdna": "aaa\n\nccc"}  # type: ignore[index]


# --- budget ---------------------------------------------------------------------------------


def tiny(tokens: int) -> Capabilities:
    return replace(CAPS, token_budget=tokens)


PAGES = [Page(number=i + 1, start=i * 100, end=(i + 1) * 100) for i in range(10)]
LONG = DocumentView(text="".join(f"{i}" * 100 for i in range(10)), pages=PAGES, title="t")


def test_a_document_that_fits_is_sent_whole() -> None:
    (request,), _ = plan_requests(LONG, judgment_specs(None, [q("w")]), CAPS)
    assert not request.coverage.truncated
    assert request.coverage.spans == [Span(start=0, end=1000)]
    assert request.coverage.page_ranges == [(1, 10)]


def test_over_budget_cuts_head_and_tail_on_page_boundaries() -> None:
    caps = tiny(215)  # room for ~ 500 chars of text
    (request,), _ = plan_requests(LONG, judgment_specs(None, [q("w")]), caps)
    cov = request.coverage
    assert cov.truncated and not cov.shrunk
    head, tail = cov.spans
    assert head.start == 0 and head.end % 100 == 0  # ends on a Page end
    assert tail.end == 1000 and tail.start % 100 == 0  # starts on a Page start
    assert tail.end - tail.start > head.end - head.start  # the tail is the bigger share
    text = request.state["document"]["text"]  # type: ignore[index]
    assert "[…]" in text and len(text) < len(LONG.text)
    assert cov.est_tokens <= 215  # the whole request fits the budget
    # Whole Pages only: page_ranges does not overstate what was read.
    assert cov.page_ranges == [(1, head.end // 100), (tail.start // 100 + 1, 10)]


def test_a_cut_inside_one_huge_page_still_records_exact_spans() -> None:
    view = DocumentView(text="x" * 1000, pages=[Page(number=1, start=0, end=1000)])
    (request,), _ = plan_requests(view, judgment_specs(None, [q("w")]), tiny(215))
    covered = sum(s.end - s.start for s in request.coverage.spans)
    assert 0 < covered < 1000
    assert request.coverage.truncated


def test_a_groups_sections_are_cut_in_proportion() -> None:
    view = DocumentView(
        text="a" * 100 + "b" * 400,
        sections=[section("business", 0, 100), section("mdna", 100, 500)],
    )
    (request,), _ = plan_requests(
        view, judgment_specs(None, [q("s", ["business", "mdna"])]), tiny(190)
    )
    read = {s.start: s.end - s.start for s in request.coverage.spans}
    short = sum(n for start, n in read.items() if start < 100)
    long = sum(n for start, n in read.items() if start >= 100)
    assert request.coverage.truncated
    assert 3.5 < long / short < 4.5  # 400:100 in, so about 4:1 out


def test_the_per_request_overhead_reduces_the_room_for_text() -> None:
    plain, _ = plan_requests(LONG, judgment_specs(None, [q("w")]), tiny(400))
    costly, _ = plan_requests(
        LONG, judgment_specs(None, [q("w")]), replace(tiny(400), request_overhead_tokens=100)
    )
    kept = lambda r: sum(s.end - s.start for s in r[0].coverage.spans)  # noqa: E731
    assert kept(costly) < kept(plain)


async def test_max_tokens_exceeded_shrinks_by_15_percent_and_retries_once() -> None:
    view = DocumentView(text="y" * 900, title="t")
    fake = FakeClassifier(fail_sizes_over=800)
    outcome = await judge_document(view, judgment_specs(None, [q("w")]), fake)
    sent = [len(str(c[0]["document"]["text"])) for c in fake.calls]  # type: ignore[index, call-overload]
    assert sent[0] == 900 and 0.80 * 900 < sent[1] <= 0.85 * 900
    judgment = outcome.answers["w"]
    assert isinstance(judgment, Judgment)
    assert judgment.coverage.shrunk and judgment.coverage.truncated


async def test_a_second_max_tokens_exceeded_is_an_error_not_a_retry_loop() -> None:
    fake = FakeClassifier(fail_sizes_over=10)
    outcome = await judge_document(
        DocumentView(text="y" * 900), judgment_specs(None, [q("w")]), fake
    )
    assert len(fake.calls) == 2
    assert "w" not in outcome.answers
    assert [e.code for e in outcome.errors] == ["ClassifierTokensExceeded"]


async def test_jev_400_max_tokens_exceeded_becomes_the_seams_error() -> None:
    body = {"detail": {"error_type": "max_tokens_exceeded"}}
    client = ts.AsyncTypeSafeClient(
        api_key="k", transport=httpx2.MockTransport(lambda r: httpx2.Response(400, json=body))
    )
    with pytest.raises(ClassifierTokensExceeded):
        await JevClassifier(client=client).judge({"document": {"text": "x"}}, {"q": Noul("?")})


# --- missing Sections -----------------------------------------------------------------------

NO_CONTROLS = DocumentView(text="A" * 100, sections=[section("business", 0, 100)], title="t")


async def test_a_question_whose_section_is_missing_skips_with_a_reason() -> None:
    fake = FakeClassifier()
    outcome = await judge_document(NO_CONTROLS, judgment_specs(None, [q("c", ["controls"])]), fake)
    skipped = outcome.answers["c"]
    assert isinstance(skipped, SkippedAnswer)
    assert "controls" in skipped.skipped
    assert skipped.coverage.sections == [] and skipped.coverage.spans == []
    assert fake.calls == []


@pytest.mark.parametrize(
    "bad",
    [
        {"flags": ["pointer"]},
        {"verification": Verification(status="rejected", p=0.1)},
        {"confidence": 0.2},
    ],
)
def test_pointer_rejected_and_doubtful_sections_count_as_missing(bad: dict[str, object]) -> None:
    view = DocumentView(text="A" * 100, sections=[section("controls", 0, 100, **bad)])
    _, skipped = plan_requests(view, judgment_specs(None, [q("c", ["controls"])]), CAPS)
    assert "c" in skipped


async def test_fallback_head_tail_reads_the_document_and_says_so() -> None:
    fake = FakeClassifier()
    specs = judgment_specs(None, [q("c", ["controls"], fallback="head_tail")])
    outcome = await judge_document(NO_CONTROLS, specs, fake)
    judgment = outcome.answers["c"]
    assert isinstance(judgment, Judgment)
    assert judgment.coverage.sections == []
    assert "controls" in (judgment.coverage.note or "")
    assert "text" in fake.calls[0][0]["document"]  # type: ignore[operator]


async def test_the_taxonomy_never_skips() -> None:
    fake = FakeClassifier()
    taxonomy = TAXONOMY.model_copy(update={"reads": ["controls"]})
    outcome = await judge_document(NO_CONTROLS, judgment_specs(taxonomy, []), fake)
    assert outcome.classification is not None
    assert outcome.classification.coverage.spans == [Span(start=0, end=100)]


# --- Coverage -------------------------------------------------------------------------------


async def test_every_judgment_carries_its_coverage() -> None:
    specs = judgment_specs(TAXONOMY, [q("s", ["business"]), q("w")])
    outcome = await judge_document(BIG, specs, FakeClassifier())
    judgments = [outcome.classification, *outcome.answers.values()]
    assert all(isinstance(j, Judgment) for j in judgments)
    sector = outcome.answers["s"]
    assert isinstance(sector, Judgment)
    assert sector.coverage.sections == ["business"]
    assert sector.coverage.spans == [Span(start=0, end=100)]
    assert sector.coverage.est_tokens > 0 and sector.coverage.input_tokens == 100
    assert outcome.usage.requests == 2


# --- cache ----------------------------------------------------------------------------------


@pytest.fixture
def ws(tmp_path: Path):
    with Workspace.open(tmp_path / "ws", filesystem_type=lambda _p: "apfs") as w:
        yield w


async def test_a_repeat_run_is_answered_from_the_cache(ws: Workspace) -> None:
    specs = judgment_specs(TAXONOMY, [q("s", ["business"])])
    first, second = FakeClassifier(), FakeClassifier()
    one = await judge_document(BIG, specs, first, ws.cache)
    two = await judge_document(BIG, specs, second, ws.cache)
    assert second.calls == [] and two.usage.requests == 0 and two.usage.cache_hits == 2
    assert two.classification is not None and one.classification is not None
    assert two.classification.value == one.classification.value
    assert two.classification.classifier == INFO
    assert two.classification.coverage.input_tokens is None  # nothing was sent


async def test_changing_one_question_pays_only_for_that_question(ws: Workspace) -> None:
    await judge_document(BIG, judgment_specs(None, [q("a"), q("b")]), FakeClassifier(), ws.cache)
    fake = FakeClassifier()
    changed = Question(name="b", kind="noul", instructions="A different question?")
    await judge_document(BIG, judgment_specs(None, [q("a"), changed]), fake, ws.cache)
    assert len(fake.calls) == 1
    assert [a for a in fake.calls[0][1]] == ["q_b"]


async def test_a_different_model_version_misses_the_cache(ws: Workspace) -> None:
    specs = judgment_specs(None, [q("a")])
    await judge_document(BIG, specs, FakeClassifier(), ws.cache)
    fake = FakeClassifier(replace(CAPS, model="m-2"))
    await judge_document(BIG, specs, fake, ws.cache)
    assert len(fake.calls) == 1


async def test_no_cache_means_no_lookups_and_no_writes(ws: Workspace) -> None:
    specs = judgment_specs(None, [q("a")])
    await judge_document(BIG, specs, FakeClassifier(), None)
    fake = FakeClassifier()
    await judge_document(BIG, specs, fake, ws.cache)
    assert len(fake.calls) == 1  # the --no-cache run wrote nothing


# --- review fixes ---------------------------------------------------------------------------


async def test_a_shrunk_request_is_still_a_cache_hit_on_rerun(ws: Workspace) -> None:
    view = DocumentView(text="y" * 900, title="t")
    specs = judgment_specs(None, [q("w")])
    first = FakeClassifier(fail_sizes_over=800)
    await judge_document(view, specs, first, ws.cache)
    assert len(first.calls) == 2  # it needed the shrink
    again = FakeClassifier(fail_sizes_over=800)
    outcome = await judge_document(view, specs, again, ws.cache)
    assert again.calls == [] and outcome.usage.cache_hits == 1


def test_the_cut_is_measured_in_the_estimates_own_units() -> None:
    nasty = 'a "quoted"\nline\n' * 200  # every newline and quote is two characters of JSON
    view = DocumentView(text=nasty, title="t")
    caps = tiny(300)
    (request,), _ = plan_requests(view, judgment_specs(None, [q("w")]), caps)
    asks = {s.wire_name: s.ask for s in request.specs}
    assert request.coverage.truncated
    assert estimate_tokens(request.state, asks, caps) <= caps.token_budget


def test_the_budget_is_state_plus_the_longest_judgment_not_all_of_them() -> None:
    one, _ = plan_requests(LONG, judgment_specs(None, [q("a")]), tiny(215))
    three, _ = plan_requests(LONG, judgment_specs(None, [q("a"), q("b"), q("c")]), tiny(215))
    kept = lambda r: sum(s.end - s.start for s in r[0].coverage.spans)  # noqa: E731
    assert kept(three) == kept(one)


async def test_a_judgment_the_classifier_did_not_answer_is_an_error() -> None:
    class Forgetful(FakeClassifier):
        async def judge(self, state: JSONState, judgments: Mapping[str, Ask]) -> ClassifierResponse:
            response = await super().judge(state, judgments)
            return replace(response, results={})

    outcome = await judge_document(BIG, judgment_specs(None, [q("w")]), Forgetful())
    assert "w" not in outcome.answers
    assert [(e.stage, e.code) for e in outcome.errors] == [("classify", "missing_result")]


def test_a_section_with_several_keys_is_read_once() -> None:
    both = Section(
        keys=["business", "risk_factors"],
        label="Items 1 and 1A",
        spans=[Span(start=0, end=100)],
        method="regex",
    )
    view = DocumentView(text="Z" * 100, sections=[both])
    (request,), _ = plan_requests(
        view, judgment_specs(None, [q("s", ["business", "risk_factors"])]), CAPS
    )
    assert request.coverage.spans == [Span(start=0, end=100)]
    assert str(request.state).count("Z" * 100) == 1


async def test_a_cache_needs_to_know_the_model_and_classifier(ws: Workspace) -> None:
    anonymous = FakeClassifier(replace(CAPS, model=""))
    with pytest.raises(ValueError, match="model"):
        await judge_document(BIG, judgment_specs(None, [q("a")]), anonymous, ws.cache)


async def test_two_classifiers_with_one_model_name_do_not_share_cache_entries(
    ws: Workspace,
) -> None:
    specs = judgment_specs(None, [q("a")])
    await judge_document(BIG, specs, FakeClassifier(), ws.cache)
    other = FakeClassifier(replace(CAPS, classifier_id="other"))
    await judge_document(BIG, specs, other, ws.cache)
    assert len(other.calls) == 1


async def test_a_pointer_section_is_reported_with_where_it_points() -> None:
    pointer = section("controls", 0, 100, flags=["pointer"], form_ref="Exhibit 13")
    view = DocumentView(text="A" * 100, sections=[pointer])
    outcome = await judge_document(
        view, judgment_specs(None, [q("c", ["controls"])]), FakeClassifier()
    )
    skipped = outcome.answers["c"]
    assert isinstance(skipped, SkippedAnswer)
    assert "pointer → Exhibit 13" in (skipped.coverage.note or "")
    assert "pointer → Exhibit 13" in skipped.skipped
