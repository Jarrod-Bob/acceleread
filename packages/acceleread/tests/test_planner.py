# SPDX-License-Identifier: Apache-2.0
"""The Planner: grouping, state, budget, Coverage and cache (docs/spec/v0.md §5.2-§5.3)."""

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
    JudgmentSpec,
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
)
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


def test_estimate_adds_the_per_request_overhead() -> None:
    state = {"document": {"text": "x" * 290}}  # 314 chars compact
    asks = {"q": Noul("0123456789")}
    bare = estimate_tokens(state, asks, CAPS)
    with_overhead = estimate_tokens(state, asks, replace(CAPS, request_overhead_tokens=220))
    assert bare == (314 + 10) / 3.0
    assert with_overhead == bare + 220


def test_jev_declares_its_measured_overhead() -> None:
    # A 2-page fixture estimated 286 tokens against 506 actual (the tracer's finding).
    assert JEV_CAPABILITIES.request_overhead_tokens >= 200


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


def test_jev_400_max_tokens_exceeded_becomes_the_seams_error() -> None:
    body = {"detail": {"error_type": "max_tokens_exceeded"}}
    client = ts.AsyncTypeSafeClient(
        api_key="k", transport=httpx2.MockTransport(lambda r: httpx2.Response(400, json=body))
    )
    import asyncio

    with pytest.raises(ClassifierTokensExceeded):
        asyncio.run(
            JevClassifier(client=client).judge({"document": {"text": "x"}}, {"q": Noul("?")})
        )


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


def test_judgment_spec_wire_names_never_collide() -> None:
    assert (
        JudgmentSpec("taxonomy", Noul("?")).wire_name
        != JudgmentSpec("x", Noul("?"), is_taxonomy=True).wire_name
    )
