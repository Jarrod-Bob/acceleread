# SPDX-License-Identifier: Apache-2.0
"""The Planner: everything between a Job's Judgments and the Classifier call (spec §5.2-§5.3).

It groups a Document's Judgments by the Sections they read (ADR 0006), builds each group's state,
fits it to the Classifier's budget (head+tail, proportional across Sections), records Coverage on
every Judgment, answers from the Workspace Judgment cache where it can, and shrinks and retries
once when the Classifier says the state is too big. Nothing here logs Document text or state.
"""

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from acceleread.classifier import (
    Ask,
    Capabilities,
    Choice,
    Classifier,
    ClassifierResponse,
    ClassifierTokensExceeded,
    JSONValue,
    JudgmentResult,
    Noul,
    Score,
    ask_chars,
    ask_definition,
    estimate_tokens,
)
from acceleread.models import (
    ClassifierInfo,
    Coverage,
    Judgment,
    Page,
    Question,
    RecordError,
    Section,
    SkippedAnswer,
    Span,
    Taxonomy,
    Usage,
)
from acceleread.workspace.cache import JudgmentCache

HEAD_SHARE = 0.25
SHRINK_FACTOR = 0.85  # the state shrinks by 15% on max_tokens_exceeded
ELISION = "\n[…]\n"
JOIN = "\n\n"  # between the pieces of one Section key

# Starting values awaiting user confirmation (spec spirit of §14: to be measured, not decided).
# A Section is "missing" for `reads` when it is flagged `pointer`, its Verification status is
# `rejected`, or its confidence is below MIN_SECTION_CONFIDENCE. A Judgment reading several keys
# counts as missing (skips, or falls back) when ANY of its keys is missing.
POINTER_FLAG = "pointer"
UNUSABLE_VERIFICATION = frozenset({"rejected"})
MIN_SECTION_CONFIDENCE = 0.5

QUESTION_PREFIX = "q_"  # wire names: a Question can never collide with the Taxonomy's `taxonomy`
TAXONOMY_WIRE_NAME = "taxonomy"


@dataclass(frozen=True)
class JudgmentSpec:
    """One Judgment to plan: the Taxonomy's, or a Question's."""

    name: str  # the Question's name; "taxonomy" for the Taxonomy (never a key in `answers`)
    ask: Ask
    reads: tuple[str, ...] | None = None  # canonical Section keys; None reads the whole Document
    fallback: bool = False  # read head+tail when a Section is missing (always, for the Taxonomy)
    is_taxonomy: bool = False

    @property
    def wire_name(self) -> str:
        return TAXONOMY_WIRE_NAME if self.is_taxonomy else QUESTION_PREFIX + self.name


@dataclass(frozen=True)
class DocumentView:
    """What the Planner needs to know about one extracted Document."""

    text: str
    pages: Sequence[Page] = ()
    sections: Sequence[Section] = ()
    title: str = ""
    company: str | None = None
    form: str | None = None
    period: str | None = None


def taxonomy_choice(taxonomy: Taxonomy) -> Choice:
    return Choice(
        instructions="Which category best describes `document`?",
        options={c.name: c.description for c in taxonomy.categories},
    )


def question_ask(question: Question) -> Ask:
    match question.kind:
        case "noul":
            return Noul(question.instructions)
        case "score":
            return Score(question.instructions, tuple(c.name for c in question.criteria))
        case "choice":
            return Choice(question.instructions, {c.name: c.description for c in question.criteria})


def judgment_specs(taxonomy: Taxonomy | None, questions: Sequence[Question]) -> list[JudgmentSpec]:
    specs: list[JudgmentSpec] = []
    if taxonomy is not None:
        specs.append(
            JudgmentSpec(
                "taxonomy",
                taxonomy_choice(taxonomy),
                tuple(taxonomy.reads) if taxonomy.reads else None,
                fallback=True,
                is_taxonomy=True,
            )
        )
    for q in questions:
        reads = tuple(q.reads) if q.reads else None
        specs.append(JudgmentSpec(q.name, question_ask(q), reads, fallback=q.fallback is not None))
    return specs


def usable(section: Section) -> bool:
    """A Section exists for `reads` unless it is a pointer, failed Verification or is doubtful."""
    if POINTER_FLAG in section.flags:
        return False
    if section.verification is not None and section.verification.status in UNUSABLE_VERIFICATION:
        return False
    return section.confidence is None or section.confidence >= MIN_SECTION_CONFIDENCE


@dataclass
class Request:
    """One Classifier request: a group of Judgments sharing a state."""

    specs: list[JudgmentSpec]
    reads: tuple[str, ...] | None
    state: dict[str, JSONValue]
    coverage: Coverage  # the group's; each Judgment copies it with its own input_tokens and note
    notes: dict[str, str] = field(default_factory=dict)  # wire name → why it fell back
    text_chars: int = 0


@dataclass(frozen=True)
class _Extent:
    key: str | None
    start: int
    end: int


def _pages_in(pages: Sequence[Page], spans: Sequence[Span]) -> list[tuple[int, int]]:
    """Collapse the Pages the read spans touch into page-number ranges."""
    hit = sorted({p.number for p in pages for s in spans if p.start < s.end and s.start < p.end})
    ranges: list[tuple[int, int]] = []
    for number in hit:
        if ranges and ranges[-1][1] == number - 1:
            ranges[-1] = (ranges[-1][0], number)
        else:
            ranges.append((number, number))
    return ranges


def _snap_head(pages: Sequence[Page], start: int, end: int) -> int:
    """The last Page end within (start, end], else `end`."""
    ends = [p.end for p in pages if start < p.end <= end]
    return max(ends) if ends else end


def _snap_tail(pages: Sequence[Page], start: int, end: int) -> int:
    """The first Page start within [start, end), else `start`."""
    starts = [p.start for p in pages if start <= p.start < end]
    return min(starts) if starts else start


def _cut(
    view: DocumentView, extents: list[_Extent], max_chars: int
) -> tuple[list[tuple[_Extent, list[tuple[int, int]]]], bool]:
    """Keep the spans of each extent that fit `max_chars` in all, ~25% head and ~75% tail."""
    total = sum(e.end - e.start for e in extents)
    max_chars -= len(JOIN) * len(extents)
    if total == 0 or total <= max_chars:
        return [(e, [(e.start, e.end)]) for e in extents], False
    kept: list[tuple[_Extent, list[tuple[int, int]]]] = []
    for ext in extents:
        length = ext.end - ext.start
        allowed = max(0, int(max_chars * length / total) - len(ELISION))
        head = int(allowed * HEAD_SHARE)
        tail = allowed - head
        if allowed >= length:
            kept.append((ext, [(ext.start, ext.end)]))
            continue
        head_end = _snap_head(view.pages, ext.start, ext.start + head) if head else ext.start
        tail_start = _snap_tail(view.pages, ext.end - tail, ext.end) if tail else ext.end
        spans = [(a, b) for a, b in ((ext.start, head_end), (tail_start, ext.end)) if b > a]
        kept.append((ext, spans))
    return kept, True


def _text_of(view: DocumentView, spans: list[tuple[int, int]]) -> str:
    return ELISION.join(view.text[a:b] for a, b in spans)


def _metadata(view: DocumentView) -> dict[str, JSONValue]:
    document: dict[str, JSONValue] = {"title": view.title}
    for name in ("company", "form", "period"):
        if (value := getattr(view, name)) is not None:
            document[name] = value
    return document


def _extents(view: DocumentView, reads: tuple[str, ...] | None) -> list[_Extent]:
    """What to read. A Section carrying several keys is read once, under the first key read."""
    if reads is None:
        return [_Extent(None, 0, len(view.text))]
    extents: list[_Extent] = []
    for section in view.sections:
        key = next((k for k in reads if k in section.keys), None)
        if key is not None and usable(section):
            extents += [_Extent(key, span.start, span.end) for span in section.spans]
    return extents


@dataclass(frozen=True)
class _Rendered:
    state: dict[str, JSONValue]
    spans: list[Span]
    truncated: bool
    text_chars: int


def _render(
    view: DocumentView, extents: list[_Extent], reads: tuple[str, ...] | None, max_chars: int
) -> _Rendered:
    kept, truncated = _cut(view, extents, max_chars)
    by_key: dict[str | None, list[str]] = {}
    for ext, pieces in kept:
        by_key.setdefault(ext.key, []).append(_text_of(view, pieces))
    joined = {key: JOIN.join(parts) for key, parts in by_key.items()}
    document = _metadata(view)
    if reads is None:
        document["text"] = joined.get(None, "")
    else:
        document["sections"] = {key: joined[key] for key in reads if key in joined}
    return _Rendered(
        state={"document": document},
        spans=[Span(start=a, end=b) for _, pieces in kept for a, b in pieces],
        truncated=truncated,
        text_chars=sum(len(text) for text in joined.values()),
    )


def _longest(asks: dict[str, Ask]) -> dict[str, Ask]:
    """The limit covers the state plus the longest Judgment (spec §5.3), not every Judgment."""
    name = max(asks, key=lambda n: ask_chars(asks[n]))
    return {name: asks[name]}


def _build(
    view: DocumentView,
    specs: list[JudgmentSpec],
    reads: tuple[str, ...] | None,
    caps: Capabilities,
    *,
    shrink_from: int | None = None,
) -> Request:
    """Build one group's state. `shrink_from` is the text size a previous attempt sent."""
    extents = _extents(view, reads)
    asks = {s.wire_name: s.ask for s in specs}
    budgeted = _longest(asks)
    empty = _render(view, [replace(e, end=e.start) for e in extents], reads, 0)
    room = caps.token_budget - estimate_tokens(empty.state, budgeted, caps)
    max_chars = max(0, int(room * caps.chars_per_token))
    if shrink_from is not None:
        max_chars = min(max_chars, int(shrink_from * SHRINK_FACTOR))
    while True:
        rendered = _render(view, extents, reads, max_chars)
        excess = estimate_tokens(rendered.state, budgeted, caps) - caps.token_budget
        if excess <= 0 or max_chars == 0:  # measured in the estimate's own units (JSON escapes)
            break
        max_chars = max(0, max_chars - math.ceil(excess * caps.chars_per_token))
    coverage = Coverage(
        sections=[] if reads is None else list(reads),
        spans=rendered.spans,
        page_ranges=_pages_in(view.pages, rendered.spans),
        truncated=rendered.truncated or shrink_from is not None,
        shrunk=shrink_from is not None,
        est_tokens=round(estimate_tokens(rendered.state, asks, caps)),
    )
    return Request(specs, reads, rendered.state, coverage, text_chars=rendered.text_chars)


def _missing_reason(view: DocumentView, key: str) -> str:
    for section in view.sections:
        if key in section.keys and POINTER_FLAG in section.flags:
            return f"{key} (pointer → {section.form_ref or section.label})"
    return key


def plan_requests(
    view: DocumentView, specs: Sequence[JudgmentSpec], caps: Capabilities
) -> tuple[list[Request], dict[str, SkippedAnswer]]:
    """Group Judgments by distinct `reads` set; Sections that don't exist skip or fall back."""
    present = {key for s in view.sections if usable(s) for key in s.keys}
    groups: dict[tuple[str, ...] | None, list[JudgmentSpec]] = {}
    fallen: dict[str, str] = {}  # wire name → why it fell back
    skipped: dict[str, SkippedAnswer] = {}
    for spec in specs:
        reads = None if spec.reads is None else tuple(sorted(set(spec.reads)))
        missing = [] if reads is None else [k for k in reads if k not in present]
        if missing:
            reason = "missing section: " + ", ".join(_missing_reason(view, k) for k in missing)
            if not spec.fallback:
                skipped[spec.name] = SkippedAnswer(
                    skipped=reason, coverage=Coverage(est_tokens=0, note=reason)
                )
                continue
            fallen[spec.wire_name] = reason + "; read head+tail instead"
            reads = None
        groups.setdefault(reads, []).append(spec)
    requests: list[Request] = []
    for reads, members in groups.items():
        request = _build(view, members, reads, caps)
        request.notes = {m.wire_name: fallen[m.wire_name] for m in members if m.wire_name in fallen}
        requests.append(request)
    return requests, skipped


def cache_key(state: dict[str, JSONValue], ask: Ask, caps: Capabilities) -> str:
    """hash(state, Judgment definition, Classifier and model version) (spec §5.3)."""
    payload = json.dumps(
        {
            "state": state,
            "judgment": ask_definition(ask),
            "classifier": caps.classifier_id,
            "model": caps.model,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class Outcome:
    """The Judgments the Planner produced for one Document."""

    classification: Judgment | None = None
    answers: dict[str, Judgment | SkippedAnswer] = field(default_factory=dict)
    errors: list[RecordError] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


def _judgment(
    result: JudgmentResult,
    info: ClassifierInfo,
    coverage: Coverage,
    input_tokens: int | None,
    note: str | None,
) -> Judgment:
    return Judgment(
        kind=result.kind,
        value=result.value,
        probabilities=result.probabilities,
        confidence=result.confidence,
        classifier=info,
        coverage=coverage.model_copy(update={"input_tokens": input_tokens, "note": note}),
    )


def _to_cache(result: JudgmentResult, info: ClassifierInfo) -> dict[str, Any]:
    return {
        "kind": result.kind,
        "value": result.value,
        "probabilities": result.probabilities,
        "confidence": result.confidence,
        "classifier": info.model_dump(),
    }


def _from_cache(raw: dict[str, Any]) -> tuple[JudgmentResult, ClassifierInfo]:
    result = JudgmentResult(
        kind=raw["kind"],
        value=raw["value"],
        probabilities=raw["probabilities"],
        confidence=raw["confidence"],
    )
    return result, ClassifierInfo.model_validate(raw["classifier"])


async def _call(
    view: DocumentView,
    request: Request,
    misses: list[JudgmentSpec],
    classifier: Classifier,
) -> tuple[Request, ClassifierResponse]:
    """One Classifier call for the uncached Judgments, shrinking and retrying once if too big."""
    caps = classifier.capabilities
    try:
        return request, await classifier.judge(request.state, {m.wire_name: m.ask for m in misses})
    except ClassifierTokensExceeded:
        shrunk = _build(
            view,
            request.specs,
            request.reads,
            caps,
            shrink_from=request.text_chars,
        )
        shrunk.notes = request.notes
        response = await classifier.judge(shrunk.state, {m.wire_name: m.ask for m in misses})
        return shrunk, response


async def judge_document(
    view: DocumentView,
    specs: Sequence[JudgmentSpec],
    classifier: Classifier,
    cache: JudgmentCache | None = None,
) -> Outcome:
    """Plan and run every Judgment for one Document.

    A failing group is reported in `errors` (code = the exception name, so the runner can act on
    `ClassifierRejected`) and leaves the Document's other groups alone.
    """
    caps = classifier.capabilities
    if cache is not None and not (caps.model and caps.classifier_id):
        raise ValueError("the Judgment cache needs the Classifier's model and id to key entries")
    requests, skipped = plan_requests(view, specs, caps)
    outcome = Outcome(answers=dict(skipped))
    for request in requests:
        hits: dict[str, tuple[JudgmentResult, ClassifierInfo]] = {}
        misses: list[JudgmentSpec] = []
        for spec in request.specs:
            raw = cache.get(cache_key(request.state, spec.ask, caps)) if cache else None
            if raw is not None:
                hits[spec.wire_name] = _from_cache(raw)
            else:
                misses.append(spec)
        used = request
        response: ClassifierResponse | None = None
        if misses:
            try:
                used, response = await _call(view, request, misses, classifier)
            except Exception as exc:  # one group failing leaves the Document's other groups alone
                outcome.errors.append(
                    RecordError(stage="classify", code=type(exc).__name__, message=str(exc))
                )
        for spec in request.specs:
            if spec.wire_name in hits:
                result, info = hits[spec.wire_name]
                judgment = _judgment(
                    result, info, request.coverage, None, request.notes.get(spec.wire_name)
                )
                outcome.usage.cache_hits += 1
            elif response is not None and spec.wire_name in response.results:
                result = response.results[spec.wire_name]
                judgment = _judgment(
                    result,
                    response.info,
                    used.coverage,
                    response.input_tokens,
                    request.notes.get(spec.wire_name),
                )
                if cache is not None:
                    cache.put(
                        cache_key(request.state, spec.ask, caps),  # the unshrunk state: reruns hit
                        _to_cache(result, response.info),
                    )
            else:
                if response is not None:
                    outcome.errors.append(
                        RecordError(
                            stage="classify",
                            code="missing_result",
                            message=f"the Classifier gave no result for {spec.name}",
                        )
                    )
                continue
            if spec.is_taxonomy:
                outcome.classification = judgment
            else:
                outcome.answers[spec.name] = judgment
        if response is not None:
            outcome.usage.requests += response.requests
            outcome.usage.input_tokens += response.input_tokens or 0
            outcome.usage.output_tokens += response.output_tokens or 0
    return outcome
