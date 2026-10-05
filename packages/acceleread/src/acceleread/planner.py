# SPDX-License-Identifier: Apache-2.0
"""The Planner: everything between a Job's Judgments and the Classifier call (spec §5.2-§5.3).

It groups a Document's Judgments by the Sections they read (ADR 0006), builds each group's state,
fits it to the Classifier's budget (head+tail, proportional across Sections), records Coverage on
every Judgment, answers from the Workspace Judgment cache where it can, and shrinks and retries
once when the Classifier says the state is too big. Nothing here logs Document text or state.
"""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
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
MIN_SECTION_CONFIDENCE = 0.5
UNUSABLE_VERIFICATION = frozenset({"rejected"})
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
    if "pointer" in section.flags:
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
    coverage: Coverage  # the group's, before the Judgment's own est_tokens
    notes: dict[str, str] = field(default_factory=dict)  # wire name → why it fell back
    text_chars: int = 0


@dataclass(frozen=True)
class _Segment:
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
    view: DocumentView, segments: list[_Segment], max_chars: int
) -> tuple[list[tuple[_Segment, list[tuple[int, int]]]], bool]:
    """Keep the spans of each segment that fit `max_chars` in all, ~25% head and ~75% tail."""
    total = sum(s.end - s.start for s in segments)
    max_chars -= len(JOIN) * len(segments)
    if total <= max_chars:
        return [(s, [(s.start, s.end)]) for s in segments], False
    kept: list[tuple[_Segment, list[tuple[int, int]]]] = []
    for seg in segments:
        length = seg.end - seg.start
        allowed = max(0, int(max_chars * length / total) - len(ELISION))
        head = int(allowed * HEAD_SHARE)
        tail = allowed - head
        if allowed >= length:
            kept.append((seg, [(seg.start, seg.end)]))
            continue
        head_end = _snap_head(view.pages, seg.start, seg.start + head) if head else seg.start
        tail_start = _snap_tail(view.pages, seg.end - tail, seg.end) if tail else seg.end
        spans = [(a, b) for a, b in ((seg.start, head_end), (tail_start, seg.end)) if b > a]
        kept.append((seg, spans))
    return kept, True


def _text_of(view: DocumentView, spans: list[tuple[int, int]]) -> str:
    return ELISION.join(view.text[a:b] for a, b in spans)


def _metadata(view: DocumentView) -> dict[str, JSONValue]:
    document: dict[str, JSONValue] = {"title": view.title}
    for name in ("company", "form", "period"):
        if (value := getattr(view, name)) is not None:
            document[name] = value
    return document


def _segments(view: DocumentView, reads: tuple[str, ...] | None) -> list[_Segment]:
    if reads is None:
        return [_Segment(None, 0, len(view.text))]
    return [
        _Segment(key, span.start, span.end)
        for key in reads
        for section in view.sections
        if key in section.keys and usable(section)
        for span in section.spans
    ]


def _build(
    view: DocumentView,
    specs: list[JudgmentSpec],
    reads: tuple[str, ...] | None,
    caps: Capabilities,
    *,
    shrink_from: int | None = None,
) -> Request:
    """Build one group's state. `shrink_from` is the text size a previous attempt sent."""
    segments = _segments(view, reads)
    asks = {s.wire_name: s.ask for s in specs}
    shell: dict[str, JSONValue] = {"document": {**_metadata(view)}}
    shell_doc = shell["document"]
    assert isinstance(shell_doc, dict)
    if reads is None:
        shell_doc["text"] = ""
    else:
        shell_doc["sections"] = {key: "" for key in reads}
    available = caps.token_budget - estimate_tokens(shell, asks, caps)
    max_chars = max(0, int(available * caps.chars_per_token))
    if shrink_from is not None:
        max_chars = min(max_chars, int(shrink_from * SHRINK_FACTOR))
    kept, truncated = _cut(view, segments, max_chars)
    spans = [Span(start=a, end=b) for _, pieces in kept for a, b in pieces]
    document = _metadata(view)
    if reads is None:
        document["text"] = _text_of(view, kept[0][1])
        text_chars = len(str(document["text"]))
    else:
        by_key: dict[str, list[str]] = {key: [] for key in reads}
        for seg, pieces in kept:
            assert seg.key is not None
            by_key[seg.key].append(_text_of(view, pieces))
        document["sections"] = {key: JOIN.join(parts) for key, parts in by_key.items()}
        text_chars = sum(len(JOIN.join(parts)) for parts in by_key.values())
    state: dict[str, JSONValue] = {"document": document}
    coverage = Coverage(
        sections=[] if reads is None else list(reads),
        spans=spans,
        page_ranges=_pages_in(view.pages, spans),
        truncated=truncated or shrink_from is not None,
        shrunk=shrink_from is not None,
        est_tokens=round(estimate_tokens(state, asks, caps)),
    )
    return Request(specs, reads, state, coverage, text_chars=text_chars)


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
            reason = "missing section: " + ", ".join(missing)
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


def _ask_definition(ask: Ask) -> dict[str, Any]:
    match ask:
        case Score(instructions=instructions, criteria=criteria):
            return {"kind": "score", "instructions": instructions, "criteria": list(criteria)}
        case Choice(instructions=instructions, options=options):
            return {"kind": "choice", "instructions": instructions, "options": dict(options)}
        case _:
            return {"kind": "noul", "instructions": ask.instructions}


def cache_key(state: dict[str, JSONValue], ask: Ask, model: str) -> str:
    """hash(state, Judgment definition, model version) (spec §5.3)."""
    payload = json.dumps(
        {"state": state, "judgment": _ask_definition(ask), "model": model},
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
    state = request.state
    # The state is shared; the Coverage of an unshrunk request is already on `request`.
    try:
        return request, await classifier.judge(state, {m.wire_name: m.ask for m in misses})
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
    requests, skipped = plan_requests(view, specs, caps)
    outcome = Outcome(answers=dict(skipped))
    for request in requests:
        hits: dict[str, tuple[JudgmentResult, ClassifierInfo]] = {}
        misses: list[JudgmentSpec] = []
        for spec in request.specs:
            raw = cache.get(cache_key(request.state, spec.ask, caps.model)) if cache else None
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
                        cache_key(used.state, spec.ask, caps.model),
                        _to_cache(result, response.info),
                    )
            else:
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
