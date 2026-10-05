# SPDX-License-Identifier: Apache-2.0
"""Escalation: flag low-confidence Judgments, and optionally re-ask an LLM (docs/spec/v0.md §5.5).

The Runner calls `escalate` once per Document, after the first pass. A Judgment whose confidence is
below its `escalate_below` is always flagged. With an LLM Classifier and budget left, the flagged
Judgments are re-planned against that Classifier's capabilities (so it reads untruncated Sections),
through the Planner like any other call. The LLM's result becomes the Judgment of record, with
`probabilities` and `confidence` null, and the first pass is kept in `escalation.first`.
Nothing here logs Document text.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from acceleread.classifier import Classifier
from acceleread.models import (
    DEFAULT_ESCALATION_MAX,
    Escalation,
    FirstPass,
    Judgment,
)
from acceleread.planner import DocumentView, JudgmentSpec, Outcome, judge_document
from acceleread.workspace.cache import JudgmentCache

_EPSILON = 1e-9  # so 100 documents at 7% is exactly seven, not eight


@dataclass
class EscalationBudget:
    """`escalation_max`: the share of a Job's Documents whose Judgments may go to the LLM.

    A Document takes one slot however many of its Judgments escalate, and a failed escalation
    keeps its slot. A Document whose escalation was answered wholly from the Judgment cache gives
    its slot back: it cost nothing. The cap rounds up, so any `escalation_max` above 0 allows at
    least one Document.
    One budget is shared by every Document of a Job; `claim` never awaits, so concurrent
    Documents cannot overspend it.
    """

    total_documents: int
    escalation_max: float = DEFAULT_ESCALATION_MAX
    escalated: int = 0

    def __post_init__(self) -> None:
        if self.total_documents < 0:
            raise ValueError("total_documents cannot be negative")
        if not 0 <= self.escalation_max <= 1:
            raise ValueError("escalation_max is a share between 0 and 1")

    @property
    def cap(self) -> int:
        return math.ceil(self.total_documents * self.escalation_max - _EPSILON)

    def claim(self) -> bool:
        if self.escalated >= self.cap:
            return False
        self.escalated += 1
        return True

    def release(self) -> None:
        self.escalated = max(0, self.escalated - 1)


@dataclass(frozen=True)
class _Flagged:
    spec: JudgmentSpec
    first: Judgment
    reason: str


def _current(outcome: Outcome, spec: JudgmentSpec) -> Judgment | None:
    found = outcome.classification if spec.is_taxonomy else outcome.answers.get(spec.name)
    return found if isinstance(found, Judgment) else None


def _store(outcome: Outcome, spec: JudgmentSpec, judgment: Judgment) -> None:
    if spec.is_taxonomy:
        outcome.classification = judgment
    else:
        outcome.answers[spec.name] = judgment


def _flag_reason(judgment: Judgment, threshold: float) -> str | None:
    if judgment.confidence is None or judgment.confidence >= threshold:
        return None
    return f"confidence {judgment.confidence:g} below escalate_below {threshold:g}"


def _with_escalation(judgment: Judgment, escalation: Escalation) -> Judgment:
    return judgment.model_copy(update={"escalation": escalation})


async def escalate(
    view: DocumentView,
    specs: Sequence[JudgmentSpec],
    outcome: Outcome,
    *,
    classifier: Classifier | None,
    budget: EscalationBudget | None,
    cache: JudgmentCache | None = None,
) -> None:
    """Flag, and where allowed escalate, a Document's low-confidence Judgments in `outcome`.

    `classifier` is the LLM; None means flag only. A failed escalation leaves the first pass in
    place, marked `failed`, and adds an `escalate`-stage error to `outcome.errors`.
    """
    flagged: list[_Flagged] = []
    for spec in specs:
        judgment = _current(outcome, spec)
        if judgment is None or spec.escalate_below is None:
            continue
        reason = _flag_reason(judgment, spec.escalate_below)
        if reason is not None:
            flagged.append(_Flagged(spec, judgment, reason))
    if not flagged:
        return

    if classifier is not None and (budget is None or budget.claim()):
        await _escalate(view, flagged, outcome, classifier, budget, cache)
        return
    note = "; not escalated: escalation_max reached" if classifier is not None else ""
    for item in flagged:
        escalation = Escalation(status="flagged", reason=item.reason + note)
        _store(outcome, item.spec, _with_escalation(item.first, escalation))


async def _escalate(
    view: DocumentView,
    flagged: list[_Flagged],
    outcome: Outcome,
    classifier: Classifier,
    budget: EscalationBudget | None,
    cache: JudgmentCache | None,
) -> None:
    second = await judge_document(view, [item.spec for item in flagged], classifier, cache)
    outcome.usage.add(second.usage)
    if budget is not None and second.usage.requests == 0 and not second.errors:
        budget.release()  # every Judgment came from the cache
    why_not = "; ".join(sorted({e.code for e in second.errors})) or "no result"
    for item in flagged:
        result = _current(second, item.spec)
        if result is None:
            escalation = Escalation(
                status="failed", reason=f"{item.reason}; escalation failed: {why_not}"
            )
            _store(outcome, item.spec, _with_escalation(item.first, escalation))
            continue
        first = item.first
        kept = FirstPass(
            classifier=first.classifier,
            value=first.value,
            probabilities=first.probabilities,
            confidence=first.confidence,
            coverage=first.coverage,
        )
        escalation = Escalation(status="escalated", reason=item.reason, first=kept)
        _store(outcome, item.spec, _with_escalation(result, escalation))
    outcome.errors.extend(e.model_copy(update={"stage": "escalate"}) for e in second.errors)
