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
    RecordError,
)
from acceleread.planner import DocumentView, JudgmentSpec, Outcome, judge_document
from acceleread.workspace.cache import JudgmentCache

_EPSILON = 1e-9  # so 200 documents at 0.5% is exactly one, not 0.999…


@dataclass
class EscalationBudget:
    """`escalation_max`: the share of a Job's Documents whose Judgments may go to the LLM.

    A Document takes one slot however many of its Judgments escalate. The cap rounds down.
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
        return math.floor(self.total_documents * self.escalation_max + _EPSILON)

    def claim(self) -> bool:
        if self.escalated >= self.cap:
            return False
        self.escalated += 1
        return True


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
) -> list[RecordError]:
    """Flag, and where allowed escalate, a Document's low-confidence Judgments in `outcome`.

    `classifier` is the LLM; None means flag only. Returns the errors of a failed escalation
    (stage `escalate`); the Judgments themselves keep their first-pass value, marked `failed`.
    """
    flagged: list[tuple[JudgmentSpec, Judgment, str]] = []
    for spec in specs:
        judgment = _current(outcome, spec)
        if judgment is None or spec.escalate_below is None:
            continue
        reason = _flag_reason(judgment, spec.escalate_below)
        if reason is not None:
            flagged.append((spec, judgment, reason))
    if not flagged:
        return []

    note = ""
    if classifier is not None and (budget is None or budget.claim()):
        return await _escalate(view, flagged, outcome, classifier, cache)
    if classifier is not None:
        note = "; not escalated: escalation_max reached"
    for spec, judgment, reason in flagged:
        _store(
            outcome,
            spec,
            _with_escalation(judgment, Escalation(status="flagged", reason=reason + note)),
        )
    return []


async def _escalate(
    view: DocumentView,
    flagged: list[tuple[JudgmentSpec, Judgment, str]],
    outcome: Outcome,
    classifier: Classifier,
    cache: JudgmentCache | None,
) -> list[RecordError]:
    second = await judge_document(view, [spec for spec, _, _ in flagged], classifier, cache)
    outcome.usage.requests += second.usage.requests
    outcome.usage.input_tokens += second.usage.input_tokens
    outcome.usage.output_tokens += second.usage.output_tokens
    outcome.usage.cache_hits += second.usage.cache_hits
    why_not = "; ".join(sorted({e.code for e in second.errors})) or "no result"
    for spec, first, reason in flagged:
        result = _current(second, spec)
        if result is None:
            escalation = Escalation(
                status="failed", reason=f"{reason}; escalation failed: {why_not}"
            )
            _store(outcome, spec, _with_escalation(first, escalation))
            continue
        kept = FirstPass(
            classifier=first.classifier,
            value=first.value,
            probabilities=first.probabilities,
            confidence=first.confidence,
            coverage=first.coverage,
        )
        _store(
            outcome,
            spec,
            _with_escalation(result, Escalation(status="escalated", reason=reason, first=kept)),
        )
    return [e.model_copy(update={"stage": "escalate"}) for e in second.errors]
