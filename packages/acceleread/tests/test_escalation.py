# SPDX-License-Identifier: Apache-2.0
"""Escalation: flagging below `escalate_below`, and re-asking an LLM (docs/spec/v0.md §5.5)."""

from collections.abc import Mapping

import pytest

from acceleread.classifier import (
    Ask,
    Capabilities,
    ClassifierResponse,
    ClassifierThrottled,
    JSONState,
    JudgmentResult,
)
from acceleread.escalation import EscalationBudget, escalate
from acceleread.models import (
    ClassifierInfo,
    Coverage,
    Judgment,
    Question,
    Section,
    Span,
    Taxonomy,
)
from acceleread.planner import DocumentView, Outcome, judgment_specs

JEV_INFO = ClassifierInfo(id="jev", model="jev-1.13.0", version="1")
LLM_INFO = ClassifierInfo(id="claude", model="claude-opus-5-5", version="1")
LLM_CAPS = Capabilities(
    kinds=frozenset({"noul", "score", "choice"}),
    max_choice_options=1000,
    token_budget=100_000,
    chars_per_token=3.0,
    model="claude-opus-5-5",
    classifier_id="claude",
)
BODY = "word " * 400
VIEW = DocumentView(
    text=BODY,
    title="Acme 10-K",
    sections=[
        Section(
            keys=["risk_factors"],
            label="Risk Factors",
            spans=[Span(start=0, end=len(BODY))],
            method="regex",
        )
    ],
)
TAXONOMY = Taxonomy(name="t", categories=[{"name": "bank"}, {"name": "tech"}]).with_other()  # type: ignore[list-item]


def question(name: str, **kw: object) -> Question:
    return Question(name=name, kind="noul", instructions=f"Is {name} so?", **kw)  # type: ignore[arg-type]


def first_pass(confidence: float | None, value: str | float = "bank") -> Judgment:
    return Judgment(
        kind="choice" if confidence is not None else "noul",
        value=value,
        probabilities=None if confidence is None else {"bank": confidence, "tech": 1 - confidence},
        confidence=confidence,
        classifier=JEV_INFO,
        coverage=Coverage(est_tokens=10, truncated=True, note="head+tail"),
    )


class FakeLLM:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[tuple[JSONState, Mapping[str, Ask]]] = []
        self.error = error

    @property
    def capabilities(self) -> Capabilities:
        return LLM_CAPS

    async def judge(self, state: JSONState, judgments: Mapping[str, Ask]) -> ClassifierResponse:
        self.calls.append((state, judgments))
        if self.error is not None:
            raise self.error
        return ClassifierResponse(
            results={
                name: JudgmentResult(kind="noul", value=1.0)
                if name.startswith("q_")
                else JudgmentResult(kind="choice", value="tech")
                for name in judgments
            },
            info=LLM_INFO,
            input_tokens=500,
            output_tokens=5,
        )


def outcome_with(classification: Judgment | None, **answers: Judgment) -> Outcome:
    return Outcome(classification=classification, answers=dict(answers))


def answer(outcome: Outcome, name: str) -> Judgment:
    found = outcome.answers[name]
    assert isinstance(found, Judgment)
    return found


async def test_a_judgment_below_escalate_below_is_flagged() -> None:
    specs = judgment_specs(TAXONOMY, [question("going_concern", escalate_below=0.8)])
    outcome = outcome_with(None, going_concern=first_pass(0.3, value=1.0))
    await escalate(VIEW, specs, outcome, classifier=None, budget=None)
    flagged = answer(outcome, "going_concern")
    assert flagged.escalation.status == "flagged"
    assert flagged.escalation.reason is not None and "0.3" in flagged.escalation.reason
    assert flagged.value == 1.0  # the first pass stays the record


async def test_a_judgment_at_or_above_the_threshold_or_without_one_is_left_alone() -> None:
    specs = judgment_specs(
        TAXONOMY,
        [question("a", escalate_below=0.5), question("b"), question("c", escalate_below=0.5)],
    )
    outcome = outcome_with(
        first_pass(0.1),  # the Taxonomy sets no threshold
        a=first_pass(0.5),  # exactly at it
        b=first_pass(0.01),  # no threshold at all
        c=first_pass(None),  # no confidence to compare (a Noul)
    )
    await escalate(VIEW, specs, outcome, classifier=None, budget=None)
    assert outcome.classification is not None
    assert outcome.classification.escalation.status == "none"
    assert {answer(outcome, n).escalation.status for n in "abc"} == {"none"}


async def test_the_jobs_escalate_below_applies_where_a_question_sets_none() -> None:
    specs = judgment_specs(TAXONOMY, [question("a"), question("b", escalate_below=0.1)], 0.6)
    outcome = outcome_with(first_pass(0.5), a=first_pass(0.5), b=first_pass(0.5))
    await escalate(VIEW, specs, outcome, classifier=None, budget=None)
    assert outcome.classification is not None
    assert outcome.classification.escalation.status == "flagged"
    assert answer(outcome, "a").escalation.status == "flagged"
    assert answer(outcome, "b").escalation.status == "none"


async def test_the_llm_re_reads_flagged_judgments_and_its_result_is_the_record() -> None:
    specs = judgment_specs(TAXONOMY, [question("a", escalate_below=0.6, reads=["risk_factors"])])
    outcome = outcome_with(first_pass(0.4), a=first_pass(0.4, value=0.0))
    errors = await escalate(
        VIEW, specs, outcome, classifier=FakeLLM(), budget=EscalationBudget(total_documents=100)
    )
    assert errors == []
    escalated = answer(outcome, "a")
    assert (escalated.value, escalated.probabilities, escalated.confidence) == (1.0, None, None)
    assert escalated.classifier == LLM_INFO
    assert escalated.escalation.status == "escalated"
    kept = escalated.escalation.first
    assert kept is not None
    assert (kept.value, kept.confidence, kept.classifier) == (0.0, 0.4, JEV_INFO)
    assert kept.coverage.truncated is True
    # The Taxonomy had no threshold of its own and was not asked about.
    assert outcome.classification is not None and outcome.classification.escalation.status == "none"


async def test_escalated_judgments_that_share_reads_go_in_one_call_untruncated() -> None:
    specs = judgment_specs(
        TAXONOMY,
        [
            question("a", escalate_below=0.9, reads=["risk_factors"]),
            question("b", escalate_below=0.9, reads=["risk_factors"]),
            question("c", escalate_below=0.9),
        ],
    )
    outcome = outcome_with(None, a=first_pass(0.1), b=first_pass(0.1), c=first_pass(0.1))
    llm = FakeLLM()
    await escalate(VIEW, specs, outcome, classifier=llm, budget=EscalationBudget(10_000))
    assert len(llm.calls) == 2  # one per distinct `reads` set
    by_names = {frozenset(judgments): state for state, judgments in llm.calls}
    state = by_names[frozenset({"q_a", "q_b"})]
    document = state["document"]
    assert isinstance(document, dict)
    sections = document["sections"]
    assert isinstance(sections, dict)
    assert sections["risk_factors"] == BODY
    assert answer(outcome, "a").coverage.truncated is False


async def test_the_taxonomy_escalates_too() -> None:
    taxonomy = TAXONOMY.model_copy(update={"escalate_below": 0.7})
    specs = judgment_specs(taxonomy, [])
    outcome = outcome_with(first_pass(0.4))
    await escalate(VIEW, specs, outcome, classifier=FakeLLM(), budget=EscalationBudget(1000))
    assert outcome.classification is not None
    assert outcome.classification.value == "tech"
    assert outcome.classification.escalation.status == "escalated"


async def test_the_llms_usage_is_added_to_the_documents() -> None:
    specs = judgment_specs(TAXONOMY, [question("a", escalate_below=0.9)])
    outcome = outcome_with(None, a=first_pass(0.1))
    outcome.usage.requests = 1
    await escalate(VIEW, specs, outcome, classifier=FakeLLM(), budget=EscalationBudget(1000))
    assert (outcome.usage.requests, outcome.usage.input_tokens, outcome.usage.output_tokens) == (
        2,
        500,
        5,
    )


def test_escalation_max_is_a_share_of_the_jobs_documents() -> None:
    budget = EscalationBudget(total_documents=100, escalation_max=0.02)
    assert [budget.claim() for _ in range(3)] == [True, True, False]
    assert budget.escalated == 2


def test_the_default_escalation_max_is_two_percent() -> None:
    assert EscalationBudget(total_documents=100).cap == 2
    assert EscalationBudget(total_documents=10).cap == 0  # rounds down; see the PR notes


async def test_past_the_cap_judgments_are_only_flagged() -> None:
    specs = judgment_specs(TAXONOMY, [question("a", escalate_below=0.9)])
    budget = EscalationBudget(total_documents=200, escalation_max=0.005)
    llm = FakeLLM()
    results = []
    for _ in range(2):
        outcome = outcome_with(None, a=first_pass(0.1))
        await escalate(VIEW, specs, outcome, classifier=llm, budget=budget)
        results.append(answer(outcome, "a"))
    assert [r.escalation.status for r in results] == ["escalated", "flagged"]
    assert "escalation_max" in (results[1].escalation.reason or "")
    assert len(llm.calls) == 1


async def test_a_document_uses_one_slot_however_many_judgments_it_escalates() -> None:
    specs = judgment_specs(
        TAXONOMY, [question("a", escalate_below=0.9), question("b", escalate_below=0.9)]
    )
    outcome = outcome_with(None, a=first_pass(0.1), b=first_pass(0.1))
    budget = EscalationBudget(total_documents=100, escalation_max=0.01)
    await escalate(VIEW, specs, outcome, classifier=FakeLLM(), budget=budget)
    assert budget.escalated == 1
    assert {answer(outcome, n).escalation.status for n in "ab"} == {"escalated"}


async def test_nothing_flagged_costs_no_slot_and_no_call() -> None:
    specs = judgment_specs(TAXONOMY, [question("a", escalate_below=0.2)])
    outcome = outcome_with(None, a=first_pass(0.9))
    budget = EscalationBudget(total_documents=100, escalation_max=1.0)
    llm = FakeLLM()
    await escalate(VIEW, specs, outcome, classifier=llm, budget=budget)
    assert (budget.escalated, llm.calls) == (0, [])


async def test_a_failed_escalation_keeps_the_first_pass_and_says_so() -> None:
    specs = judgment_specs(TAXONOMY, [question("a", escalate_below=0.9)])
    outcome = outcome_with(None, a=first_pass(0.1, value=0.0))
    errors = await escalate(
        VIEW,
        specs,
        outcome,
        classifier=FakeLLM(error=ClassifierThrottled()),
        budget=EscalationBudget(1000, escalation_max=1.0),
    )
    kept = answer(outcome, "a")
    assert (kept.value, kept.confidence, kept.classifier) == (0.0, 0.1, JEV_INFO)
    assert kept.escalation.status == "failed"
    assert "ClassifierThrottled" in (kept.escalation.reason or "")
    assert [(e.stage, e.code) for e in errors] == [("escalate", "ClassifierThrottled")]


async def test_flagged_judgments_stay_flagged_when_no_llm_is_configured() -> None:
    specs = judgment_specs(TAXONOMY, [question("a", escalate_below=0.9)])
    outcome = outcome_with(None, a=first_pass(0.1))
    await escalate(VIEW, specs, outcome, classifier=None, budget=None)
    assert answer(outcome, "a").escalation.status == "flagged"


def test_a_budget_cannot_be_negative() -> None:
    with pytest.raises(ValueError):
        EscalationBudget(total_documents=-1)
