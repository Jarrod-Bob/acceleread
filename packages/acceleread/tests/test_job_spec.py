# SPDX-License-Identifier: Apache-2.0
"""The Job spec every surface shares (docs/spec/v0.md §7.1)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from acceleread import JobSpec, Taxonomy
from acceleread.models import (
    Category,
    DocumentInput,
    DocumentOverride,
    LLMEscalation,
    Question,
)

TAXONOMY = Taxonomy(name="t", categories=[Category(name="a")])


def test_defaults_match_the_spec() -> None:
    spec = JobSpec(inputs=[Path("a.pdf")], taxonomy=TAXONOMY)
    assert spec.extraction_profile == "fast"
    assert spec.ocr_languages == ["en"]
    assert spec.escalate_below is None  # no default threshold
    assert spec.llm_escalation == LLMEscalation(
        enabled=False, model="claude-opus-5-5", escalation_max=0.02
    )
    assert spec.max_cost_usd is None
    assert spec.cache is True
    assert spec.user_agent is None
    assert spec.question_sets == [] and spec.questions == []


def test_a_spec_may_have_no_taxonomy_here() -> None:
    # "At least one Taxonomy or Question" is a submit-time check in validate(), because a
    # Question Set can supply the Taxonomy.
    JobSpec(inputs=[Path("a.pdf")])


def test_carries_questions_sets_and_per_document_overrides() -> None:
    spec = JobSpec(
        inputs=[Path("a.pdf"), Path("b.pdf")],
        question_sets=[Path("risk.yaml")],
        questions=[Question(name="q", kind="noul", instructions="Is `document` a filing?")],
        extraction_profile="quality",
        ocr_languages=["en", "de"],
        escalate_below=0.7,
        llm_escalation=LLMEscalation(enabled=True, escalation_max=0.05),
        max_cost_usd=12.5,
        cache=False,
        user_agent="acme research bot (ops@acme.test)",
        overrides={"b.pdf": DocumentOverride(extraction_profile="fast", ocr_languages=["fr"])},
    )
    assert spec.overrides["b.pdf"].ocr_languages == ["fr"]
    assert JobSpec.model_validate(spec.model_dump(mode="json")) == spec


@pytest.mark.parametrize(
    "bad",
    [
        {"inputs": []},
        {"inputs": ["a.pdf"], "extraction_profile": "turbo"},
        {"inputs": ["a.pdf"], "ocr_languages": []},
        {"inputs": ["a.pdf"], "escalate_below": 2},
        {"inputs": ["a.pdf"], "max_cost_usd": 0},
        {"inputs": ["a.pdf"], "llm_escalation": {"escalation_max": 1.5}},
        {"inputs": ["a.pdf"], "surprise": True},
    ],
)
def test_invalid_specs_are_rejected(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        JobSpec.model_validate(bad)


def test_inputs_are_verbatim_strings_or_objects_with_ids_and_metadata() -> None:
    spec = JobSpec.model_validate(
        {
            "inputs": [
                "https://example.test/a.pdf?x=1",
                "reports/*.pdf",
                Path("local.pdf"),
                {"source": "b.pdf", "external_id": "ext-1", "user_metadata": {"batch": 3}},
            ]
        }
    )
    assert [i.source for i in spec.inputs] == [
        "https://example.test/a.pdf?x=1",
        "reports/*.pdf",
        "local.pdf",
        "b.pdf",
    ]
    assert spec.inputs[0] == DocumentInput(source="https://example.test/a.pdf?x=1")
    assert spec.inputs[3].external_id == "ext-1" and spec.inputs[3].user_metadata == {"batch": 3}
    assert JobSpec.model_validate(spec.model_dump(mode="json")) == spec
