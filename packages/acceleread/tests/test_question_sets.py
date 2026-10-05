# SPDX-License-Identifier: Apache-2.0
"""Questions and Question Sets: loading from YAML/JSON (docs/spec/v0.md §5.1)."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from acceleread.models import Category, Question, QuestionSet, Taxonomy

SETS = Path(__file__).parent / "fixtures" / "sets"


def test_loads_a_question_set_from_yaml() -> None:
    qs = QuestionSet.from_file(SETS / "filing_risk.yaml")
    assert (qs.name, qs.version) == ("filing-risk", "2")  # a bare YAML number is a version string
    assert qs.taxonomy is not None and qs.taxonomy.name == "filing-type"
    going_concern, outlook, sector = qs.questions
    assert going_concern.kind == "noul"
    assert going_concern.reads == ["controls", "financial_statements"]
    assert outlook.kind == "score"
    assert [c.name for c in outlook.criteria] == ["bleak", "cautious", "confident"]
    assert outlook.fallback == "head_tail" and outlook.escalate_below == 0.6
    assert [c.name for c in sector.criteria] == ["energy", "tech"]
    assert sector.criteria[0].description == "Oil, gas and power"


def test_loads_the_same_set_from_json(tmp_path: Path) -> None:
    yaml_set = QuestionSet.from_file(SETS / "filing_risk.yaml")
    path = tmp_path / "set.json"
    path.write_text(json.dumps(yaml_set.model_dump(mode="json")))
    assert QuestionSet.from_file(path) == yaml_set


def test_hash_is_stable_and_tracks_content() -> None:
    qs = QuestionSet.from_file(SETS / "filing_risk.yaml")
    assert qs.hash == QuestionSet.from_file(SETS / "filing_risk.yaml").hash
    assert qs.hash.startswith("sha256:")
    changed = qs.model_copy(update={"version": "3"})
    assert changed.hash != qs.hash


@pytest.mark.parametrize(
    "bad",
    [
        {"name": "q", "kind": "noul", "instructions": "x", "criteria": ["a", "b"]},
        {"name": "q", "kind": "score", "instructions": "x", "criteria": ["only"]},
        {"name": "q", "kind": "choice", "instructions": "x"},
        {"name": "q", "kind": "noul", "instructions": "x", "escalate_below": 1.5},
        {"name": "q", "kind": "noul", "instructions": "x", "fallback": "whole"},
        {"name": "q", "kind": "guess", "instructions": "x"},
    ],
)
def test_malformed_questions_are_rejected(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Question.model_validate(bad)


def test_a_misspelled_field_in_a_set_file_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "set.yaml"
    path.write_text(
        "name: s\nversion: 1\nquestions:\n"
        "  - {name: q, kind: noul, instructions: x, escalate_bellow: 0.5}\n"
    )
    with pytest.raises(ValidationError, match="escalate_bellow"):
        QuestionSet.from_file(path)


@pytest.mark.parametrize(
    ("model", "bad"),
    [
        (Taxonomy, {"name": "t", "categories": [{"name": "a"}], "escalate_bellow": 0.1}),
        (Taxonomy, {"name": "t", "categories": [{"name": "a", "descripton": "x"}]}),
        (QuestionSet, {"name": "s", "version": "1", "questions": [], "taxonomi": {}}),
    ],
)
def test_every_spec_file_model_forbids_unknown_fields(
    model: type[Taxonomy] | type[QuestionSet], bad: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(bad)


def test_taxonomy_hash_ignores_unset_optional_fields() -> None:
    plain = Taxonomy(name="t", categories=[Category(name="a", description="x")])
    # Pinned: adding optional fields to the models must not change an unchanged Taxonomy's hash.
    assert plain.hash == "sha256:cec99bf699157e592cf889965d151637bea5f147e1290305c9689c3984727b6e"
    assert plain.model_copy(update={"reads": ["business"]}).hash != plain.hash


def test_duplicate_category_and_criteria_names_are_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        Taxonomy(name="t", categories=[Category(name="a"), Category(name="a")])
    with pytest.raises(ValidationError, match="duplicate"):
        Question(
            name="q",
            kind="score",
            instructions="x",
            criteria=[Category(name="lo"), Category(name="lo")],
        )
