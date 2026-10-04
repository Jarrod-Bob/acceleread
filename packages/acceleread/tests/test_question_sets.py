# SPDX-License-Identifier: Apache-2.0
"""Questions and Question Sets: loading from YAML/JSON (docs/spec/v0.md §5.1)."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from acceleread.models import Question, QuestionSet

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
