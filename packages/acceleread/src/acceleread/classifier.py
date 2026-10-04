# SPDX-License-Identifier: Apache-2.0
"""The thin Classifier seam (ADR 0007, docs/spec/v0.md §5.4).

A Classifier answers a set of named Judgments about one state. Planning (grouping, budgets,
Coverage, caching, Escalation) lives above this seam, so implementations stay small.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from acceleread.models import ClassifierInfo

type JSONValue = str | int | float | bool | Sequence[JSONValue] | Mapping[str, JSONValue] | None
JSONState = Mapping[str, JSONValue]


@dataclass(frozen=True)
class Noul:
    instructions: str


@dataclass(frozen=True)
class Score:
    instructions: str
    criteria: tuple[str, ...]  # ordered low to high


@dataclass(frozen=True)
class Choice:
    instructions: str
    options: Mapping[str, str | None]  # option name → description


Ask = Noul | Score | Choice


@dataclass(frozen=True)
class JudgmentResult:
    kind: Literal["noul", "score", "choice"]
    value: str | float
    probabilities: dict[str, float] | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class Capabilities:
    kinds: frozenset[str]
    max_choice_options: int
    token_budget: int  # state plus the longest Judgment
    chars_per_token: float  # conservative estimate used before sending


@dataclass(frozen=True)
class ClassifierResponse:
    results: dict[str, JudgmentResult]
    info: ClassifierInfo
    input_tokens: int | None = None
    output_tokens: int | None = None
    requests: int = 1


class Classifier(Protocol):
    @property
    def capabilities(self) -> Capabilities: ...

    async def judge(self, state: JSONState, judgments: Mapping[str, Ask]) -> ClassifierResponse: ...
