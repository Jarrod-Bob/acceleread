# SPDX-License-Identifier: Apache-2.0
"""The thin Classifier seam (ADR 0007, docs/spec/v0.md §5.4).

A Classifier answers a set of named Judgments about one state. Planning (grouping, budgets,
Coverage, caching, Escalation) lives above this seam, so implementations stay small.
"""

import json
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
    request_overhead_tokens: int = 0  # fixed cost the Classifier adds to every request
    model: str = ""  # the model version, part of every Judgment cache key


def ask_chars(ask: Ask) -> int:
    chars = len(ask.instructions)
    match ask:
        case Score(criteria=criteria):
            chars += sum(len(c) for c in criteria)
        case Choice(options=options):
            chars += sum(len(k) + len(v or "") for k, v in options.items())
    return chars


def estimate_tokens(
    state: JSONState, judgments: Mapping[str, Ask], capabilities: Capabilities
) -> float:
    """The one token estimate, shared by the Planner's budget and the rate limiter's charge.

    Characters of state plus each Judgment's instructions and criteria, divided by the
    Classifier's chars-per-token, plus its fixed per-request overhead.
    """
    chars = len(json.dumps(state, separators=(",", ":"), ensure_ascii=False))
    chars += sum(ask_chars(ask) for ask in judgments.values())
    return chars / capabilities.chars_per_token + capabilities.request_overhead_tokens


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


class ClassifierError(Exception):
    """Base of the failures a Classifier reports to the rate limiter (docs/spec/v0.md §7.5)."""


class ClassifierThrottled(ClassifierError):
    """The Classifier asked us to slow down (429 or 529). Never fails a Document."""

    def __init__(self, retry_after: float | None = None) -> None:
        super().__init__("classifier throttled")
        self.retry_after = retry_after  # seconds


class ClassifierTransient(ClassifierError):
    """A timeout or connection error. Never fails a Document; the request is re-queued."""


class ClassifierUnavailable(ClassifierError):
    """A server-side failure (5xx). Persistent ones fail the Document."""


class ClassifierTokensExceeded(ClassifierError):
    """The state was over the Classifier's token limit (400 `max_tokens_exceeded`).

    The Planner shrinks the state and retries once; the rate limiter does not retry it.
    """


class ClassifierRejected(ClassifierError):
    """The Classifier refused the request as invalid (422). The Job auto-cancels."""
