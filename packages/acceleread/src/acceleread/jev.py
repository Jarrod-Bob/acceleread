# SPDX-License-Identifier: Apache-2.0
"""Jev, the default Classifier, over TypeSafe's async SDK (docs/spec/v0.md §5.4)."""

from collections.abc import Mapping
from importlib.metadata import version
from typing import cast

import typesafe_sdk as ts

from acceleread.classifier import (
    Ask,
    Capabilities,
    Choice,
    ClassifierResponse,
    JSONState,
    JudgmentResult,
    Noul,
    Score,
)
from acceleread.models import DEFAULT_JEV_MODEL, ClassifierInfo

JEV_CAPABILITIES = Capabilities(
    kinds=frozenset({"noul", "score", "choice"}),
    max_choice_options=255,
    token_budget=32_000,
    chars_per_token=3.0,
)


def _question(ask: Ask) -> ts.Question:
    match ask:
        case Noul(instructions=instructions):
            return ts.Noul(instructions=instructions)
        case Score(instructions=instructions, criteria=criteria):
            return ts.Score(instructions=instructions, criteria=list(criteria))
        case Choice(instructions=instructions, options=options):
            return ts.Choice(instructions=instructions, criteria=dict(options))
    raise TypeError(f"unsupported Judgment: {ask!r}")


def _result(answer: ts.Answer) -> JudgmentResult:
    match answer:
        case ts.ChoiceAnswer():
            return JudgmentResult(
                kind="choice",
                value=answer.choice,
                probabilities=dict(answer.probabilities),
                confidence=answer.confidence,
            )
        case ts.ScoreAnswer():
            return JudgmentResult(
                kind="score",
                value=answer.score,
                probabilities={str(k): v for k, v in answer.probabilities.items()},
                confidence=answer.confidence,
            )
        case ts.NoulAnswer():
            return JudgmentResult(kind="noul", value=answer.noul)
    raise TypeError(f"unsupported answer: {answer!r}")


class JevClassifier:
    def __init__(
        self, model: str = DEFAULT_JEV_MODEL, client: ts.AsyncTypeSafeClient | None = None
    ) -> None:
        self.model = model
        self._client = client

    @property
    def capabilities(self) -> Capabilities:
        return JEV_CAPABILITIES

    def _get_client(self) -> ts.AsyncTypeSafeClient:
        if self._client is None:  # created lazily so a missing key fails at first use, not import
            self._client = ts.AsyncTypeSafeClient()
        return self._client

    async def judge(self, state: JSONState, judgments: Mapping[str, Ask]) -> ClassifierResponse:
        questions = {name: _question(ask) for name, ask in judgments.items()}
        # Same JSON shape as the SDK's own alias, which mypy can't match structurally.
        state_json = cast(ts.JSONContent, dict(state))
        response = await self._get_client().system_one(state_json, questions, model=self.model)
        return ClassifierResponse(
            results={name: _result(answer) for name, answer in response.answers.items()},
            info=ClassifierInfo(id="jev", model=response.model, version=version("typesafe-sdk")),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
