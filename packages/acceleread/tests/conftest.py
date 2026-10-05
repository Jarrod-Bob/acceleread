# SPDX-License-Identifier: Apache-2.0
"""Keep every test out of the real `~/.acceleread` Workspace, and share a fake Classifier."""

import asyncio
from collections.abc import Callable, Mapping

import pytest

from acceleread.classifier import (
    Ask,
    Capabilities,
    Choice,
    ClassifierResponse,
    JSONState,
    JudgmentResult,
    Score,
)
from acceleread.models import ClassifierInfo


@pytest.fixture(autouse=True)
def isolated_workspace(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ACCELEREAD_HOME", str(tmp_path_factory.mktemp("home") / ".acceleread"))
    monkeypatch.delenv("ACCELEREAD_OFFLINE", raising=False)


class FakeClassifier:
    """A Classifier for runner tests: answers the first option, and can block or fail on demand."""

    def __init__(
        self,
        *,
        model: str = "fake-1",
        input_tokens: int = 100,
        token_budget: int = 32_000,
        gate: asyncio.Event | None = None,
        raises: Callable[[JSONState], Exception | None] | None = None,
    ) -> None:
        self.model = model
        self.input_tokens = input_tokens
        self.token_budget = token_budget
        self.gate = gate  # when set, judge() waits for it
        self.raises = raises
        self.calls: list[tuple[JSONState, list[str]]] = []

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            kinds=frozenset({"noul", "score", "choice"}),
            max_choice_options=255,
            token_budget=self.token_budget,
            chars_per_token=3.0,
            model=self.model,
            classifier_id="fake",
        )

    async def judge(self, state: JSONState, judgments: Mapping[str, Ask]) -> ClassifierResponse:
        self.calls.append((state, list(judgments)))
        if self.gate is not None:
            await self.gate.wait()
        if self.raises is not None and (error := self.raises(state)) is not None:
            raise error
        results: dict[str, JudgmentResult] = {}
        for name, ask in judgments.items():
            if isinstance(ask, Choice):
                first = next(iter(ask.options))
                results[name] = JudgmentResult("choice", first, {first: 1.0}, 1.0)
            elif isinstance(ask, Score):
                low = ask.criteria[0]
                results[name] = JudgmentResult("score", low, {low: 1.0}, 1.0)
            else:
                results[name] = JudgmentResult("noul", 1.0)
        return ClassifierResponse(
            results=results,
            info=ClassifierInfo(id="fake", model=self.model, version="1"),
            input_tokens=self.input_tokens,
            output_tokens=1,
        )


@pytest.fixture
def fake_classifier() -> Callable[..., FakeClassifier]:
    return FakeClassifier
