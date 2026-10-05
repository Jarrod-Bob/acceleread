# SPDX-License-Identifier: Apache-2.0
"""Keep every test out of the real `~/.acceleread` Workspace, and share a fake Classifier."""

import asyncio
from collections.abc import Callable, Mapping
from typing import Any

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
        self.in_flight = 0
        self.max_in_flight = 0

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
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.gate is not None:
                await self.gate.wait()
        finally:
            self.in_flight -= 1
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


class FakeClock:
    """A clock the test moves by hand, shared by a Workspace (lease) and a Runner (heartbeat)."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.t = start

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds

    async def sleep(self, seconds: float) -> None:
        """Time passes instantly: advance the clock and let other tasks run."""
        self.t += seconds
        await asyncio.sleep(0.001)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


async def _until(condition: Callable[[], bool], timeout: float = 10.0) -> None:
    """Wait for something to become true, polling quickly; fail rather than hang."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.005)


@pytest.fixture
def until() -> Callable[..., Any]:
    return _until
