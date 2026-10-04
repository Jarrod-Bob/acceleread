# SPDX-License-Identifier: Apache-2.0
"""The rate limiter around a Classifier (docs/spec/v0.md §7.5), on a fake clock."""

import asyncio
from collections.abc import Mapping

import pytest

from acceleread.classifier import (
    Ask,
    Capabilities,
    ClassifierRejected,
    ClassifierResponse,
    ClassifierThrottled,
    ClassifierTransient,
    ClassifierUnavailable,
    JSONState,
    JudgmentResult,
    Noul,
)
from acceleread.models import ClassifierInfo
from acceleread.ratelimit import (
    JEV_RATE_LIMIT,
    RateLimit,
    RateLimitedClassifier,
    priority_lane,
)

CAPS = Capabilities(
    kinds=frozenset({"noul"}), max_choice_options=255, token_budget=32_000, chars_per_token=3.0
)
FAST = RateLimit(tokens_per_s=1e12, requests_per_s=1e12)
ASK: Mapping[str, Ask] = {"q": Noul("?")}
# About 1000 estimated tokens: 3000 characters at 3.0 characters per token, plus a little overhead.
BIG_STATE: JSONState = {"text": "x" * 2980}


class FakeClock:
    """Sleeping advances time at once, so tests run instantly and deterministically."""

    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        self.t += max(seconds, 0.0)
        await asyncio.sleep(0)


class ScriptedClassifier:
    """Plays back a script of failures, then answers; records calls."""

    def __init__(self, script: list[Exception] | None = None, input_tokens: int | None = None):
        self.script = list(script or [])
        self.input_tokens = input_tokens
        self.calls: list[JSONState] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self.gate: asyncio.Event | None = None

    @property
    def capabilities(self) -> Capabilities:
        return CAPS

    async def judge(self, state: JSONState, judgments: Mapping[str, Ask]) -> ClassifierResponse:
        self.calls.append(state)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.gate is not None:
                await self.gate.wait()
            if self.script:
                raise self.script.pop(0)
            return ClassifierResponse(
                results={n: JudgmentResult(kind="noul", value=1.0) for n in judgments},
                info=ClassifierInfo(id="fake", model="m", version="1"),
                input_tokens=self.input_tokens,
            )
        finally:
            self.in_flight -= 1


def limited(
    inner: ScriptedClassifier, ceiling: RateLimit = FAST, clock: FakeClock | None = None
) -> tuple[RateLimitedClassifier, FakeClock]:
    clock = clock or FakeClock()
    return RateLimitedClassifier(inner, ceiling, clock=clock), clock


def test_jev_ceiling_is_80_percent_of_published_limits() -> None:
    assert JEV_RATE_LIMIT.tokens_per_s == pytest.approx(64_000)
    assert JEV_RATE_LIMIT.requests_per_s == pytest.approx(51.2)
    assert JEV_RATE_LIMIT.max_in_flight == 64


async def test_passes_results_and_capabilities_through() -> None:
    limiter, _ = limited(ScriptedClassifier())
    assert limiter.capabilities == CAPS
    response = await limiter.judge({"a": 1}, ASK)
    assert response.results["q"].value == 1.0


async def test_token_bucket_paces_requests_by_estimated_tokens() -> None:
    limiter, clock = limited(ScriptedClassifier(), RateLimit(2000, 1e9))
    for _ in range(5):
        await limiter.judge(BIG_STATE, ASK)
    # 5 x ~1000 tokens against a 2000-token burst, refilling at 2000/s: about 1.5 s.
    assert 1.4 <= clock.t <= 1.6


async def test_request_bucket_paces_requests_per_second() -> None:
    limiter, clock = limited(ScriptedClassifier(), RateLimit(1e12, 2))
    for _ in range(5):
        await limiter.judge({}, ASK)
    assert clock.t == pytest.approx(1.5)  # 2 in the burst, then one every 0.5 s


async def test_estimate_is_reconciled_with_reported_usage() -> None:
    cheap, clock = limited(ScriptedClassifier(input_tokens=100), RateLimit(2000, 1e9))
    for _ in range(5):
        await cheap.judge(BIG_STATE, ASK)
    assert clock.t < 0.1  # charged 100 each, not the ~1000 estimated

    costly, clock = limited(ScriptedClassifier(input_tokens=3000), RateLimit(2000, 1e9))
    for _ in range(3):
        await costly.judge(BIG_STATE, ASK)
    assert 2.4 <= clock.t <= 2.6  # the overshoot is paid back before the next request


async def test_at_most_64_requests_in_flight() -> None:
    inner = ScriptedClassifier()
    inner.gate = asyncio.Event()
    limiter, _ = limited(inner)
    tasks = [asyncio.create_task(limiter.judge({}, ASK)) for _ in range(70)]
    for _ in range(5):
        await asyncio.sleep(0)
    assert inner.max_in_flight == 64
    inner.gate.set()
    await asyncio.gather(*tasks)
    assert len(inner.calls) == 70 and inner.max_in_flight == 64


async def test_throttling_requeues_honours_retry_after_and_cuts_the_rate() -> None:
    inner = ScriptedClassifier([ClassifierThrottled(retry_after=5)])
    limiter, clock = limited(inner, RateLimit(1000, 100))
    response = await limiter.judge({}, ASK)
    assert response.results["q"].value == 1.0
    assert clock.t >= 5
    assert len(inner.calls) == 2
    assert limiter.rate == RateLimit(700, 70)
    assert limiter.ceiling == RateLimit(1000, 100)


async def test_throttling_without_retry_after_still_requeues() -> None:
    inner = ScriptedClassifier([ClassifierThrottled(), ClassifierThrottled()])
    limiter, clock = limited(inner, RateLimit(1000, 100))
    await limiter.judge({}, ASK)
    assert len(inner.calls) == 3 and clock.t > 0
    assert limiter.rate.tokens_per_s == pytest.approx(490)


async def test_rate_rises_5_percent_of_ceiling_per_quiet_minute() -> None:
    limiter, clock = limited(ScriptedClassifier([ClassifierThrottled(0)]), RateLimit(1000, 100))
    await limiter.judge({}, ASK)
    assert limiter.rate.tokens_per_s == pytest.approx(700)
    clock.t += 59
    assert limiter.rate.tokens_per_s == pytest.approx(700)
    clock.t += 1
    assert limiter.rate.tokens_per_s == pytest.approx(750)
    clock.t += 600
    assert limiter.rate == RateLimit(1000, 100)  # never above the ceiling


async def test_timeouts_and_connection_errors_requeue_without_cutting_the_rate() -> None:
    inner = ScriptedClassifier([ClassifierTransient()] * 8)
    limiter, _ = limited(inner, RateLimit(1000, 100))
    await limiter.judge({}, ASK)
    assert len(inner.calls) == 9
    assert limiter.rate == RateLimit(1000, 100)


async def test_persistent_5xx_fails_after_five_tries() -> None:
    inner = ScriptedClassifier([ClassifierUnavailable()] * 10)
    limiter, _ = limited(inner)
    with pytest.raises(ClassifierUnavailable):
        await limiter.judge({}, ASK)
    assert len(inner.calls) == 5


async def test_a_5xx_that_clears_up_succeeds() -> None:
    inner = ScriptedClassifier([ClassifierUnavailable()] * 4)
    limiter, _ = limited(inner)
    assert (await limiter.judge({}, ASK)).results["q"].value == 1.0
    assert len(inner.calls) == 5


async def test_rejection_is_not_retried() -> None:
    inner = ScriptedClassifier([ClassifierRejected("bad")])
    limiter, _ = limited(inner)
    with pytest.raises(ClassifierRejected):
        await limiter.judge({}, ASK)
    assert len(inner.calls) == 1


async def test_stalled_after_15_minutes_without_success_while_work_waits() -> None:
    inner = ScriptedClassifier()
    inner.gate = asyncio.Event()
    limiter, clock = limited(inner)
    assert not limiter.stalled
    task = asyncio.create_task(limiter.judge({}, ASK))
    await asyncio.sleep(0)
    clock.t += 14 * 60
    assert not limiter.stalled
    clock.t += 60
    assert limiter.stalled
    inner.gate.set()
    await task
    assert not limiter.stalled


async def test_idle_limiter_is_never_stalled() -> None:
    limiter, clock = limited(ScriptedClassifier())
    await limiter.judge({}, ASK)
    clock.t += 3600
    assert not limiter.stalled


async def test_priority_lane_jumps_queued_requests() -> None:
    inner = ScriptedClassifier()
    limiter, _ = limited(inner, RateLimit(1e12, 1))
    order: list[str] = []

    async def call(name: str) -> None:
        await limiter.judge({"name": name}, ASK)
        order.append(name)

    first = asyncio.create_task(call("first"))
    queued = [asyncio.create_task(call(f"batch{i}")) for i in range(3)]
    with priority_lane():
        urgent = asyncio.create_task(call("urgent"))
    await asyncio.gather(first, urgent, *queued)
    assert order[:2] == ["first", "urgent"]
    assert order[2:] == ["batch0", "batch1", "batch2"]
