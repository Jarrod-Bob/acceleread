# SPDX-License-Identifier: Apache-2.0
"""Rate limiting and failure policy around a Classifier (docs/spec/v0.md §7.5).

`RateLimitedClassifier` wraps any Classifier. Per Classifier per runner it keeps a dual token
bucket (tokens/s charged by estimate and reconciled with reported usage, plus requests/s), caps
requests in flight, backs off with AIMD on throttling, and applies the failure policy. Nothing
here logs or stores Document text or Classifier state.
"""

import asyncio
import contextvars
import itertools
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol

from acceleread.classifier import (
    Ask,
    Capabilities,
    Classifier,
    ClassifierResponse,
    ClassifierThrottled,
    ClassifierTransient,
    ClassifierUnavailable,
    JSONState,
    estimate_tokens,
)

MAX_IN_FLIGHT = 64
CEILING_FRACTION = 0.8  # of the published limits
DECREASE_FACTOR = 0.7  # AIMD: multiplicative decrease on 429 or 529
INCREASE_STEP = 0.05  # AIMD: additive increase, as a fraction of the ceiling
INCREASE_INTERVAL = 60.0  # seconds without a 429 before each increase
MIN_FACTOR = 0.05  # the rate never falls below this fraction of the ceiling
MAX_UNAVAILABLE_TRIES = 5
STALL_AFTER = 15 * 60.0  # seconds without a success while work waits
BACKOFF_INITIAL = 0.5
BACKOFF_MAX = 5.0
_EPSILON = 1e-9


@dataclass(frozen=True)
class RateLimit:
    tokens_per_s: float
    requests_per_s: float
    max_in_flight: int = MAX_IN_FLIGHT


def ceiling_for(published: RateLimit) -> RateLimit:
    """80% of a Classifier's published limits (the default ceiling)."""
    return RateLimit(
        published.tokens_per_s * CEILING_FRACTION,
        published.requests_per_s * CEILING_FRACTION,
        published.max_in_flight,
    )


class Clock(Protocol):
    def now(self) -> float: ...

    async def sleep(self, seconds: float) -> None: ...


class MonotonicClock:
    def now(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


_priority: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "acceleread_priority", default=False
)


@contextmanager
def priority_lane() -> Iterator[None]:
    """Requests made inside this block (single-Document ingests) jump the limiter's queue."""
    token = _priority.set(True)
    try:
        yield
    finally:
        _priority.reset(token)


def _backoff(attempt: int) -> float:
    return min(BACKOFF_INITIAL * (2.0**attempt), BACKOFF_MAX)


class RateLimitedClassifier:
    def __init__(
        self,
        inner: Classifier,
        ceiling: RateLimit,
        *,
        clock: Clock | None = None,
        max_unavailable_tries: int = MAX_UNAVAILABLE_TRIES,
    ) -> None:
        self._inner = inner
        self._ceiling = ceiling
        self._clock: Clock = clock or MonotonicClock()
        self._max_unavailable_tries = max_unavailable_tries
        now = self._clock.now()
        self._factor = 1.0
        self._last_event = now  # last 429 or AIMD increase
        self._epoch = 0  # bumped on each rate cut, so one throttling episode cuts once
        self._blocked_until = 0.0  # retry-after: nothing is admitted before this
        self._tokens = ceiling.tokens_per_s
        self._requests = max(1.0, ceiling.requests_per_s)
        self._refilled_at = now
        self._in_flight = 0
        self._active = 0  # judge() calls underway, including ones sleeping between retries
        self._waiting: set[tuple[int, int]] = set()
        self._sequence = itertools.count()
        self._parked: list[asyncio.Future[None]] = []
        self._progress_at = now  # last success, or when work arrived at an idle limiter

    @property
    def capabilities(self) -> Capabilities:
        return self._inner.capabilities

    @property
    def ceiling(self) -> RateLimit:
        return self._ceiling

    def update_ceiling(self, ceiling: RateLimit) -> None:
        """Adopt a new ceiling, e.g. the limits a Classifier announces in its response headers.

        The AIMD factor is kept, so a backoff in progress survives the update.
        """
        self._advance()
        self._ceiling = ceiling
        self._credit(0)  # clamp the buckets to the new rate

    @property
    def rate(self) -> RateLimit:
        """The effective rate right now, never above the ceiling."""
        self._advance()
        return self._current()

    @property
    def stalled(self) -> bool:
        """True after 15 minutes without a success while requests are waiting or in flight."""
        return self._active > 0 and self._clock.now() - self._progress_at >= STALL_AFTER

    def _current(self) -> RateLimit:
        c = self._ceiling
        return RateLimit(
            c.tokens_per_s * self._factor, c.requests_per_s * self._factor, c.max_in_flight
        )

    def _advance(self) -> None:
        """Apply AIMD increases and refill both buckets up to now."""
        now = self._clock.now()
        while self._factor < 1.0 and now - self._last_event >= INCREASE_INTERVAL:
            self._factor = min(1.0, self._factor + INCREASE_STEP)
            self._last_event += INCREASE_INTERVAL
        rate = self._current()
        elapsed = max(0.0, now - self._refilled_at)
        self._tokens = min(rate.tokens_per_s, self._tokens + elapsed * rate.tokens_per_s)
        self._requests = min(
            max(1.0, rate.requests_per_s), self._requests + elapsed * rate.requests_per_s
        )
        self._refilled_at = now

    def _credit(self, tokens: float) -> None:
        """Add (or, if negative, charge) tokens, up to one second's burst."""
        self._advance()
        self._tokens = min(self._current().tokens_per_s, self._tokens + tokens)

    def _wake_all(self) -> None:
        parked, self._parked = self._parked, []
        for future in parked:
            if not future.done():
                future.set_result(None)

    async def _park(self) -> None:
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._parked.append(future)
        await future

    async def _admit(self, cost: float, priority: bool) -> None:
        """Wait for this request's turn: priority first, then arrival order."""
        key = (0 if priority else 1, next(self._sequence))
        self._waiting.add(key)
        try:
            while True:
                self._advance()
                if key != min(self._waiting):
                    await self._park()
                    continue
                now = self._clock.now()
                if now < self._blocked_until:
                    await self._clock.sleep(self._blocked_until - now)
                    continue
                rate = self._current()
                if self._in_flight >= rate.max_in_flight:
                    await self._park()
                    continue
                need = min(cost, rate.tokens_per_s)
                delay = max(
                    (need - self._tokens) / rate.tokens_per_s,
                    (1.0 - self._requests) / rate.requests_per_s,
                )
                if delay > _EPSILON:
                    await self._clock.sleep(delay)
                    continue
                self._tokens -= cost
                self._requests -= 1.0
                return
        finally:
            self._waiting.discard(key)
            self._wake_all()

    def _throttled(self, epoch: int, error: ClassifierThrottled, attempt: int) -> None:
        now = self._clock.now()
        if epoch == self._epoch:  # one cut per episode, however many requests it hit
            self._advance()
            self._factor = max(MIN_FACTOR, self._factor * DECREASE_FACTOR)
            self._last_event = now
            self._epoch += 1
            self._credit(0)  # clamp the bucket to the lowered rate
        delay = error.retry_after if error.retry_after is not None else _backoff(attempt)
        self._blocked_until = max(self._blocked_until, now + delay)

    def _reconcile(self, estimated: float, actual: int | None) -> None:
        self._progress_at = self._clock.now()
        if actual is not None:
            self._credit(estimated - actual)

    async def judge(self, state: JSONState, judgments: Mapping[str, Ask]) -> ClassifierResponse:
        if self._active == 0:  # genuinely new work, not a retry
            self._progress_at = self._clock.now()
        self._active += 1
        try:
            return await self._judge_with_retries(state, judgments)
        finally:
            self._active -= 1

    async def _send(
        self, state: JSONState, judgments: Mapping[str, Ask], cost: float
    ) -> ClassifierResponse:
        """One attempt. A failed attempt is not charged; a success is reconciled with usage."""
        self._in_flight += 1
        try:
            response = await self._inner.judge(state, judgments)
        except BaseException:
            self._credit(cost)
            raise
        else:
            self._reconcile(cost, response.input_tokens)
            return response
        finally:
            self._in_flight -= 1
            self._wake_all()

    async def _judge_with_retries(
        self, state: JSONState, judgments: Mapping[str, Ask]
    ) -> ClassifierResponse:
        priority = _priority.get()
        cost = estimate_tokens(state, judgments, self.capabilities)
        throttles = transients = unavailable = 0
        while True:
            await self._admit(cost, priority)
            epoch = self._epoch
            try:
                return await self._send(state, judgments, cost)
            except ClassifierThrottled as error:
                self._throttled(epoch, error, throttles)
                throttles += 1
            except ClassifierTransient:
                await self._clock.sleep(_backoff(transients))
                transients += 1
            except ClassifierUnavailable:
                unavailable += 1
                if unavailable >= self._max_unavailable_tries:
                    raise
                await self._clock.sleep(_backoff(unavailable - 1))
