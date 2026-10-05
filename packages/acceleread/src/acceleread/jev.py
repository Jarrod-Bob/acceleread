# SPDX-License-Identifier: Apache-2.0
"""Jev, the default Classifier, over TypeSafe's async SDK (docs/spec/v0.md §5.4)."""

import logging
import os
from collections.abc import Mapping
from dataclasses import replace
from importlib.metadata import version
from typing import cast

import typesafe_sdk as ts

from acceleread.classifier import (
    Ask,
    Capabilities,
    Choice,
    ClassifierError,
    ClassifierRejected,
    ClassifierResponse,
    ClassifierThrottled,
    ClassifierTokensExceeded,
    ClassifierTransient,
    ClassifierUnavailable,
    JSONState,
    JudgmentResult,
    Noul,
    Score,
)
from acceleread.models import DEFAULT_JEV_MODEL, ClassifierInfo
from acceleread.ratelimit import RateLimit, ceiling_for

logger = logging.getLogger(__name__)

JEV_CAPABILITIES = Capabilities(
    kinds=frozenset({"noul", "score", "choice"}),
    max_choice_options=255,
    token_budget=32_000,
    chars_per_token=3.0,
    # Measured live: a 2-page Document estimated 286 tokens against 506 actual.
    request_overhead_tokens=220,
)

# Published limits (docs/spec/v0.md §7.5); the default ceiling is 80% of them.
JEV_PUBLISHED_LIMIT = RateLimit(tokens_per_s=80_000, requests_per_s=64)
JEV_RATE_LIMIT = ceiling_for(JEV_PUBLISHED_LIMIT)

# The rate limiter owns retries, so the SDK must never retry underneath it.
NO_SDK_RETRIES = ts.RetryPolicy(max_retries=0)


def configure_sdk_logging() -> None:
    """Keep the SDK's DEBUG logs, which carry request payloads, off unless opted in.

    The SDK enables them itself when TYPESAFE_LOG_LEVEL=debug, so clamp that unless
    ACCELEREAD_DEBUG_PAYLOADS=1.
    """
    sdk_logger = logging.getLogger("typesafe_sdk")
    if os.environ.get("ACCELEREAD_DEBUG_PAYLOADS") == "1":
        logger.warning(
            "ACCELEREAD_DEBUG_PAYLOADS=1: SDK debug logging is ON and may print Document text "
            "and Classifier state. Do not use this with real data."
        )
    elif sdk_logger.isEnabledFor(logging.DEBUG):
        sdk_logger.setLevel(logging.INFO)


def _retry_after_header(error: ts.TypeSafeAPIError) -> float | None:
    raw = error.headers.get("retry-after")
    try:
        return None if raw is None else max(0.0, float(raw))
    except ValueError:
        return None


def _is_max_tokens_exceeded(body: object) -> bool:
    detail = body.get("detail") if isinstance(body, dict) else None
    return isinstance(detail, dict) and detail.get("error_type") == "max_tokens_exceeded"


def _translate(error: ts.TypeSafeError) -> ClassifierError | None:
    """Map an SDK failure to a seam error; None means it propagates unchanged."""
    if isinstance(error, ts.TypeSafeRateLimitError):
        retry_after = None if error.retry_after_ms is None else error.retry_after_ms / 1000
        return ClassifierThrottled(retry_after)
    if isinstance(error, ts.TypeSafeAPIConnectionError):  # includes timeouts
        return ClassifierTransient("classifier connection failed or timed out")
    if isinstance(error, ts.TypeSafeAPIError):
        if error.status == 529:
            return ClassifierThrottled(_retry_after_header(error))
        if error.status >= 500:
            return ClassifierUnavailable(f"classifier returned {error.status}")
        if error.status == 400 and _is_max_tokens_exceeded(error.body):
            return ClassifierTokensExceeded("state over the token limit")
        if error.status == 422:
            return ClassifierRejected(f"classifier rejected the request ({error.status})")
    return None


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
        configure_sdk_logging()

    @property
    def capabilities(self) -> Capabilities:
        return replace(JEV_CAPABILITIES, model=self.model)

    def _get_client(self) -> ts.AsyncTypeSafeClient:
        if self._client is None:  # created lazily so a missing key fails at first use, not import
            self._client = ts.AsyncTypeSafeClient(retry=NO_SDK_RETRIES)
        return self._client

    async def judge(self, state: JSONState, judgments: Mapping[str, Ask]) -> ClassifierResponse:
        questions = {name: _question(ask) for name, ask in judgments.items()}
        # Same JSON shape as the SDK's own alias, which mypy can't match structurally.
        state_json = cast(ts.JSONContent, dict(state))
        try:
            response = await self._get_client().system_one(
                state_json, questions, model=self.model, retry=NO_SDK_RETRIES
            )
        except ts.TypeSafeError as error:
            translated = _translate(error)
            if translated is None:
                raise
            raise translated from error
        return ClassifierResponse(
            results={name: _result(answer) for name, answer in response.answers.items()},
            info=ClassifierInfo(id="jev", model=response.model, version=version("typesafe-sdk")),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
