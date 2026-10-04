# SPDX-License-Identifier: Apache-2.0
"""The Jev adapter: Judgments out, results in, and SDK failures mapped to seam errors."""

import json
import logging
from collections.abc import Callable

import httpx2
import pytest
import typesafe_sdk as ts

from acceleread.classifier import (
    Choice,
    ClassifierRejected,
    ClassifierThrottled,
    ClassifierTransient,
    ClassifierUnavailable,
    Noul,
    Score,
)
from acceleread.jev import JevClassifier, configure_sdk_logging

Handler = Callable[[httpx2.Request], httpx2.Response]


def jev(handler: Handler) -> JevClassifier:
    client = ts.AsyncTypeSafeClient(
        api_key="test-key",
        transport=httpx2.MockTransport(handler),
        retry=ts.RetryPolicy(max_retries=0),
    )
    return JevClassifier(client=client)


def answering(answers: dict[str, object], seen: list[dict[str, object]] | None = None) -> Handler:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if seen is not None:
            seen.append(json.loads(request.content))
        body = {"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": 7}}
        return httpx2.Response(200, json=body)

    return handler


def test_capabilities_declare_jev_limits() -> None:
    caps = jev(answering({})).capabilities
    assert caps.kinds == {"noul", "score", "choice"}
    assert caps.max_choice_options == 255
    assert caps.chars_per_token == 3.0
    assert caps.token_budget == 32_000


async def test_judges_noul_score_and_choice_in_one_call() -> None:
    seen: list[dict[str, object]] = []
    answers = {
        "going_concern": {"type": "noul", "noul": 0.97},
        "risk": {
            "type": "score",
            "score": 1.6,
            "confidence": 0.8,
            "legend": {"0": "low", "1": "mid", "2": "high"},
            "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6},
        },
        "sector": {
            "type": "choice",
            "choice": "energy",
            "confidence": 0.9,
            "probabilities": {"energy": 0.9, "other": 0.1},
        },
    }
    response = await jev(answering(answers, seen)).judge(
        {"document": {"text": "hello"}},
        {
            "going_concern": Noul("Is there a going-concern warning?"),
            "risk": Score("How risky?", ("low", "mid", "high")),
            "sector": Choice("Which sector?", {"energy": "Energy", "other": None}),
        },
    )

    (request,) = seen
    assert request["state"] == {"document": {"text": "hello"}}
    assert request["model"] == "jev-1.13.0"
    questions = request["questions"]
    assert isinstance(questions, dict)
    assert {name: q["type"] for name, q in questions.items()} == {
        "going_concern": "noul",
        "risk": "score",
        "sector": "choice",
    }

    noul, score, choice = (response.results[n] for n in ("going_concern", "risk", "sector"))
    assert (noul.kind, noul.value, noul.probabilities, noul.confidence) == (
        "noul",
        0.97,
        None,
        None,
    )
    assert (score.kind, score.value, score.confidence) == ("score", 1.6, 0.8)
    assert score.probabilities == {"0": 0.1, "1": 0.3, "2": 0.6}
    assert (choice.kind, choice.value, choice.confidence) == ("choice", "energy", 0.9)
    assert choice.probabilities == {"energy": 0.9, "other": 0.1}
    assert response.info.id == "jev" and response.info.model == "jev-1.13.0"
    assert response.input_tokens == 7 and response.requests == 1


async def failing(status: int, headers: dict[str, str] | None = None) -> BaseException:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status, json={"error": {"message": "nope"}}, headers=headers)

    with pytest.raises(Exception) as caught:
        await jev(handler).judge({}, {"q": Noul("?")})
    return caught.value


async def test_429_is_throttling_with_retry_after() -> None:
    error = await failing(429, {"retry-after": "3"})
    assert isinstance(error, ClassifierThrottled) and error.retry_after == 3.0


async def test_429_without_retry_after_has_none() -> None:
    error = await failing(429)
    assert isinstance(error, ClassifierThrottled) and error.retry_after is None


async def test_529_is_throttling() -> None:
    assert isinstance(await failing(529, {"retry-after": "1"}), ClassifierThrottled)


@pytest.mark.parametrize("status", [500, 502, 503])
async def test_other_5xx_is_unavailable(status: int) -> None:
    assert isinstance(await failing(status), ClassifierUnavailable)


async def test_422_is_rejected() -> None:
    assert isinstance(await failing(422), ClassifierRejected)


async def test_connection_error_is_transient() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("boom", request=request)

    with pytest.raises(ClassifierTransient):
        await jev(handler).judge({}, {"q": Noul("?")})


async def test_other_4xx_propagates_unchanged() -> None:
    assert isinstance(await failing(400), ts.TypeSafeBadRequestError)


def test_sdk_debug_payload_logging_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    sdk_logger = logging.getLogger("typesafe_sdk")
    monkeypatch.delenv("ACCELEREAD_DEBUG_PAYLOADS", raising=False)
    sdk_logger.setLevel(logging.DEBUG)  # as TYPESAFE_LOG_LEVEL=debug would
    configure_sdk_logging()
    assert not sdk_logger.isEnabledFor(logging.DEBUG)


def test_sdk_debug_payload_logging_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    sdk_logger = logging.getLogger("typesafe_sdk")
    monkeypatch.setenv("ACCELEREAD_DEBUG_PAYLOADS", "1")
    sdk_logger.setLevel(logging.DEBUG)
    configure_sdk_logging()
    assert sdk_logger.isEnabledFor(logging.DEBUG)


def test_constructing_the_classifier_applies_the_logging_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk_logger = logging.getLogger("typesafe_sdk")
    monkeypatch.delenv("ACCELEREAD_DEBUG_PAYLOADS", raising=False)
    sdk_logger.setLevel(logging.DEBUG)
    JevClassifier()
    assert not sdk_logger.isEnabledFor(logging.DEBUG)
