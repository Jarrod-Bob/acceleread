# SPDX-License-Identifier: Apache-2.0
"""The Jev adapter: Judgments out, results in, and SDK failures mapped to seam errors."""

import json
import logging
from collections.abc import Callable, Iterator

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
from acceleread.jev import JEV_RATE_LIMIT, JevClassifier, configure_sdk_logging

Handler = Callable[[httpx2.Request], httpx2.Response]


@pytest.fixture
def sdk_logger() -> Iterator[logging.Logger]:
    """The SDK logger, with its level restored afterwards."""
    logger = logging.getLogger("typesafe_sdk")
    saved = logger.level
    yield logger
    logger.setLevel(saved)


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


def test_jev_ceiling_is_80_percent_of_published_limits() -> None:
    assert JEV_RATE_LIMIT.tokens_per_s == pytest.approx(64_000)
    assert JEV_RATE_LIMIT.requests_per_s == pytest.approx(51.2)
    assert JEV_RATE_LIMIT.max_in_flight == 64


def test_capabilities_declare_jev_limits() -> None:
    caps = jev(answering({})).capabilities
    assert caps.kinds == {"noul", "score", "choice"}
    assert caps.max_choice_options == 255
    assert caps.chars_per_token == 3.0
    assert caps.token_budget == 32_000


async def test_a_noul_with_low_probability_is_as_confident_as_its_complement() -> None:
    answers = {"q": {"type": "noul", "noul": 0.1}, "r": {"type": "noul", "noul": 0.5}}
    response = await jev(answering(answers)).judge(
        {"document": {}}, {"q": Noul("?"), "r": Noul("?")}
    )
    assert response.results["q"].confidence == pytest.approx(0.9)
    assert response.results["r"].confidence == 0.5


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
        0.97,  # max(p, 1 - p)
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


def test_sdk_debug_payload_logging_is_off_by_default(
    monkeypatch: pytest.MonkeyPatch, sdk_logger: logging.Logger
) -> None:
    monkeypatch.delenv("ACCELEREAD_DEBUG_PAYLOADS", raising=False)
    sdk_logger.setLevel(logging.DEBUG)  # as TYPESAFE_LOG_LEVEL=debug would
    configure_sdk_logging()
    assert not sdk_logger.isEnabledFor(logging.DEBUG)


def test_sdk_debug_payload_logging_opt_in_is_loud(
    monkeypatch: pytest.MonkeyPatch, sdk_logger: logging.Logger, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("ACCELEREAD_DEBUG_PAYLOADS", "1")
    sdk_logger.setLevel(logging.DEBUG)
    with caplog.at_level(logging.WARNING, logger="acceleread.jev"):
        configure_sdk_logging()
    assert sdk_logger.isEnabledFor(logging.DEBUG)
    (warning,) = [r for r in caplog.records if r.name == "acceleread.jev"]
    assert warning.levelno == logging.WARNING
    assert "ACCELEREAD_DEBUG_PAYLOADS" in warning.getMessage()


def test_constructing_the_classifier_applies_the_logging_policy(
    monkeypatch: pytest.MonkeyPatch, sdk_logger: logging.Logger
) -> None:
    monkeypatch.delenv("ACCELEREAD_DEBUG_PAYLOADS", raising=False)
    sdk_logger.setLevel(logging.DEBUG)
    JevClassifier()
    assert not sdk_logger.isEnabledFor(logging.DEBUG)


SECRET = "SECRET-DOCUMENT-TEXT"


async def test_seam_errors_never_carry_request_or_server_text() -> None:
    def echo(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(422, json={"error": {"message": f"bad {SECRET}"}})

    with pytest.raises(ClassifierRejected) as rejected:
        await jev(echo).judge({"text": SECRET}, {"q": Noul(SECRET)})
    assert SECRET not in str(rejected.value) and "422" in str(rejected.value)

    def broken(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError(f"cannot send {SECRET}", request=request)

    with pytest.raises(ClassifierTransient) as transient:
        await jev(broken).judge({"text": SECRET}, {"q": Noul("?")})
    assert SECRET not in str(transient.value)


async def test_timeout_is_transient() -> None:
    def slow(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("slow", request=request)

    with pytest.raises(ClassifierTransient):
        await jev(slow).judge({}, {"q": Noul("?")})


async def test_529_carries_retry_after_end_to_end() -> None:
    error = await failing(529, {"retry-after": "2.5"})
    assert isinstance(error, ClassifierThrottled) and error.retry_after == 2.5


async def test_sdk_never_retries_underneath_the_limiter() -> None:
    calls = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        return httpx2.Response(500, json={})

    # An injected client with the SDK's default retry policy (2 retries).
    client = ts.AsyncTypeSafeClient(api_key="k", transport=httpx2.MockTransport(handler))
    with pytest.raises(ClassifierUnavailable):
        await JevClassifier(client=client).judge({}, {"q": Noul("?")})
    assert calls == 1


async def test_lazily_created_client_does_not_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        return httpx2.Response(500, json={})

    real = ts.AsyncTypeSafeClient

    def factory(**kwargs: object) -> ts.AsyncTypeSafeClient:
        # Default retry policy, so the adapter must be the one switching retries off.
        return real(api_key="k", transport=httpx2.MockTransport(handler))

    monkeypatch.setattr("acceleread.jev.ts.AsyncTypeSafeClient", factory)
    with pytest.raises(ClassifierUnavailable):
        await JevClassifier().judge({}, {"q": Noul("?")})
    assert calls == 1
