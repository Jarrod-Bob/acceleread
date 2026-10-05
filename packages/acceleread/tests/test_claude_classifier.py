# SPDX-License-Identifier: Apache-2.0
"""The Claude Classifier behind `[llm]`, driven through the real SDK over a mock transport.

No test touches the network: the responses below are hand-written in the Messages API's shape.
"""

import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest

anthropic = pytest.importorskip("anthropic")  # the [llm] extra; CI has a leg for it

from acceleread.classifier import (
    Choice,
    ClassifierRefused,
    ClassifierRejected,
    ClassifierThrottled,
    ClassifierTokensExceeded,
    ClassifierTransient,
    ClassifierUnavailable,
    Noul,
    Score,
)
from acceleread.claude import ClaudeClassifier, claude_rate_limit
from acceleread.escalation import EscalationBudget, escalate
from acceleread.models import ClassifierInfo, Coverage, Judgment, Question, Taxonomy
from acceleread.planner import DocumentView, Outcome, judgment_specs
from acceleread.ratelimit import RateLimit, RateLimitedClassifier

Handler = Callable[[httpx2.Request], httpx2.Response]
STATE = {"document": {"title": "Acme", "text": "Revenue fell sharply."}}
JUDGMENTS = {
    "q_going_concern": Noul("Is there going-concern doubt in `document`?"),
    "q_risk": Score("How risky is `document`?", ("low", "medium", "high")),
    "taxonomy": Choice("Which sector?", {"bank": "A bank", "tech": None}),
}


def message(content: dict[str, Any] | str, **overrides: Any) -> dict[str, Any]:
    text = content if isinstance(content, str) else json.dumps(content)
    body: dict[str, Any] = {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 120, "output_tokens": 14},
    }
    body.update(overrides)
    return body


ANSWERS = {"q_going_concern": True, "q_risk": "high", "taxonomy": "tech"}


def claude(handler: Handler, **kwargs: Any) -> ClaudeClassifier:
    client = anthropic.AsyncAnthropic(
        api_key="test-key",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    return ClaudeClassifier(client=client, **kwargs)


def replying(
    body: dict[str, Any],
    seen: list[httpx2.Request] | None = None,
    status: int = 200,
    headers: dict[str, str] | None = None,
) -> Handler:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if seen is not None:
            seen.append(request)
        return httpx2.Response(status, json=body, headers=headers)

    return handler


def error_body(kind: str, text: str = "nope") -> dict[str, Any]:
    return {"type": "error", "error": {"type": kind, "message": text}}


def test_it_declares_its_capabilities() -> None:
    caps = claude(replying({})).capabilities
    assert (caps.model, caps.classifier_id) == ("claude-opus-5-5", "claude")
    assert caps.kinds == {"noul", "score", "choice"}
    assert caps.token_budget > 100_000  # far above Jev's 32k: escalated Sections go untruncated


async def test_one_call_asks_every_judgment_with_structured_output_at_low_effort() -> None:
    seen: list[httpx2.Request] = []
    await claude(replying(message(ANSWERS), seen)).judge(STATE, JUDGMENTS)
    assert len(seen) == 1
    request = seen[0]
    body = json.loads(request.content)
    assert body["model"] == "claude-opus-5-5"
    assert body["output_config"]["effort"] == "low"
    schema = body["output_config"]["format"]["schema"]
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert set(schema["required"]) == set(JUDGMENTS)
    assert schema["additionalProperties"] is False
    props = schema["properties"]
    assert props["q_going_concern"]["type"] == "boolean"
    assert props["q_risk"]["enum"] == ["low", "medium", "high"]
    assert props["taxonomy"]["enum"] == ["bank", "tech"]
    prompt = json.dumps(body["messages"])
    assert "Revenue fell sharply." in prompt
    assert "going-concern doubt" in prompt
    assert "A bank" in prompt


async def test_refusals_fall_back_server_side_by_default() -> None:
    seen: list[httpx2.Request] = []
    await claude(replying(message(ANSWERS), seen)).judge(STATE, JUDGMENTS)
    body = json.loads(seen[0].content)
    assert body["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in seen[0].headers["anthropic-beta"]


async def test_answers_become_noul_score_and_choice_results() -> None:
    response = await claude(replying(message(ANSWERS))).judge(STATE, JUDGMENTS)
    results = response.results
    assert (results["q_going_concern"].kind, results["q_going_concern"].value) == ("noul", 1.0)
    assert (results["q_risk"].kind, results["q_risk"].value) == ("score", 2.0)  # index of "high"
    assert (results["taxonomy"].kind, results["taxonomy"].value) == ("choice", "tech")
    assert all(r.probabilities is None and r.confidence is None for r in results.values())


async def test_a_false_noul_is_zero() -> None:
    answers = {**ANSWERS, "q_going_concern": False}
    response = await claude(replying(message(answers))).judge(STATE, JUDGMENTS)
    assert response.results["q_going_concern"].value == 0.0


async def test_usage_and_the_serving_model_are_reported() -> None:
    body = message(
        ANSWERS,
        model="claude-opus-4-8",  # a refusal fallback served this one
        usage={
            "input_tokens": 100,
            "cache_read_input_tokens": 30,
            "cache_creation_input_tokens": 20,
            "output_tokens": 14,
        },
    )
    response = await claude(replying(body)).judge(STATE, JUDGMENTS)
    assert response.info.id == "claude"
    assert response.info.model == "claude-opus-4-8"
    assert (response.input_tokens, response.output_tokens, response.requests) == (150, 14, 1)


async def test_a_refusal_is_an_error_not_an_answer() -> None:
    refusal = message("", stop_reason="refusal", stop_details={"type": "refusal", "category": None})
    with pytest.raises(ClassifierRefused):
        await claude(replying(refusal)).judge(STATE, JUDGMENTS)


@pytest.mark.parametrize(
    ("status", "kind", "expected"),
    [
        (429, "rate_limit_error", ClassifierThrottled),
        (529, "overloaded_error", ClassifierThrottled),
        (500, "api_error", ClassifierUnavailable),
        (503, "api_error", ClassifierUnavailable),
        (422, "invalid_request_error", ClassifierRejected),
        (401, "authentication_error", ClassifierRejected),
        (403, "permission_error", ClassifierRejected),
        (404, "not_found_error", ClassifierRejected),
        (400, "invalid_request_error", ClassifierRejected),
        (413, "request_too_large", ClassifierTokensExceeded),
    ],
)
async def test_http_failures_map_to_seam_errors(
    status: int, kind: str, expected: type[Exception]
) -> None:
    calls: list[httpx2.Request] = []
    classifier = claude(replying(error_body(kind), calls, status=status))
    with pytest.raises(expected):
        await classifier.judge(STATE, JUDGMENTS)
    assert len(calls) == 1  # the SDK never retries underneath the rate limiter


async def test_an_unknown_error_body_is_still_a_seam_error() -> None:
    handler = replying({"unexpected": True}, status=418)
    with pytest.raises(ClassifierRejected):
        await claude(handler).judge(STATE, JUDGMENTS)


async def test_a_prompt_over_the_token_limit_is_tokens_exceeded() -> None:
    body = error_body("invalid_request_error", "prompt is too long: 1100000 tokens > 1000000")
    with pytest.raises(ClassifierTokensExceeded):
        await claude(replying(body, status=400)).judge(STATE, JUDGMENTS)


async def test_retry_after_is_passed_on() -> None:
    handler = replying(error_body("rate_limit_error"), status=429, headers={"retry-after": "7"})
    with pytest.raises(ClassifierThrottled) as caught:
        await claude(handler).judge(STATE, JUDGMENTS)
    assert caught.value.retry_after == 7.0


async def test_timeouts_and_connection_errors_are_transient() -> None:
    def boom(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("down", request=request)

    with pytest.raises(ClassifierTransient):
        await claude(boom).judge(STATE, JUDGMENTS)

    def slow(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("slow", request=request)

    with pytest.raises(ClassifierTransient):
        await claude(slow).judge(STATE, JUDGMENTS)


RATE_HEADERS = {
    "anthropic-ratelimit-requests-limit": "1000",
    "anthropic-ratelimit-input-tokens-limit": "2000000",
    "anthropic-ratelimit-tokens-limit": "2400000",
}


def test_the_ceiling_is_80_percent_of_the_published_per_minute_limits() -> None:
    limit = claude_rate_limit(RATE_HEADERS)
    assert limit is not None
    assert limit.tokens_per_s == pytest.approx(2_000_000 / 60 * 0.8)
    assert limit.requests_per_s == pytest.approx(1000 / 60 * 0.8)


def test_the_combined_tokens_limit_is_not_mistaken_for_input_tokens() -> None:
    headers = {k: v for k, v in RATE_HEADERS.items() if "input-tokens" not in k}
    assert claude_rate_limit(headers) is None


def test_without_the_headers_there_is_no_ceiling() -> None:
    assert claude_rate_limit({}) is None
    assert claude_rate_limit({"anthropic-ratelimit-requests-limit": "x"}) is None


async def test_rate_limit_headers_are_read_from_responses_and_from_errors() -> None:
    seen: list[RateLimit] = []
    ok = claude(replying(message(ANSWERS), headers=RATE_HEADERS), on_rate_limit=seen.append)
    await ok.judge(STATE, JUDGMENTS)
    throttled = claude(
        replying(error_body("rate_limit_error"), status=429, headers=RATE_HEADERS),
        on_rate_limit=seen.append,
    )
    with pytest.raises(ClassifierThrottled):
        await throttled.judge(STATE, JUDGMENTS)
    assert len(seen) == 2
    assert seen[0].tokens_per_s == pytest.approx(2_000_000 / 60 * 0.8)


async def test_the_limiter_adopts_the_ceiling_the_headers_announce() -> None:
    inner = claude(replying(message(ANSWERS), headers=RATE_HEADERS))
    limiter = RateLimitedClassifier(inner, RateLimit(tokens_per_s=10, requests_per_s=1))
    inner.on_rate_limit = limiter.update_ceiling
    await limiter.judge(STATE, JUDGMENTS)
    assert limiter.ceiling.tokens_per_s == pytest.approx(2_000_000 / 60 * 0.8)


async def test_a_user_set_rate_limit_is_never_raised_by_the_headers() -> None:
    mine = RateLimit(tokens_per_s=100, requests_per_s=1)
    inner = claude(replying(message(ANSWERS), headers=RATE_HEADERS))
    limiter = RateLimitedClassifier(inner, mine, user_set=True)
    inner.on_rate_limit = limiter.update_ceiling
    await limiter.judge(STATE, JUDGMENTS)
    assert limiter.ceiling == mine
    limiter.update_ceiling(RateLimit(tokens_per_s=10, requests_per_s=5))
    assert limiter.ceiling == RateLimit(tokens_per_s=10, requests_per_s=1)  # lower is honoured


async def test_a_flagged_judgment_escalates_through_the_planner_and_this_classifier() -> None:
    body = message({"q_going_concern": True, "taxonomy": "tech"})
    seen: list[httpx2.Request] = []
    view = DocumentView(text="Revenue fell sharply. " * 50, title="Acme")
    taxonomy = Taxonomy(name="t", categories=[{"name": "bank"}, {"name": "tech"}]).with_other()  # type: ignore[list-item]
    questions = [
        Question(name="going_concern", kind="noul", instructions="Doubt?", escalate_below=0.9)
    ]
    specs = judgment_specs(taxonomy, questions)
    first = Judgment(
        kind="choice",
        value="bank",
        probabilities={"bank": 0.5, "tech": 0.5},
        confidence=0.5,
        classifier=ClassifierInfo(id="jev", model="jev-1.13.0", version="1"),
        coverage=Coverage(est_tokens=5),
    )
    outcome = Outcome(answers={"going_concern": first})
    await escalate(
        view,
        specs,
        outcome,
        classifier=claude(replying(body, seen)),
        budget=EscalationBudget(total_documents=100, escalation_max=1.0),
    )
    assert outcome.errors == []
    done = outcome.answers["going_concern"]
    assert isinstance(done, Judgment)
    assert (done.escalation.status, done.value, done.confidence) == ("escalated", 1.0, None)
    assert done.classifier.id == "claude"
    assert len(seen) == 1
