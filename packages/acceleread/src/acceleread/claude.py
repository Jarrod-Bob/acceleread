# SPDX-License-Identifier: Apache-2.0
"""Claude as a Classifier, behind the `[llm]` extra (docs/spec/v0.md §5.4-§5.5).

One structured-output call answers every Judgment of a request. The default is `claude-opus-5-5`
at low effort, sent with the server-side refusal `fallbacks` parameter. Nothing here logs Document
text or Classifier state.
"""

import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from importlib.metadata import version
from typing import Any, Literal

import anthropic
from anthropic.types.beta import BetaOutputConfigParam

from acceleread.classifier import (
    Ask,
    Capabilities,
    Choice,
    ClassifierError,
    ClassifierRefused,
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
    ask_definition,
)
from acceleread.models import DEFAULT_ESCALATION_MODEL, ClassifierInfo
from acceleread.ratelimit import CEILING_FRACTION, RateLimit

CLAUDE_CAPABILITIES = Capabilities(
    kinds=frozenset({"noul", "score", "choice"}),
    max_choice_options=255,
    token_budget=900_000,  # the 1M context, less room for the answer and a safety margin
    chars_per_token=3.0,  # conservative; the newer tokenizers run up to ~1.35x older ones
    request_overhead_tokens=500,  # the system prompt and the output schema
)
type Effort = Literal["low", "medium", "high", "xhigh", "max"]
EFFORT: Effort = "low"
MAX_TOKENS = 16_000  # thinking at low effort plus a few short answers; non-streaming safe
# "fallbacks" here is the API's server-side refusal fallback (another model answers a request the
# first declined), not acceleread's Fallback: head+tail reads, or Escalation (see CONTEXT.md).
FALLBACK_BETA = "server-side-fallback-2026-07-01"
SYSTEM_PROMPT = (
    "You answer questions about a document. The document is the JSON under `document`. "
    "Answer every question under `questions`, using only the document, and reply with the "
    "JSON object the output schema asks for: one key per question."
)
_TOO_LONG = "too long"


def _int_header(headers: Mapping[str, str], name: str) -> int | None:
    try:
        value = int(headers[name])
    except (KeyError, ValueError):
        return None
    return value if value > 0 else None


def claude_rate_limit(headers: Mapping[str, str]) -> RateLimit | None:
    """The default ceiling (80% of published limits) from Anthropic's rate-limit headers.

    The headers give per-minute limits. None when they are absent or unreadable.
    """
    requests = _int_header(headers, "anthropic-ratelimit-requests-limit")
    # Input tokens only: the combined `tokens-limit` also counts output, so it is no substitute.
    tokens = _int_header(headers, "anthropic-ratelimit-input-tokens-limit")
    if requests is None or tokens is None:
        return None
    return RateLimit(
        tokens_per_s=tokens / 60 * CEILING_FRACTION,
        requests_per_s=requests / 60 * CEILING_FRACTION,
    )


def _schema(judgments: Mapping[str, Ask]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for name, ask in judgments.items():
        match ask:
            case Noul():
                properties[name] = {"type": "boolean"}
            case Score(criteria=criteria):
                properties[name] = {"type": "string", "enum": list(criteria)}
            case Choice(options=options):
                properties[name] = {"type": "string", "enum": list(options)}
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _prompt(state: JSONState, judgments: Mapping[str, Ask]) -> str:
    questions = {name: ask_definition(ask) for name, ask in judgments.items()}
    return json.dumps({"document": state["document"], "questions": questions}, ensure_ascii=False)


def _result(ask: Ask, answer: object) -> JudgmentResult:
    match ask:
        case Noul():
            return JudgmentResult(kind="noul", value=1.0 if answer else 0.0)
        case Score(criteria=criteria):
            return JudgmentResult(kind="score", value=float(criteria.index(str(answer))))
        case Choice():
            return JudgmentResult(kind="choice", value=str(answer))


def _answers(response: Any, judgments: Mapping[str, Ask]) -> dict[str, JudgmentResult]:
    if response.stop_reason == "refusal":
        raise ClassifierRefused("the model declined the request")
    if response.stop_reason == "max_tokens":
        raise ClassifierError("the answer was cut off at max_tokens")
    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ClassifierError("the answer was not valid JSON") from exc
    results: dict[str, JudgmentResult] = {}
    for name, ask in judgments.items():
        if isinstance(raw, dict) and name in raw:
            try:
                results[name] = _result(ask, raw[name])
            except ValueError:  # an answer outside the options; the Planner reports it missing
                continue
    return results


class ClaudeClassifier:
    def __init__(
        self,
        model: str = DEFAULT_ESCALATION_MODEL,
        client: anthropic.AsyncAnthropic | None = None,
        *,
        effort: Effort = EFFORT,
        on_rate_limit: Callable[[RateLimit], None] | None = None,
    ) -> None:
        self.model = model
        self.effort = effort
        self.on_rate_limit = on_rate_limit  # told the ceiling Anthropic's headers announce
        self._client = client

    @property
    def capabilities(self) -> Capabilities:
        return replace(CLAUDE_CAPABILITIES, model=self.model, classifier_id="claude")

    def _get_client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:  # lazily, so a missing key fails at first use, not at import
            self._client = anthropic.AsyncAnthropic()
        return self._client

    def _observe(self, headers: Mapping[str, str]) -> None:
        limit = claude_rate_limit(headers)
        if limit is not None and self.on_rate_limit is not None:
            self.on_rate_limit(limit)

    async def judge(self, state: JSONState, judgments: Mapping[str, Ask]) -> ClassifierResponse:
        try:
            config: BetaOutputConfigParam = {
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": _schema(judgments)},
            }
            # The rate limiter owns retries, so the SDK must never retry underneath it.
            client = self._get_client().with_options(max_retries=0)
            raw = await client.beta.messages.with_raw_response.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                betas=[FALLBACK_BETA],
                fallbacks="default",
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": _prompt(state, judgments)}],
                output_config=config,
            )
        except anthropic.APIConnectionError as error:  # includes timeouts
            raise ClassifierTransient("classifier connection failed or timed out") from error
        except anthropic.APIStatusError as error:
            self._observe(error.response.headers)
            raise _translate(error) from error
        self._observe(raw.headers)
        response = await raw.parse()
        usage = response.usage
        tokens_in = (
            usage.input_tokens
            + (usage.cache_creation_input_tokens or 0)
            + (usage.cache_read_input_tokens or 0)
        )
        return ClassifierResponse(
            results=_answers(response, judgments),
            info=ClassifierInfo(id="claude", model=response.model, version=version("anthropic")),
            input_tokens=tokens_in,
            output_tokens=usage.output_tokens,
        )


def _retry_after(error: anthropic.APIStatusError) -> float | None:
    try:
        return max(0.0, float(error.response.headers["retry-after"]))
    except (KeyError, ValueError):
        return None


def _error_type(error: anthropic.APIStatusError) -> str | None:
    body = error.body
    detail = body.get("error") if isinstance(body, dict) else None
    kind = detail.get("type") if isinstance(detail, dict) else None
    return kind if isinstance(kind, str) else None


def _translate(error: anthropic.APIStatusError) -> ClassifierError:
    """Map every API failure to a seam error (docs/spec/v0.md §7.5).

    Throttling and 5xx are retried by the limiter. A request too large for the model is
    `ClassifierTokensExceeded`. Everything else (401, 403, 404, 422, other 400s) is a request or
    setup the Classifier will never accept, so it is `ClassifierRejected` and cancels the Job.
    """
    status = error.status_code
    if status in (429, 529):
        return ClassifierThrottled(_retry_after(error))
    if status >= 500:
        return ClassifierUnavailable(f"classifier returned {status}")
    if status == 413 or _error_type(error) == "request_too_large":
        return ClassifierTokensExceeded("state over the request size limit")
    # The API has no distinct type for an over-long prompt: it is a plain 400
    # `invalid_request_error` whose message says so, so only that one 400 is read by message.
    if (
        status == 400
        and _error_type(error) == "invalid_request_error"
        and _TOO_LONG in error.message.lower()
    ):
        return ClassifierTokensExceeded("state over the token limit")
    return ClassifierRejected(f"classifier rejected the request ({status})")
