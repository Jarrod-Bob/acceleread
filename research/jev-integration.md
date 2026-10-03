# Jev integration: request shape and cost per document

Research for [#4](https://github.com/Jarrod-Bob/acceleread/issues/4), part of the v0 spec map ([#1](https://github.com/Jarrod-Bob/acceleread/issues/1)).
Sources read 2026-10-03. TypeSafe facts come from the live docs at `docs.typesafe.ai`. Each page is cited inline, and its `.md` form is the one that was read. Version-dependent facts are tied to **`jev-1.13.0`** and **`typesafe-sdk` v0.7.2**.

## TL;DR

- **One request per Document.** `state` is the extracted text, and the question is a single `Choice` whose `criteria` *is* the Taxonomy (one option per Category, up to 255). The answer gives the chosen Category, the full `probabilities` map and a `confidence` value. That covers everything a Classification needs, in one call.
- **Cost.** Jev charges $0.042 per 1M input tokens, and output is free. That is about **$0.16 per 1k Documents at 3k tokens** and **about $0.87 per 1k at 20k tokens**. Claude Haiku 4.5 costs about 24x more (about $3.95 and $20.95), and Gemini 3.1 Flash-Lite about 6x more (about $1.00 and $5.25).
- **Throughput, not cost, is the binding constraint at our scale.** The 100K tokens/s limit allows about 26 docs/s at 3k tokens and about 4.8 docs/s at 20k. A 100k-Document Job of 20k-token docs therefore needs at least about 5.8 h of wall-clock time, and the limits are "adjusting dynamically".
- **Use the async SDK with a concurrency cap of our own.** The SDK retries 408, 429 and 5xx responses (529 included) by default and honors `retry-after`. It has no built-in concurrency limiter or token budgeter.
- **Escalate on `confidence`, with a threshold tuned on our data.** For a Choice, `confidence = (p_max − 1/n)/(1 − 1/n)`, so its meaning depends on the Taxonomy size *n*. Add an `other` Category.

## 1. Request shape for one Document against a flat Taxonomy

**Endpoint:** `POST https://api.typesafe.ai/v1/systemone` with the header `Authorization: Bearer <key>`. The body has three parts: `state` (a string, object or array), `model`, and `questions` (a map from an id you choose to a Question). The id is "not sent to the underlying model and is not used in inference". ([api](https://docs.typesafe.ai/api.md))

**Choice** ([api](https://docs.typesafe.ai/api.md), [primitives/choice](https://docs.typesafe.ai/primitives/choice.md)):
- `criteria` maps each option to a description, or to `null`. The maximum is **255 options per Choice**. Both option names and descriptions are sent to the model, "so write descriptions that separate the options from each other".
- "Add an `other` or `none of the above` option when the list might not cover every input."
- A description can be a structured object, for example `{what, not_for, examples}`, when two Categories keep getting confused.
- The SEC classification cookbook says "a Choice works reliably up to roughly 240 options" and uses 75 options in one Choice ([cookbooks/classification_using_confidence](https://docs.typesafe.ai/cookbooks/classification_using_confidence.md)).

**Answer** ([api](https://docs.typesafe.ai/api.md)):
- `choice` is the highest-probability option.
- `probabilities` maps every option to a probability, and the values sum to 1.
- `confidence` is a value from 0 to 1.
- The top-level `model` field gives the versioned id that answered, for example `jev-1.13.0`.
- `usage.input_tokens` and `usage.output_tokens` give the token counts.

The **Classification** in `CONTEXT.md` (chosen Category, full distribution and confidence) maps 1:1 onto a Choice answer.

**Noul and Score** are not needed for single-label flat classification. Noul is the right primitive if multi-label is ever wanted, with one Noul per Category ([primitives/noul](https://docs.typesafe.ai/primitives/noul.md)). Multi-label is out of scope for v0.

### HTTP body

```json
{
  "model": "jev-1.13.0",
  "state": {
    "document": { "title": "Attention Is All You Need", "text": "<extracted text>" }
  },
  "questions": {
    "category": {
      "type": "choice",
      "instructions": "Which category best describes `document` as a whole? Judge by its main subject and purpose.",
      "criteria": {
        "research_paper": "Academic research: a paper or preprint reporting methods and results",
        "news": "Reporting on recent events",
        "tutorial": "Step-by-step instructional material",
        "opinion": "An essay or commentary arguing a position",
        "other": "Fits none of the categories above"
      }
    }
  }
}
```

### Worked example (Python SDK, `typesafe-sdk` v0.7.2)

Install with `pip install 'typesafe-sdk[http2]'` and set `TYPESAFE_API_KEY` ([sdk/python](https://docs.typesafe.ai/sdk/python.md)). Signatures follow the async client reference ([sdk/python/api/clients/async](https://docs.typesafe.ai/sdk/python/api/clients/async.md)), the response types ([sdk/python/api/types/responses](https://docs.typesafe.ai/sdk/python/api/types/responses.md)) and [RetryPolicy](https://docs.typesafe.ai/sdk/python/api/retries.md).

```python
import asyncio
from collections.abc import Mapping

import httpx2
from typesafe_sdk import AsyncTypeSafeClient, Choice, RetryPolicy, TypeSafeAPIError

MODEL = "jev-1.13.0"  # pin a version: thresholds are tuned per version (docs: models)

TAXONOMY: dict[str, str] = {  # one Job's flat Taxonomy: Category -> description
    "research_paper": "Academic research: a paper or preprint reporting methods and results",
    "news": "Reporting on recent events",
    "tutorial": "Step-by-step instructional material",
    "opinion": "An essay or commentary arguing a position",
    "other": "Fits none of the categories above",
}


def questions(taxonomy: Mapping[str, str]) -> dict:
    return {
        "category": Choice(
            instructions=(
                "Which category best describes `document` as a whole? "
                "Judge by its main subject and purpose."
            ),
            criteria=dict(taxonomy),
        )
    }


async def classify(client: AsyncTypeSafeClient, title: str, text: str) -> dict:
    resp = await client.system_one(
        state={"document": {"title": title, "text": text}},
        questions=questions(TAXONOMY),
        model=MODEL,
    )
    a = resp.choices["category"]
    return {
        "category": a.choice,
        "probabilities": dict(a.probabilities),
        "confidence": a.confidence,
        "model": resp.model,                    # versioned id that answered
        "input_tokens": resp.usage.input_tokens,  # for per-Job cost reporting
        "request_id": resp.request_id,
    }


async def classify_job(docs: list[tuple[str, str]], concurrency: int = 32) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)  # SDK has no built-in concurrency cap
    async with AsyncTypeSafeClient(
        http_client=httpx2.AsyncClient(http2=True),  # multiplex many requests (docs: usage#http2)
        timeout=60.0,  # default is 10s per HTTP operation; long states need more
        retry=RetryPolicy(max_retries=5, timeout=300.0),  # defaults: 2 retries, 30s budget
    ) as client:

        async def one(title: str, text: str) -> dict:
            async with sem:
                try:
                    return await classify(client, title, text)
                except TypeSafeAPIError as e:  # retries exhausted, or a 4xx such as 422
                    return {"error": e.status, "request_id": e.request_id}

        return await asyncio.gather(*(one(t, x) for t, x in docs))
```

Response for one document, in the shape shown in the docs ([api](https://docs.typesafe.ai/api.md)). The values here are illustrative:

```json
{"model": "jev-1.13.0",
 "answers": {"category": {"type": "choice", "choice": "research_paper",
   "probabilities": {"research_paper": 0.97, "tutorial": 0.02, "news": 0.01, "opinion": 0.0, "other": 0.0},
   "confidence": 0.96}},
 "usage": {"input_tokens": 3412, "output_tokens": 30}}
```

Notes on state ([concepts/state](https://docs.typesafe.ai/concepts/state.md), [model-jaggedness/jev-1.13](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md)):
- Name the fields in `state` and refer to them with backticked paths, as `document` is referenced in the example ([primitives](https://docs.typesafe.ai/primitives.md)).
- Jev input is **text only**, so OCR must happen before the request ([models](https://docs.typesafe.ai/models.md)).

## 2. Batching several questions per call

- All questions in one request share the `state`, which is ingested once. They are evaluated independently and in parallel. "Adding questions barely changes the response time and costs only the tokens for the extra questions" ([primitives](https://docs.typesafe.ai/primitives.md)).
- Questions cannot see each other's answers. If one judgment depends on another, make a second request ([primitives](https://docs.typesafe.ai/primitives.md)).
- Measured in the parallel-questions cookbook: one call with 13 questions over the ~54k-character GDPR article cost $0.000497 and took 0.27 s. Thirteen single-question calls cost $0.006090 and took 2.71 s, so batching was **12.2x cheaper and 10.0x faster** with the same answers. The run used jev-1.12 at $0.042/M input ([cookbooks/parallel_questions](https://docs.typesafe.ai/cookbooks/parallel_questions.md)). As a side calculation, $0.000497 at $0.042/M is about 11.8k input tokens for ~54k characters plus 13 questions, or roughly 4.5 characters per token.
- **For acceleread:** one Classification needs only one question. If more per-Document judgments are added later, they ride in the *same* request at near-zero marginal cost: Speculative fan-out ([patterns/fan-out](https://docs.typesafe.ai/patterns/fan-out.md)). One example would be a Noul for "is this text garbled or unreadable?" to flag bad Extraction.
- **Never send one request per Category.** That re-sends the document *n* times.

## 3. Async, concurrency, retries and 429s

- **Clients.** `AsyncTypeSafeClient` and `TypeSafeClient` both offer `system_one(state, questions, *, model, retry, timeout, extra_headers, extra_body, response_model)`. Passing a pydantic `response_model` gives typed access to answers ([sdk/python/usage](https://docs.typesafe.ai/sdk/python/usage.md)).
- **Concurrency.** The docs show no batch endpoint and no built-in concurrency limiter. Concurrency is the caller's job, for example `asyncio.Semaphore` with `asyncio.gather`. The docs recommend the `http2` extra "when sending many concurrent requests" ([sdk/python/usage#http2](https://docs.typesafe.ai/sdk/python/usage.md)).
- **Default retries** ([sdk/python/api/retries](https://docs.typesafe.ai/sdk/python/api/retries.md)):
  - `RetryPolicy(max_retries=2, backoff_initial=0.5, backoff_max=5.0, backoff_jitter=0.25)`
  - `http_statuses={408, 429, *range(500, 600)}`, which includes **529 Overloaded**
  - `respect_retry_after=True`, which honors `Retry-After` and `retry-after-ms`
  - connection and timeout errors are retried
  - `timeout=30.0`, the total retry budget per call
- **HTTP timeout.** The default is **10 s per HTTP operation** ([sdk/python/api/constants](https://docs.typesafe.ai/sdk/python/api/constants.md)). The SEC cookbook uses `timeout=120.0` for long filings.
- **Errors** ([sdk/python/api/exceptions](https://docs.typesafe.ai/sdk/python/api/exceptions.md)): `TypeSafeRateLimitError` (429, with `.retry_after_ms`), `TypeSafeUnprocessableEntityError` (422, a malformed question), `TypeSafeInternalServerError` (5xx), `TypeSafeAPIConnectionError` and `TypeSafeAPITimeoutError`. All HTTP errors carry `.status`, `.body` and `.request_id`.
- **Rate limits.** The limits are 100K tokens/s and 80 requests/s, and a request over either gets a 429. The models page warns: "Rate limits are adjusting dynamically … can change without notice". Higher limits are available on custom or enterprise plans ([models](https://docs.typesafe.ai/models.md)).
- **Implication.** A long-running Job should treat 429s as normal backpressure. Once the SDK's retries are exhausted, the Job queue should put the Document back rather than fail the Job. A client-side token-rate limiter (tokens/s) is the right shape for a throttle, because for documents of any size the 100K tok/s limit binds before 80 req/s does.

## 4. Gating Escalation on confidence

- **What confidence measures.** `confidence` summarizes how concentrated `probabilities` is: 1 when all the mass is on one option, 0 when the mass is spread evenly. For a Choice with *n* options, `confidence = (p_max − 1/n) / (1 − 1/n)`. Nouls have no confidence, and `|2p − 1|` is the equivalent ([confidence](https://docs.typesafe.ai/confidence.md)).
  - **Consequence for user-defined Taxonomies:** the same `p_max` maps to different confidence values for different *n*. A threshold tuned on a 5-Category Job does not carry over unchanged to a 60-Category Job.
- **Three bands.** The docs suggest high (act automatically), medium (flag or confirm) and low (route elsewhere). "Thresholds scale with risk". Their examples use a 0.5 floor, with 0.85 to 0.9 for high-stakes actions, and they say the values "depend on your domain … start conservative, test with your own data" ([confidence](https://docs.typesafe.ai/confidence.md), [patterns/confidence-routing](https://docs.typesafe.ai/patterns/confidence-routing.md)).
- **Evidence from the SEC cookbook** (75 options, 60 filings of 700 to 2,200 words, jev-1.12) ([cookbooks/classification_using_confidence](https://docs.typesafe.ai/cookbooks/classification_using_confidence.md)):
  - A 0.9 cutoff split the filings in half. The confident half was right 27/30 times and the rest 12/30.
  - With a hierarchy, the unsure answers fall back to the parent division (70% useful), with no second call.
  - acceleread's Taxonomy is flat, so there is no free fallback level. A low-confidence Classification has to Escalate (to an LLM or a person), or be recorded as `uncertain` with its top-k Categories.
- **Proposed v0 policy** (to be validated on the sample corpus):
  1. Store the full distribution and confidence on every Document Record.
  2. Mark a Document `needs_escalation` when `confidence < τ` or `choice == "other"`. τ is configurable per Job, starting around 0.5 to 0.9.
  3. Escalation is a separate Classifier pass over only those Documents.
- **Calibration hygiene:**
  - Pin `jev-1.13.0` rather than `jev-latest`, because the aliases move and thresholds are tuned per version ([models](https://docs.typesafe.ai/models.md)).
  - Record `response.model` in each Classification.
  - Jev 1.13 can lean toward the **first Choice option** ([jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md)). Consider shuffling the Category order per Document, or re-asking borderline cases with reversed order.

## 5. Cost per 1k Documents

**Jev price:** $42 per billion tokens, which is $0.042 per 1M **input** tokens. Output is free ([models](https://docs.typesafe.ai/models.md)).

**Assumptions:**
- About 800 tokens of per-request overhead for the instruction and a roughly 20-Category Taxonomy with one-line descriptions. The docs' toy requests show around 300 input tokens for one short question ([api](https://docs.typesafe.ai/api.md)), and each Choice option "costs a few tokens" ([primitives/choice](https://docs.typesafe.ai/primitives/choice.md)).
- For the LLMs: the same input, about 30 output tokens for a JSON label, no prompt caching, and standard (non-batch) pricing unless noted.
- Token counts differ between vendors' tokenizers, so treat these figures as order-of-magnitude comparisons.

| Classifier | Price (in / out per 1M) | 3k-token docs (3.8M in) | 20k-token docs (20.8M in) |
|---|---|---|---|
| **Jev 1.13** | $0.042 / free | **$0.16** | **$0.87** |
| Gemini 3.1 Flash-Lite | $0.25 / $1.50 ([Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing)) | $1.00 | $5.25 |
| Claude Haiku 4.5 | $1 / $5 ([Claude pricing](https://platform.claude.com/docs/en/about-claude/pricing)) | $3.95 | $20.95 |
| Claude Haiku 4.5, Batch API (−50%) | $0.50 / $2.50 | $1.98 | $10.48 |

- Jev is about 6x cheaper than Gemini 3.1 Flash-Lite and about 24x cheaper than Claude Haiku 4.5 at standard pricing. The gap widens with document length, because input dominates the bill.
- LLMs also do not return a calibrated distribution over Categories by default. Getting one means extra work, for example logprobs or repeated sampling, at extra cost.
- **Per 100k-document Job:** about $16 (3k-token docs) to about $87 (20k-token docs) with Jev. Cost is negligible compared with the hours of wall-clock time involved.

**Throughput at the published 100K tok/s** ([models](https://docs.typesafe.ai/models.md)):

| Doc size | Max docs/s | 10k-doc Job | 100k-doc Job |
|---|---|---|---|
| 3k tokens | ~26 | ~6.3 min | ~63 min |
| 20k tokens | ~4.8 | ~35 min | ~5.8 h |

## 6. Document length limits

- **Context limits.** The budget is 64k tokens per request in total, and **32k for `state` plus the longest question** ([models](https://docs.typesafe.ai/models.md)). A 20k-token document fits. A long PDF does not: at a few hundred to ~1k tokens per page, the 32k limit is crossed somewhere around 30 to 100+ pages.
- **Accuracy.** "Accuracy falls as the state grows with content unrelated to the decision … Jev suffers from context rot". The advice is to filter in code first ([jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md)). The SEC cookbook trimmed each 10-K to Item 1 "Business" for that reason.
- **Token counting.** The docs show no tokenizer and no token-count endpoint. Token counts are only known afterwards, from `usage`. acceleread will need a conservative characters-per-token estimate to decide when to truncate or select sections.
- **Language.** English is strongest. Other languages "are handled but not equally well" ([models](https://docs.typesafe.ai/models.md)).

## 7. Other integration facts

- **Gateways.** The SDK works through OpenRouter, Vercel AI Gateway and Pydantic AI Gateway via `base_url` ([sdk/python/usage](https://docs.typesafe.ai/sdk/python/usage.md)).
- **No self-hosting and no fine-tuning.** Domain fit comes from `state`, `instructions` and `criteria` ([models](https://docs.typesafe.ai/models.md)).
- **Data handling.** Jev is not trained on customer requests. Zero data retention (ZDR) is enterprise-only ([models](https://docs.typesafe.ai/models.md)), which matters for the map's privacy note.
- **Logging.** At `debug` level the SDK logs request and response bodies **unredacted** ([sdk/python/usage](https://docs.typesafe.ai/sdk/python/usage.md)). acceleread must not ship with `TYPESAFE_LOG_LEVEL=debug` by default.

## Open questions surfaced

1. **Long-Document policy.** What should happen for Documents over about 30k tokens? Options are truncating, selecting leading or salient pages, or chunking and aggregating distributions. This also covers context rot below the limit, and how to estimate tokens before sending.
2. **Job throughput and rate-limit budgeting.** The token-rate limit (dynamic, about 5.8 h per 100k long docs) sets Job duration. The queue needs a client-side tokens/s limiter, 429 backpressure and resumability. Is an enterprise limit needed?
3. **Threshold calibration per Taxonomy size.** Choice confidence depends on *n*, so the default τ, or its per-Job tuning, belongs in the evaluation harness.
4. **Option-order bias.** Is per-Document shuffling of Categories worth it, measured on the sample corpus?
5. **Extraction-quality signal.** Is a co-asked Noul ("is the text garbled?") worth adding to flag bad OCR, at near-zero cost?
