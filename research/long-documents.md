# Classifying Documents longer than Jev's context budget

Research for [#3](https://github.com/Jarrod-Bob/acceleread/issues/3). Researched 2026-10-03 against `jev-1.13.0` and `typesafe-sdk` 0.7.2.

## Answer (TL;DR)

For v0, send the **whole Document text in one request when it fits**. When it doesn't fit, send **one request with a trimmed, structured state**: title and metadata, the head of the text, and the tail of the text (head+tail truncation), sized to a conservative token estimate. Record on the Document Record that the text was truncated and how much of it was sent. Let low confidence trigger Escalation. Don't build chunk-and-aggregate in v0. Keep it as a later Classifier variant behind the seam, and only adopt it if the evaluation harness shows it beats head+tail on our corpus.

There is **no public tokenizer and no token-counting endpoint** for Jev. You can only see an exact count after the call, in `usage.input_tokens`. Pre-flight sizing therefore has to be a conservative character-based estimate that the observed `usage.input_tokens` values keep calibrated.

## 1. The constraint, precisely

From [Models](https://docs.typesafe.ai/models.md):

- "64k tokens per request; 32k tokens for `state` plus the longest question."
- "Jev ingests the `state` once and evaluates every question against it in parallel. The 64k budget covers the `state` plus all questions combined; the 32k budget applies to the `state` plus the single longest question."
- Input is text only: a string, JSON object or array of text values.
- Price is $0.042 per 1M input tokens, and output is free. The rate limit is 100K tokens/s and 80 req/s.

What this means for acceleread:

- **The Taxonomy uses up the state budget.** A classification is one `Choice` whose `criteria` holds every Category name and description. [Choice](https://docs.typesafe.ai/primitives/choice.md) says "The option names and their descriptions are both sent to the model" and "A Choice question accepts up to 255 options, and adding options costs a few tokens each". The usable state budget is therefore `32k − tokens(classification question)`, so it shrinks as the Taxonomy gets bigger or more descriptive.
- With only one question per request, the 64k total budget never binds. The 32k limit is the one that matters.

## 2. Counting tokens before sending

What the sources show:

- The [API reference](https://docs.typesafe.ai/api.md) documents a single evaluation endpoint, `POST /v1/systemone`, plus `GET /v1/models`. It has no count-tokens or tokenize endpoint.
- The `typesafe-sdk` 0.7.2 wheel from PyPI has no tokenizer, no counting helper and no tokenizer dependency. It depends only on `httpx2`, `pydantic`, `tenacity` and `typing-extensions`. Grepping its source for `token` finds only the `Usage` model.
- The response does report the exact count: `usage.input_tokens` ("Number of billable input tokens used to evaluate the request") and `usage.output_tokens` ([Answers and responses](https://docs.typesafe.ai/sdk/python/api/types/responses.md), and `typesafe_sdk/_schemas/models.py` in the wheel).
- The docs don't say what happens when a request goes over the limit. The documented errors are 401, 422 ("request body failed validation"), 429 and 529 ([API reference, Errors](https://docs.typesafe.ai/api.md)). I couldn't test this because there was no API key in this environment. The most likely outcome is a 422 error, though the server might truncate silently instead. **This is unverified.**

One data point lets us estimate the characters-per-token ratio. The [Parallel questions cookbook](https://docs.typesafe.ai/cookbooks/parallel_questions.md) sends a 53,777-character Wikipedia article plus 13 questions in one call, and that call costs $0.000497 at $0.042/Mtok. That works out to about **11.8k input tokens**, or roughly 4.5 characters per token or more for English prose once you allow for the questions. That cookbook used `jev-1.12`. That `jev-1.13` uses the same tokenizer is an assumption.

Recommended pre-flight approach:

- Estimate tokens as `ceil(len(text) / 3.5)`. This deliberately overestimates compared with the observed ~4.5, which covers OCR noise, tables, numbers and non-English text, all of which tokenize less efficiently.
- Estimate the classification question's size the same way from its serialized JSON, and subtract it from a safety budget of about 30k, leaving a margin below 32k.
- Log `usage.input_tokens` against the character count for every request. This lets the evaluation and observability work tighten the ratio for each corpus.
- If a request is still rejected for being too long, retry once with the state budget halved.

## 3. Options surveyed

| Strategy | Requests per Document | Cost | Accuracy notes | Fit for v0 |
| - | - | - | - | - |
| **Whole text** (when it fits) | 1 | ≤ ~$0.0013 at 30k tok | Best coverage, but Jev "suffers from context rot" (see below) | Yes, as the default |
| **First-N (head only)** | 1 | capped | Works when the topic is stated up front (most articles and papers). Misses documents whose substance comes late | Partly |
| **Head + tail** | 1 | capped | Best truncation method in Sun et al. 2019. Keeps the conclusions and summary as well as the introduction | **Yes, for overflow** |
| **Title/abstract/headings extraction** | 1 | small | This is TypeSafe's own pattern (the SEC cookbook trims each 10-K to Item 1 "Business"). Needs reliable structure detection in PDFs, which is fragile in general | Use cheap parts only: title/metadata as fields. Full section extraction is out of scope |
| **Chunk-and-aggregate probabilities** | ⌈len/chunk⌉ | roughly the whole document, with the Taxonomy re-sent per chunk | Mixed evidence in the literature. Needs an aggregation rule (mean, log-sum, confidence-weighted) whose calibration nobody has checked | Defer (v1, needs evaluation) |
| **Relevance-filter chunks, then classify** | chunks + 1 | ~2× the document | TypeSafe's [RAG passages cookbook](https://docs.typesafe.ai/cookbooks/classifying_rag_passages.md) uses Nouls to filter passages, but there's no query to filter against when the question is "what is this about?" | No |
| **Hierarchical classification cookbook** | per tree level | — | Solves *taxonomy depth*, not document length ([cookbook](https://docs.typesafe.ai/cookbooks/hierarchical_classification.md) uses short abstracts as state). Hierarchical taxonomies are out of scope for v0 | Not applicable |
| **Parallel questions cookbook** | 1 | — | Many questions sharing one state cost the same as one question. It doesn't help with length, but it means any extra per-Document questions (language, document type) are almost free | Not applicable to length |

### TypeSafe's own guidance points toward sending less, not more

- From [Jev 1.13 jaggedness, "Large state full of irrelevant detail"](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md): "Accuracy falls as the state grows with content unrelated to the decision. Unrelated detail acts as a distractor." Its advice is "retrieve and filter in code first, and send only the fields the question needs." The page also says "Jev suffers from context rot, so unrelated material in the `state` costs you accuracy."
- The [Classification using confidence cookbook](https://docs.typesafe.ai/cookbooks/classification_using_confidence.md) classifies SEC 10-Ks into 75 groups. It doesn't send the whole filing: each one is "trimmed to Item 1 'Business'… the only part an industry code is about", which comes to 700–2,200 words. It then gates on `confidence ≥ 0.9` and falls back to a broader label below that. This is the closest first-party analogue to acceleread.
- The [State](https://docs.typesafe.ai/concepts/state.md) page recommends using a JSON object with named fields over a plain string. That supports a structured state such as `{title, metadata, head, tail, truncated}`.

### Long-document classification literature

These results are for fine-tuned 512-token encoders, which face much harsher truncation than Jev's ~30k. The direction of the evidence still transfers.

- Sun et al. 2019, *How to Fine-Tune BERT for Text Classification?* ([arXiv:1905.05583](https://ar5iv.labs.arxiv.org/html/1905.05583)). They compared head-only (510 tokens), tail-only and head+tail (first 128 plus last 382 tokens) against hierarchical chunking with mean, max or self-attention pooling. Head+tail did best (IMDb 5.42% error, Sogou 2.43%) and **beat the chunk-and-pool methods**.
- Park et al. 2022, *Efficient Classification of Long Documents Using Transformers* (ACL, [arXiv:2203.11258](https://ar5iv.labs.arxiv.org/html/2203.11258)). Their finding: "More complex models often fail to outperform simple baselines and yield inconsistent performance across datasets." First-512 truncation stayed competitive and ran about 12× faster than Longformer. Methods that select sentences won when the key information sat at the *end* (Inverted EURLEX). This is the case that head+tail is meant to cover.
- Dai, Chalkidis et al. 2022, *Revisiting Transformer-based Models for Long Document Classification* ([arXiv:2204.06683](https://ar5iv.labs.arxiv.org/html/2204.06683)). On long clinical and legal documents, going from 512 to 4,096 tokens helped a lot: MIMIC-III micro-F1 rose from ~56 to ~68–70, and ECtHR from 73.5 to ~81. For hierarchical models they advise small overlapping segments, and they say long-document methods matter once documents average around 2K tokens or more. **Conclusion:** coverage matters up to a few thousand tokens, and Jev's ~30k window is already far past that. The gain from going *beyond* 30k with chunking is likely small for topic classification.

## 4. How large is the overflow problem?

At ~4.5 characters per token and ~2,500–3,500 characters per dense PDF page, 30k tokens is roughly **40–50 pages of prose**. Articles, papers and most reports fit whole. Theses, books, long regulatory filings and manuals don't. Using the [Models](https://docs.typesafe.ai/models.md) price and rate limit, a cap also bounds batch cost and time:

| Per-Document state | 100k-Document Job: tokens | Cost | Time at 100K tok/s |
| - | - | - | - |
| 30k (full budget) | 3.0B | $126 | ~8.3 h |
| 8k (head+tail cap) | 0.8B | $34 | ~2.2 h |

Chunk-and-aggregate would push the overflow Documents *above* the 30k-per-Document row, because it pays for the whole Document plus a copy of the Taxonomy per chunk.

## 5. Recommended v0 strategy

1. **Sizing.** `budget_state = 30_000 − est_tokens(classification_question)`, with `est_tokens = ceil(chars / 3.5)`.
2. **Fits:** state = `{"title": …, "metadata": {…}, "text": full_text}`.
3. **Overflows:** state = `{"title": …, "metadata": {…}, "head": first ~25% of budget, "tail": last ~75% of budget, "note": "middle omitted"}`. The 25/75 split follows Sun et al.'s 128/382 proportion. Treat the ratio as a tunable to evaluate, not a settled value. Cut on page or paragraph boundaries. Optionally drop pages that are obviously boilerplate (references, bibliography) before truncating, if Extraction can detect them cheaply.
4. **Record it.** The Document Record gets `truncated: bool`, `tokens_sent_est`, `input_tokens` (from `usage`) and the pages included. Downstream users can then filter or re-run truncated Documents.
5. **Gate on confidence.** Truncated Documents go through the same confidence threshold as all others, and low confidence leads to Escalation. Optionally use a stricter threshold for truncated Documents.
6. **Design for later.** Make the truncation policy a parameter of the Jev Classifier (`full | head_tail | chunk_aggregate`), so the evaluation harness can compare chunk-and-aggregate without an API change.

Possible aggregation rules for a future chunk-and-aggregate variant: mean of the per-chunk distributions, confidence-weighted mean, or a log-probability sum (product of experts). The log-sum treats chunks as independent, which makes it overconfident. None of these keeps Jev's per-answer calibration, so its thresholds would need re-tuning.

## 6. Open questions this surfaced

- **Over-limit behaviour.** What status and body does the API return for a state over 32k tokens? Does it ever truncate silently? This needs a live probe with an API key.
- **Measured characters per token** for Jev on our corpus, including OCR'd pages, from logged `usage.input_tokens`. This belongs to the evaluation harness and observability.
- **Context rot below the limit.** Does a Document that fits whole classify *better* when capped (for example at 8k head+tail) than when sent in full? If it does, the cap becomes the default for every Document and the cost per Job drops about 4×.
- **Document Record schema** needs truncation and coverage fields.
- **Taxonomy limits.** The 255-option Choice cap, and the token cost of Category descriptions reducing the state budget. Taxonomy validation should reject or warn on these.
- **Boilerplate stripping.** Should Extraction tag references, appendices and front matter so the classifier can skip them?
