# SPDX-License-Identifier: Apache-2.0
"""`jobs:validate`'s cost and Classifier-bound duration estimates (docs/spec/v0.md §7.5, §9).

Built from the Planner's token estimate (characters / the Classifier's chars-per-token, plus the
Judgments' text and the per-request overhead), the price table and the rate-limit ceiling. It is
an upper bound: it assumes every request reads the whole Document up to the token budget, ignores
Section narrowing and the Judgment cache, and counts a Page with no text layer as an OCR'd Page
of typical length. A sample of the inputs is read, then scaled to the whole Job.
"""

from collections.abc import Mapping
from pathlib import Path

import pypdfium2 as pdfium

from acceleread.classifier import Capabilities, ask_chars
from acceleread.extract import extract_html
from acceleread.models import JobSpec, ResolvedManifest
from acceleread.pipeline import document_format, judgments_of
from acceleread.planner import JudgmentSpec
from acceleread.pricing import estimate_cost_usd
from acceleread.ratelimit import RateLimit

SAMPLE_DOCUMENTS = 200
OCR_PAGE_CHARS = 3000  # a scanned Page has no text layer to count, so assume a typical one


def document_chars(path: Path) -> int | None:
    """The Document's text length, from its text layer (no OCR). None if unreadable."""
    try:
        if document_format(str(path)) == "html":
            return len(extract_html(path).text)
        pdf = pdfium.PdfDocument(path)
        try:
            total = 0
            for page in pdf:
                textpage = page.get_textpage()
                total += len(textpage.get_text_range().strip()) or OCR_PAGE_CHARS
                textpage.close()
                page.close()
            return total
        finally:
            pdf.close()
    except Exception:  # an unreadable file fails its own Document, not the estimate
        return None


def _request_tokens(
    chars: int, specs: list[JudgmentSpec], capabilities: Capabilities
) -> tuple[int, float]:
    """(requests, tokens) for one Document: one request per distinct `reads` set."""
    groups: dict[tuple[str, ...] | None, list[JudgmentSpec]] = {}
    for spec in specs:
        reads = None if spec.reads is None else tuple(sorted(set(spec.reads)))
        groups.setdefault(reads, []).append(spec)
    tokens = 0.0
    for members in groups.values():
        judged = sum(ask_chars(m.ask) for m in members)
        room = max(0.0, capabilities.token_budget - capabilities.request_overhead_tokens)
        state = min(chars, max(0.0, room * capabilities.chars_per_token - judged))
        tokens += (
            state + judged
        ) / capabilities.chars_per_token + capabilities.request_overhead_tokens
    return len(groups), tokens


def estimate_job(
    spec: JobSpec,
    manifest: ResolvedManifest,
    capabilities: Capabilities,
    rate_limit: RateLimit | None,
    prices: Mapping[str, float] | None = None,
) -> tuple[float | None, float | None]:
    """(estimated USD, estimated Classifier-bound seconds), or None where it can't be known."""
    sources = [d.source for d in spec.inputs if "://" not in d.source and Path(d.source).is_file()]
    sample = [
        c
        for source in sources[:SAMPLE_DOCUMENTS]
        if (c := document_chars(Path(source))) is not None
    ]
    if not sample:
        return None, None
    specs = judgments_of(manifest)
    per_doc = [_request_tokens(chars, specs, capabilities) for chars in sample]
    scale = len(spec.inputs) / len(sample)
    requests = sum(r for r, _ in per_doc) * scale
    tokens = sum(t for _, t in per_doc) * scale
    cost = estimate_cost_usd(manifest.model, round(tokens), 0, prices)
    seconds = (
        max(tokens / rate_limit.tokens_per_s, requests / rate_limit.requests_per_s)
        if rate_limit
        else None
    )
    return cost, seconds
