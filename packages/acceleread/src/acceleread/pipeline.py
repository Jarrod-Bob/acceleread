# SPDX-License-Identifier: Apache-2.0
"""Run a Job in memory: extract each Document and classify it against the Taxonomy.

Tracer scope (docs/spec/v0.md §5.3, §7): one request per Document and a naive head+tail cut when
the text is over budget. Proper planning, storage and the runner arrive with later build issues.
"""

import hashlib
import json
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from acceleread.classifier import Capabilities, Choice, Classifier, JSONValue
from acceleread.extract import Extracted, extract_pdf
from acceleread.jev import JevClassifier
from acceleread.models import (
    Coverage,
    DocumentRecord,
    JobSpec,
    Judgment,
    Page,
    RecordError,
    Source,
    Taxonomy,
    TaxonomyRef,
    Timings,
    Usage,
)

TAXONOMY_JUDGMENT = "taxonomy"
HEAD_SHARE = 0.25
ELISION = "\n[…]\n"


@dataclass(frozen=True)
class Plan:
    state: dict[str, JSONValue]
    coverage: Coverage


def taxonomy_choice(taxonomy: Taxonomy) -> Choice:
    return Choice(
        instructions="Which category best describes `document`?",
        options={c.name: c.description for c in taxonomy.categories},
    )


def estimate_tokens(value: object, capabilities: Capabilities) -> int:
    return round(len(json.dumps(value, ensure_ascii=False)) / capabilities.chars_per_token)


def _pages_in(pages: list[Page], spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Collapse the Pages overlapping the read character spans into page-number ranges."""
    hit = sorted({p.number for p in pages for s, e in spans if p.start < e and s < p.end})
    ranges: list[tuple[int, int]] = []
    for number in hit:
        if ranges and ranges[-1][1] == number - 1:
            ranges[-1] = (ranges[-1][0], number)
        else:
            ranges.append((number, number))
    return ranges


def plan(doc: Extracted, title: str, choice: Choice, capabilities: Capabilities) -> Plan:
    """Whole-Document state when it fits; otherwise the first 25% and last 75% of the budget."""
    question_tokens = estimate_tokens(
        {"i": choice.instructions, "c": dict(choice.options)}, capabilities
    )
    shell_tokens = estimate_tokens({"document": {"title": title, "text": ""}}, capabilities)
    available = max(0, capabilities.token_budget - question_tokens - shell_tokens)
    max_chars = int(available * capabilities.chars_per_token)
    text = doc.text
    if len(text) <= max_chars:
        spans = [(0, len(text))]
        truncated = False
    else:
        budget = max(0, max_chars - len(ELISION))
        head = int(budget * HEAD_SHARE)
        tail_start = len(text) - (budget - head)
        spans = [(0, head), (tail_start, len(text))]
        text = text[:head] + ELISION + text[tail_start:]
        truncated = True
    state: dict[str, JSONValue] = {"document": {"title": title, "text": text}}
    coverage = Coverage(
        page_ranges=_pages_in(doc.pages, spans),
        truncated=truncated,
        est_tokens=estimate_tokens(state, capabilities) + question_tokens,
    )
    return Plan(state=state, coverage=coverage)


def _source(path: Path) -> Source:
    data = path.read_bytes()
    return Source(
        filename=path.name,
        format="pdf",
        sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),
    )


async def ingest(
    path: Path, taxonomy: Taxonomy, classifier: Classifier, job_id: str
) -> DocumentRecord:
    record = DocumentRecord(
        record_id=uuid.uuid4().hex,
        job_id=job_id,
        status="ok",
        source=_source(path),
        taxonomy=TaxonomyRef(name=taxonomy.name, hash=taxonomy.hash),
    )
    timings = Timings()
    started = time.perf_counter()
    try:
        doc = extract_pdf(path)
    except Exception as exc:  # any extraction failure still yields a Record
        record.status = "failed"
        record.errors.append(
            RecordError(stage="extract", code=type(exc).__name__, message=str(exc))
        )
        return record
    timings.extract_ms = round((time.perf_counter() - started) * 1000)
    record.text, record.pages = doc.text, doc.pages

    choice = taxonomy_choice(taxonomy)
    planned = plan(doc, doc.title or path.stem, choice, classifier.capabilities)
    started = time.perf_counter()
    try:
        response = await classifier.judge(planned.state, {TAXONOMY_JUDGMENT: choice})
    except Exception as exc:  # the Record keeps its text; classification is reported as failed
        record.status = "partial"
        record.errors.append(
            RecordError(stage="classify", code=type(exc).__name__, message=str(exc))
        )
    else:
        result = response.results[TAXONOMY_JUDGMENT]
        coverage = planned.coverage.model_copy(update={"input_tokens": response.input_tokens})
        record.classification = Judgment(
            kind=result.kind,
            value=result.value,
            probabilities=result.probabilities,
            confidence=result.confidence,
            classifier=response.info,
            coverage=coverage,
        )
        record.usage = Usage(
            requests=response.requests,
            input_tokens=response.input_tokens or 0,
            output_tokens=response.output_tokens or 0,
        )
    timings.classify_ms = round((time.perf_counter() - started) * 1000)
    record.timings = timings
    return record


async def run(spec: JobSpec, classifier: Classifier | None = None) -> AsyncIterator[DocumentRecord]:
    """Yield one Document Record per input, in input order."""
    if spec.taxonomy is None:
        raise ValueError("the tracer pipeline needs a Taxonomy; Question-only Jobs come later")
    classifier = classifier or JevClassifier(model=spec.model)
    taxonomy = spec.taxonomy.with_other()
    job_id = "job_" + uuid.uuid4().hex[:12]
    for path in spec.inputs:
        yield await ingest(path, taxonomy, classifier, job_id)
