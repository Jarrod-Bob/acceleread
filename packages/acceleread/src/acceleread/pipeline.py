# SPDX-License-Identifier: Apache-2.0
"""Run a Job in memory: extract each Document and classify it against the Taxonomy.

Tracer scope (docs/spec/v0.md §7): Taxonomy only. Planning lives in `planner.py`; storage and
the runner (which wires the Judgment cache) arrive with later build issues.
"""

import asyncio
import hashlib
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

from acceleread.classifier import Classifier
from acceleread.extract import Extracted, extract_pdf
from acceleread.jev import JevClassifier
from acceleread.models import (
    DocumentRecord,
    JobSpec,
    RecordError,
    Source,
    Taxonomy,
    TaxonomyRef,
    Timings,
)
from acceleread.planner import DocumentView, judge_document, judgment_specs
from acceleread.workers import ExtractTask, WorkerPool, WorkerSettings


def _source(path: Path) -> Source:
    data = path.read_bytes()
    return Source(
        filename=path.name,
        format="pdf",
        sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),
    )


async def _extract(path: Path, pool: WorkerPool | None) -> Extracted:
    """Extract in a pool worker when there is one, else in this process."""
    if pool is None:
        return extract_pdf(path)
    outcome = await asyncio.wrap_future(pool.submit(*ExtractTask(path, "pdf").for_pool()))
    if not outcome.ok:
        raise ExtractionFailed(outcome.code or "extract_error", outcome.message or "")
    result: Extracted = outcome.result
    return result


class ExtractionFailed(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


async def ingest(
    path: Path,
    taxonomy: Taxonomy,
    classifier: Classifier,
    job_id: str,
    pool: WorkerPool | None = None,
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
        doc = await _extract(path, pool)
    except Exception as exc:  # any extraction failure still yields a Record
        record.status = "failed"
        code = exc.code if isinstance(exc, ExtractionFailed) else type(exc).__name__
        record.errors.append(RecordError(stage="extract", code=code, message=str(exc)))
        return record
    timings.extract_ms = round((time.perf_counter() - started) * 1000)
    record.text, record.pages = doc.text, doc.pages

    view = DocumentView(text=doc.text, pages=doc.pages, title=doc.title or path.stem)
    started = time.perf_counter()
    outcome = await judge_document(view, judgment_specs(taxonomy, []), classifier)
    record.classification = outcome.classification
    record.usage = outcome.usage
    if outcome.errors:  # the Record keeps its text; classification is reported as failed
        record.status = "partial"
        record.errors.extend(outcome.errors)
    timings.classify_ms = round((time.perf_counter() - started) * 1000)
    record.timings = timings
    return record


async def run(
    spec: JobSpec, classifier: Classifier | None = None, workers: WorkerSettings | None = None
) -> AsyncIterator[DocumentRecord]:
    """Yield one Document Record per input, in input order.

    With `workers`, Documents are extracted in a worker pool sized by those settings.
    """
    if spec.taxonomy is None:
        raise ValueError("the tracer pipeline needs a Taxonomy; Question-only Jobs come later")
    classifier = classifier or JevClassifier(model=spec.model)
    taxonomy = spec.taxonomy.with_other()
    job_id = "job_" + uuid.uuid4().hex[:12]
    pool = workers.pool() if workers is not None else None
    try:
        for document in spec.inputs:
            yield await ingest(Path(document.source), taxonomy, classifier, job_id, pool)
    finally:
        if pool is not None:
            pool.close()
