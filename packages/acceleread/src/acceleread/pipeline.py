# SPDX-License-Identifier: Apache-2.0
"""The per-Document pipeline stages (docs/spec/v0.md §7.3) and `acceleread.run`.

A Document goes extracting -> `extracted` -> classifying. This module holds the stages, each a
function from a Record to a Record; the runner (`runner.py`) schedules them, persists the Record
between them and enforces the Job-level policies. Nothing here logs Document text.
"""

import asyncio
import glob
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from acceleread.classifier import Classifier
from acceleread.extract import Extracted
from acceleread.languages import workspace_tessdata
from acceleread.models import (
    DocumentInput,
    DocumentRecord,
    ExtractionProfile,
    JobSpec,
    MetadataField,
    RecordError,
    ResolvedManifest,
    Section,
    Source,
    Taxonomy,
    TaxonomyRef,
)
from acceleread.planner import DocumentView, JudgmentSpec, judge_document, judgment_specs
from acceleread.sections import DetectionInput, Heading, detect_sections
from acceleread.sections.outline import outline_headings, pdf_outline
from acceleread.workers import ExtractTask, WorkerPool, extract_document
from acceleread.workspace import InputRef
from acceleread.workspace.cache import JudgmentCache

# Error codes the runner acts on (docs/spec/v0.md §7.5).
CLASSIFIER_REJECTED = "classifier_rejected"  # a 422: the Job auto-cancels
CLASSIFIER_UNAVAILABLE = "classifier_unavailable"  # persistent 5xx: the Document fails
OVER_BUDGET = "over_budget"  # a 400 after the shrink: the Document fails
_FAILING_CODES = frozenset({CLASSIFIER_UNAVAILABLE, OVER_BUDGET})
_SEAM_CODES = {
    "ClassifierRejected": CLASSIFIER_REJECTED,
    "ClassifierUnavailable": CLASSIFIER_UNAVAILABLE,
    "ClassifierTokensExceeded": OVER_BUDGET,
}
_SEAM_MESSAGES = {
    CLASSIFIER_REJECTED: "the Classifier rejected the request",
    CLASSIFIER_UNAVAILABLE: "classifier unavailable",
    OVER_BUDGET: "over budget: the state is too large for the Classifier",
}
SUFFIX_FORMATS = {".pdf": "pdf", ".html": "html", ".htm": "html"}
_GLOB = ("*", "?", "[")


class ExtractionFailed(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def expand_inputs(spec: JobSpec) -> JobSpec:
    """Replace each glob input with the files it matches, carrying its overrides along."""
    inputs = []
    overrides = dict(spec.overrides)
    for document in spec.inputs:
        source = document.source
        if "://" in source or Path(source).exists() or not any(c in source for c in _GLOB):
            inputs.append(document)
            continue
        matches = sorted(m for m in glob.glob(source, recursive=True) if Path(m).is_file())
        if not matches:
            inputs.append(document)  # reported when this Document fails to extract
            continue
        inputs += [document.model_copy(update={"source": m}) for m in matches]
        if source in overrides:
            override = overrides.pop(source)
            overrides.update({m: override for m in matches})
    return spec.model_copy(update={"inputs": inputs, "overrides": overrides})


def document_format(source: str) -> str | None:
    """`pdf` or `html` from the file name, None for anything else (URLs included)."""
    return SUFFIX_FORMATS.get(Path(source).suffix.lower())


def make_extract_task(
    path: Path, fmt: str, ocr_languages: Sequence[str], workspace: Path
) -> ExtractTask:
    """One Document's extraction task: its OCR languages, and the Workspace's language packs
    (installed by `ocr add-language`) searched after the vendored one."""
    return ExtractTask(
        path,
        "html" if fmt == "html" else "pdf",
        tuple(ocr_languages),
        (workspace_tessdata(workspace),),
    )


def judgments_of(manifest: ResolvedManifest) -> list[JudgmentSpec]:
    """The Taxonomy's and every Question's Judgment, Question Sets first (spec §5.1)."""
    questions = [q for s in manifest.question_sets for q in s.questions] + manifest.questions
    return judgment_specs(manifest.taxonomy, questions)


def new_record(
    job_id: str,
    document: DocumentInput,
    ref: InputRef | None,
    profile: ExtractionProfile,
    taxonomy: Taxonomy | None,
    attempts: int = 1,
) -> DocumentRecord:
    fmt = document_format(document.source) or "pdf"
    return DocumentRecord(
        record_id=uuid.uuid4().hex,
        job_id=job_id,
        external_id=document.external_id,
        user_metadata=document.user_metadata,
        status="ok",
        attempts=attempts,
        source=Source(
            filename=Path(document.source).name,
            format="html" if fmt == "html" else "pdf",
            sha256=ref.sha256 if ref else "",
            bytes=ref.bytes if ref else 0,
        ),
        extraction_profile=profile,
        taxonomy=TaxonomyRef(name=taxonomy.name, hash=taxonomy.hash) if taxonomy else None,
    )


def failed(record: DocumentRecord, stage: str, code: str, message: str) -> DocumentRecord:
    record.status = "failed"
    record.errors.append(
        RecordError(
            stage="classify" if stage == "classify" else "extract", code=code, message=message
        )
    )
    return record


def has_extraction(record: DocumentRecord | None) -> bool:
    """Whether a stored Record already holds a good extraction, so OCR is never redone."""
    return (
        record is not None
        and record.text is not None
        and not any(
            e.stage == "extract" and e.code != "section_detection_failed" for e in record.errors
        )
    )


def _ignore_page_counts(pages: int, ocr_pages: int) -> None:
    return None


async def extract(task: ExtractTask, pool: WorkerPool | None) -> Extracted:
    """Extract in a pool worker when there is one, else in a thread of this process.

    A pool worker can be killed, so a cancelled Job stops at once. A thread cannot be: with no
    pool (the library default) a cancel waits for the Document being extracted to finish. The
    CLI always uses a pool.
    """
    if pool is None:
        result: Extracted = await asyncio.to_thread(extract_document, task, _ignore_page_counts)
        return result
    outcome = await asyncio.wrap_future(pool.submit(*task.for_pool()))
    if not outcome.ok:
        raise ExtractionFailed(outcome.code or "extract_error", outcome.message or "")
    pooled: Extracted = outcome.result
    return pooled


async def detect(path: Path, fmt: str, extracted: Extracted) -> list[Section]:
    """Detect Sections in the extracted text (spec §4.4), off the event loop."""

    def work() -> list[Section]:
        headings: tuple[Heading, ...] = ()
        html = None
        if fmt == "html":
            html = path.read_text(encoding="utf-8", errors="replace")
        else:
            try:
                headings = tuple(
                    outline_headings(pdf_outline(path), extracted.pages, extracted.text)
                )
            except Exception:  # a broken outline just means no outline headings
                headings = ()
        return detect_sections(DetectionInput(text=extracted.text, html=html, headings=headings))

    return await asyncio.to_thread(work)


type Extractor = Callable[[ExtractTask, WorkerPool | None], Awaitable[Extracted]]


async def extract_stage(
    record: DocumentRecord,
    task: ExtractTask,
    path: Path,
    pool: WorkerPool | None,
    extractor: Extractor = extract,
) -> DocumentRecord:
    """extracting: Pages, text and Sections into the Record. Failures still yield a Record."""
    started = time.perf_counter()
    try:
        extracted = await extractor(task, pool)
    except Exception as exc:  # any extraction failure still yields a Record
        code = exc.code if isinstance(exc, ExtractionFailed) else type(exc).__name__
        return failed(record, "extract", code, str(exc))
    record.timings.extract_ms = round((time.perf_counter() - started) * 1000)
    record.timings.ocr_ms = extracted.ocr_ms
    record.usage.ocr_pages = extracted.ocr_pages
    record.text, record.pages = extracted.text, extracted.pages
    if extracted.title:
        kind = "html-title" if task.format == "html" else "pdf-metadata"
        record.metadata["title"] = MetadataField(value=extracted.title, source=kind)
    try:
        record.sections = await detect(path, task.format, extracted)
    except Exception as exc:  # the Document keeps its text; Questions with `reads` will skip
        record.status = "partial"
        record.errors.append(
            RecordError(stage="extract", code="section_detection_failed", message=str(exc))
        )
    # SECTION DETECTION ENDS HERE. Heading classification by Jev (method `heading_classified`)
    # also belongs in this stage once #37 lands; it needs the Classifier, so it takes one here.
    return record


async def verify_sections(
    record: DocumentRecord, manifest: ResolvedManifest, classifier: Classifier
) -> list[Section]:
    """SEAM for Section Verification (#37): between detection and planning.

    Runs at the start of the classifying stage, so it can use the Classifier and its result lands
    in the same Record write as the Judgments. It returns the Record's Sections with
    `verification` set, and only for keys some Question `reads`. Until #37 it changes nothing, and
    the Planner treats an unverified Section as present (spec §14).
    """
    return record.sections


@dataclass
class ClassifyResult:
    record: DocumentRecord
    rejected: bool = False  # a 422: the Job must auto-cancel (spec §7.5)


def view_of(record: DocumentRecord) -> DocumentView:
    title = record.metadata.get("title")
    return DocumentView(
        text=record.text or "",
        pages=record.pages,
        sections=record.sections,
        title=str(title.value) if title and title.value else Path(record.source.filename).stem,
    )


async def classify_stage(
    record: DocumentRecord,
    manifest: ResolvedManifest,
    specs: Sequence[JudgmentSpec],
    classifier: Classifier,
    cache: JudgmentCache | None,
) -> ClassifyResult:
    """classifying: Judge the Document, mapping Classifier failures per spec §7.5."""
    started = time.perf_counter()
    record.sections = await verify_sections(record, manifest, classifier)
    outcome = await judge_document(view_of(record), specs, classifier, cache)
    record.classification = outcome.classification
    record.answers = outcome.answers
    record.usage.requests += outcome.usage.requests
    record.usage.input_tokens += outcome.usage.input_tokens
    record.usage.output_tokens += outcome.usage.output_tokens
    record.usage.cache_hits += outcome.usage.cache_hits
    record.errors = [e for e in record.errors if e.stage != "classify"]
    for error in outcome.errors:
        code = _SEAM_CODES.get(error.code, error.code)
        message = _SEAM_MESSAGES.get(code, error.message)
        record.errors.append(RecordError(stage="classify", code=code, message=message))
    codes = {e.code for e in record.errors if e.stage == "classify"}
    if CLASSIFIER_REJECTED in codes:
        record.status = "cancelled"
    elif codes & _FAILING_CODES:
        record.status = "failed"
    else:
        record.status = "partial" if record.errors else "ok"
    record.timings.classify_ms += round((time.perf_counter() - started) * 1000)
    return ClassifyResult(record, rejected=CLASSIFIER_REJECTED in codes)
