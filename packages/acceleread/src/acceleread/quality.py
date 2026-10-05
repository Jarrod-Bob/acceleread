# SPDX-License-Identifier: Apache-2.0
"""The `quality` Extraction Profile: Docling with RapidOCR on ONNX Runtime (docs/spec/v0.md §4.1).

Docling runs layout analysis on every Page, but OCR only on the Pages the shared OCR rule flags
(`ocr_rule.py`, evaluated from pypdfium2 signals exactly as in `fast`). Runs of consecutive Pages
are converted together: unflagged runs with OCR off, flagged runs with full-Page RapidOCR. Tables
are off unless asked for. Docling's `section_header` items come back as `Extracted.headings`, the
heading list the Section detectors read.

Importing this module is cheap and works without the `[quality]` extra: Docling is imported only
when a Document is extracted. Weights come from the Workspace (`models fetch`); with
`ACCELEREAD_OFFLINE=1` a missing weight is an error rather than a download.
"""

import importlib.util
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import version
from itertools import groupby
from pathlib import Path
from typing import Any, Final

import pypdfium2 as pdfium

from acceleread.extract import (
    PAGE_SEPARATOR,
    Extracted,
    OcrLanguageUnavailable,
    PageCountsCallback,
    scan_pages,
)
from acceleread.languages import RAPIDOCR_CODES, offline
from acceleread.models import Page
from acceleread.ocr_rule import Step3Hook
from acceleread.quality_models import PINS, models_dir, models_status
from acceleread.sections.detector import Heading

RAPIDOCR_BACKEND: Final = "onnxruntime"  # pinned: never `auto`, which can pick a torch backend
DEFAULT_THREADS = 4


class QualityUnavailable(Exception):
    """The `[quality]` extra is not installed."""


class ModelsMissing(Exception):
    """Model weights are not in the Workspace and offline mode forbids downloading them."""


def docling_available() -> bool:
    return importlib.util.find_spec("docling") is not None


def pipeline_options(
    artifacts_path: Path | None,
    ocr_languages: Sequence[str],
    *,
    ocr: bool,
    tables: bool = False,
) -> Any:
    """Docling's PDF pipeline options for one run of Pages."""
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.pipeline_options import OcrMode, PdfPipelineOptions, RapidOcrOptions

    families = sorted({RAPIDOCR_CODES[code] for code in ocr_languages if code in RAPIDOCR_CODES})
    threads = int(os.environ.get("OMP_NUM_THREADS") or DEFAULT_THREADS)
    return PdfPipelineOptions(
        artifacts_path=artifacts_path,
        do_ocr=ocr,
        do_table_structure=tables,
        ocr_options=RapidOcrOptions(
            backend=RAPIDOCR_BACKEND, lang=families, mode=OcrMode.FULL_PAGE
        ),
        accelerator_options=AcceleratorOptions(device="cpu", num_threads=threads),
    )


def _artifacts_path(workspace: Path | None, tables: bool) -> Path | None:
    """The Workspace's model directory when it has the weights, else None (Docling's cache).

    Offline, missing weights are an error: nothing may be downloaded at runtime.
    """
    if workspace is not None:
        status = models_status(workspace)
        needed = [p.name for p in PINS if not p.optional or (tables and p.name == "tables")]
        if all(status[name] for name in needed):
            return models_dir(workspace)
    if offline():
        raise ModelsMissing(
            "the quality Profile's models are not installed and ACCELEREAD_OFFLINE=1 forbids "
            "downloading them; run `acceleread models fetch` first"
        )
    return None


def _text_of(item: Any, doc: Any) -> str:
    if type(item).__name__ == "TableItem":
        return str(item.export_to_markdown(doc)).strip()
    return str(getattr(item, "text", "") or "").strip()


@dataclass
class _Run:
    """A stretch of consecutive Pages converted together."""

    first: int
    last: int
    ocr: bool


def _runs(flags: Sequence[bool]) -> list[_Run]:
    runs, number = [], 1
    for flagged, group in groupby(flags):
        count = len(list(group))
        runs.append(_Run(number, number + count - 1, flagged))
        number += count
    return runs


def extract_pdf_quality(
    path: Path,
    ocr_languages: Sequence[str] = ("en",),
    step3: Step3Hook | None = None,
    workspace: Path | None = None,
    tables: bool = False,
    on_page_counts: PageCountsCallback | None = None,
) -> Extracted:
    """Extract a PDF with Docling, running OCR only on Pages the OCR rule flags."""
    languages = list(ocr_languages)
    pdf = pdfium.PdfDocument(path)
    try:
        layers = scan_pages(pdf, step3)
        title = pdf.get_metadata_dict().get("Title") or None
    finally:
        pdf.close()
    flags = [ocr_needed for _, _, ocr_needed in layers]
    if on_page_counts is not None:
        on_page_counts(len(layers), sum(flags))
    if any(flags):
        for code in languages:
            if code not in RAPIDOCR_CODES:
                raise OcrLanguageUnavailable(
                    f"The quality Profile can't OCR {code!r} (PP-OCR latin family only)"
                )
    artifacts = _artifacts_path(workspace, tables)
    if not docling_available():
        raise QualityUnavailable(
            "the quality Profile needs the extra: pip install acceleread[quality]"
        )
    if artifacts is not None or offline():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption

    converters: dict[bool, Any] = {}

    def convert(run: _Run) -> Any:
        if run.ocr not in converters:
            options = pipeline_options(artifacts, languages, ocr=run.ocr, tables=tables)
            converters[run.ocr] = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
            )
        result = converters[run.ocr].convert(path, page_range=(run.first, run.last))
        return result.document

    # Page number -> [(text, heading level or None)] in reading order.
    items: dict[int, list[tuple[str, int | None]]] = {}
    ocr_ms = 0
    for run in _runs(flags):
        started = time.perf_counter()
        doc = convert(run)
        if run.ocr:
            ocr_ms += round((time.perf_counter() - started) * 1000)
        for item, _ in doc.iterate_items():
            text = _text_of(item, doc)
            if not text or not item.prov:
                continue
            level = item.level - 1 if type(item).__name__ == "SectionHeaderItem" else None
            items.setdefault(item.prov[0].page_no, []).append((text, level))

    docling_version = version("docling-slim")
    parts: list[str] = []
    pages: list[Page] = []
    headings: list[Heading] = []
    offset = 0
    for record, _, ocr_needed in layers:
        if parts:
            offset += len(PAGE_SEPARATOR)
        page_start = offset
        lines: list[str] = []
        for text, level in items.get(record.number, []):
            if level is not None:
                headings.append(Heading(text, page_start + sum(len(x) + 1 for x in lines), level))
            lines.append(text)
        text = "\n".join(lines)
        update: dict[str, Any] = {"start": page_start, "end": page_start + len(text)}
        if ocr_needed:
            update |= {
                "method": "ocr-full",
                "engine": "rapidocr",
                "engine_version": version("rapidocr"),
                "ocr_languages": languages,
            }
        else:
            update |= {"engine": "docling", "engine_version": docling_version}
        pages.append(record.model_copy(update=update))
        parts.append(text)
        offset = page_start + len(text)
    return Extracted(
        text=PAGE_SEPARATOR.join(parts),
        pages=pages,
        title=title,
        ocr_pages=sum(flags),
        ocr_ms=ocr_ms,
        headings=tuple(headings),
    )
