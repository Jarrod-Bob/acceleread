# SPDX-License-Identifier: Apache-2.0
"""The `quality` Extraction Profile: Docling with RapidOCR on ONNX Runtime (docs/spec/v0.md §4.1).

Docling runs layout analysis on every Page, but OCR only on the Pages the shared OCR rule flags
(`ocr_rule.py`, evaluated from pypdfium2 signals exactly as in `fast`). Runs of consecutive Pages
are converted together: unflagged runs with OCR off, flagged runs with full-Page RapidOCR. Tables
are off unless asked for. Docling's `section_header` items come back as `Extracted.headings`, the
heading list the Section detectors read.

The Profile always runs on pinned, hash-verified weights from `acceleread models fetch`
(`quality_models.py`), online or offline: if they are absent or fail verification the Document
fails with `ModelsMissing`, and Docling is pointed at the Workspace's directory so it never
downloads anything itself.

Importing this module is cheap and works without the `[quality]` extra: Docling is imported only
when a Document is extracted. Converters are built once per process and reused.

Limits of what Docling reports:
- `Extracted.ocr_ms` is the wall time of converting the runs that contain OCR Pages. Docling does
  not separate OCR from layout analysis inside a run, so it includes both.
- An item that spans Pages is split between them by the character spans in its provenance.
"""

import importlib
import importlib.util
import os
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import version
from itertools import groupby
from pathlib import Path
from typing import Any, Final

import pypdfium2 as pdfium

from acceleread import quality_models
from acceleread.extract import (
    PAGE_SEPARATOR,
    Extracted,
    OcrLanguageUnavailable,
    PageCountsCallback,
    PageLayer,
    scan_pages,
)
from acceleread.languages import RAPIDOCR_CODES
from acceleread.models import Page
from acceleread.ocr_rule import Step3Hook
from acceleread.quality_models import check_model, models_dir
from acceleread.sections.detector import Heading

RAPIDOCR_BACKEND: Final = "onnxruntime"  # pinned: never `auto`, which can pick a torch backend
DEFAULT_THREADS = 4


class QualityUnavailable(Exception):
    """The `[quality]` extra is not installed."""


class ModelsMissing(Exception):
    """The pinned model weights are not in the Workspace, or fail their hash check."""


@dataclass(frozen=True)
class DoclingItem:
    """Text Docling found on one Page, in reading order. `heading_level` is 0-based for headings."""

    page: int
    text: str
    heading_level: int | None


@dataclass(frozen=True)
class Assembled:
    text: str
    pages: list[Page]
    headings: list[Heading]


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


def _verified_weights(workspace: Path | None, tables: bool) -> Path:
    """The Workspace's model directory, once every needed weight is present and verified."""
    if workspace is None:
        raise ModelsMissing(
            "the quality Profile needs its model weights in the Workspace; "
            "run `acceleread models fetch`"
        )
    for pin in quality_models.PINS:
        if pin.optional and not tables:
            continue
        state = check_model(workspace, pin)
        if state != "ok":
            problem = "is damaged or incomplete" if state == "corrupt" else "is not installed"
            fetch = "acceleread models fetch" + (" --tables" if pin.optional else "")
            raise ModelsMissing(
                f"the quality Profile's {pin.label} weights {problem}; run `{fetch}`"
            )
    return models_dir(workspace)


# One converter per (weights, OCR family, OCR on, tables) for the life of the process: building
# one loads the models, which costs seconds.
_converters: dict[tuple[Any, ...], Any] = {}


def _new_converter(options: Any) -> Any:
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def _converter(artifacts: Path, languages: Sequence[str], *, ocr: bool, tables: bool) -> Any:
    key = (
        str(artifacts),
        tuple(sorted({RAPIDOCR_CODES.get(c, c) for c in languages})),
        ocr,
        tables,
    )
    if key not in _converters:
        _converters[key] = _new_converter(
            pipeline_options(artifacts, languages, ocr=ocr, tables=tables)
        )
    return _converters[key]


def reset_converters() -> None:
    _converters.clear()


def docling_items(doc: Any) -> list[DoclingItem]:
    """A DoclingDocument's text in reading order, one entry per Page each item touches.

    An item whose provenance lists several Pages is cut at the character spans Docling records;
    when the spans are missing, overlap or all cover the whole text, it goes to its first Page.
    """
    # By module, not `from ... import`: mypy sees different exports with and without the extra.
    types = importlib.import_module("docling_core.types.doc")
    SectionHeaderItem, TableItem = types.SectionHeaderItem, types.TableItem

    found: list[DoclingItem] = []
    for item, _ in doc.iterate_items():
        if not item.prov:
            continue
        table = isinstance(item, TableItem)
        text = (item.export_to_markdown(doc) if table else getattr(item, "text", "")) or ""
        text = str(text).strip()
        if not text:
            continue
        level = item.level - 1 if isinstance(item, SectionHeaderItem) else None
        segments = [(item.prov[0].page_no, text)]
        spans = [tuple(p.charspan) for p in item.prov]
        usable = (
            not table
            and len(spans) > 1
            and len(set(spans)) == len(spans)
            and all(0 <= s < e <= len(text) for s, e in spans)
        )
        if usable:
            segments = [
                (p.page_no, text[s:e].strip()) for p, (s, e) in zip(item.prov, spans, strict=True)
            ]
        found += [
            DoclingItem(page, part, level if i == 0 else None)
            for i, (page, part) in enumerate(segments)
            if part
        ]
    return found


def assemble(
    layers: Sequence[PageLayer],
    items: Sequence[DoclingItem],
    *,
    languages: Sequence[str],
    docling_version: str,
    rapidocr_version: str,
) -> Assembled:
    """Place Docling's items into Pages, offsets and headings.

    A Page Docling returned nothing for keeps its pypdfium2 text layer, and its provenance says
    `pdfium`. An OCR Page is `rapidocr`'s, even when it came back empty.
    """
    by_page: dict[int, list[DoclingItem]] = defaultdict(list)
    for item in items:
        by_page[item.page].append(item)
    parts: list[str] = []
    pages: list[Page] = []
    headings: list[Heading] = []
    offset = 0
    for layer in layers:
        record = layer.record
        if parts:
            offset += len(PAGE_SEPARATOR)
        start = offset
        lines: list[str] = []
        for item in by_page.get(record.number, []):
            if item.heading_level is not None:
                headings.append(
                    Heading(item.text, start + sum(len(x) + 1 for x in lines), item.heading_level)
                )
            lines.append(item.text)
        text = "\n".join(lines)
        update: dict[str, Any] = {}
        if layer.needs_ocr:
            update = {
                "method": "ocr-full",
                "engine": "rapidocr",
                "engine_version": rapidocr_version,
                "ocr_languages": list(languages),
            }
        elif lines:
            update = {"engine": "docling", "engine_version": docling_version}
        else:
            text = layer.text  # Docling found nothing: keep the text layer, say so
        pages.append(record.model_copy(update=update | {"start": start, "end": start + len(text)}))
        parts.append(text)
        offset = start + len(text)
    return Assembled(PAGE_SEPARATOR.join(parts), pages, headings)


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
    """Extract a PDF with Docling, running OCR only on Pages the OCR rule flags.

    Raises `ModelsMissing` unless the Workspace holds hash-verified weights (and TableFormer's,
    with `tables`), `QualityUnavailable` without the `[quality]` extra, and `OcrLanguageUnavailable`
    when a Page needs OCR in a language the PP-OCR latin family can't read.
    """
    languages = list(ocr_languages)
    pdf = pdfium.PdfDocument(path)
    try:
        layers = scan_pages(pdf, step3)
        title = pdf.get_metadata_dict().get("Title") or None
    finally:
        pdf.close()
    flags = [layer.needs_ocr for layer in layers]
    if on_page_counts is not None:
        on_page_counts(len(layers), sum(flags))
    if any(flags):
        for code in languages:
            if code not in RAPIDOCR_CODES:
                raise OcrLanguageUnavailable(
                    f"The quality Profile can't OCR {code!r} (PP-OCR latin family only)"
                )
    artifacts = _verified_weights(workspace, tables)
    if not docling_available():
        raise QualityUnavailable(
            "the quality Profile needs the extra: pip install acceleread[quality]"
        )

    items: list[DoclingItem] = []
    ocr_ms = 0
    for run in _runs(flags):
        converter = _converter(artifacts, languages, ocr=run.ocr, tables=tables)
        started = time.perf_counter()
        result = converter.convert(path, page_range=(run.first, run.last))
        if run.ocr:
            ocr_ms += round((time.perf_counter() - started) * 1000)
        items += docling_items(result.document)

    assembled = assemble(
        layers,
        items,
        languages=languages,
        docling_version=version("docling-slim"),
        rapidocr_version=version("rapidocr") if any(flags) else "",
    )
    return Extracted(
        text=assembled.text,
        pages=assembled.pages,
        title=title,
        ocr_pages=sum(flags),
        ocr_ms=ocr_ms,
        headings=tuple(assembled.headings),
    )
