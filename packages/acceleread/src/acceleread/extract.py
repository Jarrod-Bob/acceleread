# SPDX-License-Identifier: Apache-2.0
"""Local Extraction (docs/spec/v0.md §4): PDF text layer plus Tesseract OCR on flagged Pages, and
plain HTML text.

The `fast` Profile only: each Page's pypdfium2 signals go through the OCR rule (`ocr_rule.py`) and
Pages it flags are rendered and recognised with Tesseract. Extraction never touches the network.
"""

import re
import tempfile
import time
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_raw
from lxml import html

from acceleread.languages import TESSERACT_CODES, pack_file
from acceleread.models import ExtractionProfile, JobSettings, Page
from acceleread.ocr_rule import PageSignals, Step3Hook, decide
from acceleread.sections.detector import Heading

PageCountsCallback = Callable[[int, int], None]
"""Called with (pages in the Document, pages that will be OCRed)."""

PAGE_SEPARATOR = "\n\n"
OCR_DPI = 300
VENDORED_TESSDATA = Path(__file__).parent / "tessdata"


class OcrLanguageUnavailable(Exception):
    """An OCR language has no installed Tesseract pack."""


@dataclass(frozen=True)
class Extracted:
    text: str
    pages: list[Page]
    title: str | None
    ocr_pages: int = 0
    ocr_ms: int = 0
    headings: tuple[Heading, ...] = ()
    """Heading candidates with offsets into `text`: Docling `section_header` items (`quality`)."""


def image_coverage(page: Any) -> float:
    """Fraction of the Page covered by the union of its image boxes."""
    width, height = page.get_size()
    boxes = []
    for obj in page.get_objects(filter=[pdfium_raw.FPDF_PAGEOBJ_IMAGE]):
        left, bottom, right, top = obj.get_bounds()
        left, right = max(0.0, left), min(width, right)
        bottom, top = max(0.0, bottom), min(height, top)
        if right > left and top > bottom:
            boxes.append((left, bottom, right, top))
    if not boxes or width <= 0 or height <= 0:
        return 0.0
    xs = sorted({x for left, _, right, _ in boxes for x in (left, right)})
    area = 0.0
    for x0, x1 in pairwise(xs):
        spans = sorted((b, t) for left, b, right, t in boxes if left <= x0 and right >= x1)
        covered, edge = 0.0, 0.0
        for bottom, top in spans:
            bottom = max(bottom, edge)
            if top > bottom:
                covered += top - bottom
                edge = top
        area += covered * (x1 - x0)
    return float(min(1.0, area / (width * height)))


def path_count(page: Any) -> int:
    return sum(1 for _ in page.get_objects(filter=[pdfium_raw.FPDF_PAGEOBJ_PATH]))


def tesseract_version() -> str:
    import tesserocr

    version: str = tesserocr.tesseract_version()
    return version.splitlines()[0].removeprefix("tesseract ")


def _tesseract_packs(languages: Sequence[str], tessdata: Sequence[Path]) -> dict[str, Path]:
    """Each requested language's pack file, found in the first directory that has it."""
    packs: dict[str, Path] = {}
    for code in languages:
        if code not in TESSERACT_CODES:
            raise OcrLanguageUnavailable(f"Unknown OCR language {code!r}: use an ISO 639-1 code")
        found = next((p for p in (pack_file(d, code) for d in tessdata) if p.is_file()), None)
        if found is None:
            raise OcrLanguageUnavailable(f"No Tesseract language pack installed for {code!r}")
        packs[TESSERACT_CODES[code]] = found
    return packs


class TesseractOcr:
    """Tesseract through tesserocr on rendered Pages. Create lazily: importing loads the engine."""

    def __init__(self, languages: Sequence[str], tessdata: Sequence[Path]) -> None:
        import tesserocr

        packs = _tesseract_packs(languages, tessdata)
        with ExitStack() as stack:
            # Tesseract reads one directory, so packs from the vendored and Workspace directories
            # are linked together.
            links = stack.enter_context(tempfile.TemporaryDirectory(prefix="acceleread-tessdata-"))
            for pack, source in packs.items():
                (Path(links) / f"{pack}.traineddata").symlink_to(source)
            api = tesserocr.PyTessBaseAPI(path=links, lang="+".join(packs), psm=tesserocr.PSM.AUTO)
            stack.callback(api.End)  # runs before the directory goes
            self._api = api
            self.version = tesseract_version()
            self._cleanup = stack.pop_all()

    def recognise(self, page: Any) -> tuple[str, float | None]:
        bitmap = page.render(scale=OCR_DPI / 72, grayscale=True)
        self._api.SetImageBytes(
            bytes(bitmap.buffer), bitmap.width, bitmap.height, bitmap.n_channels, bitmap.stride
        )
        text = self._api.GetUTF8Text().replace("\r\n", "\n").strip()
        confidence = self._api.MeanTextConf()
        return text, (confidence / 100 if text else None)

    def close(self) -> None:
        self._cleanup.close()


def scan_pages(pdf: Any, step3: Step3Hook | None = None) -> list[tuple[Page, str, bool]]:
    """Each Page's pypdfium2 text layer and the OCR rule's verdict, shared by both Profiles.

    Returns (Page record with `start`/`end` still 0, text layer, whether the rule flags the Page).
    """
    layers: list[tuple[Page, str, bool]] = []
    for number, page in enumerate(pdf, start=1):
        textpage = page.get_textpage()
        layer = textpage.get_text_range().replace("\r\n", "\n").strip()
        textpage.close()
        signals = PageSignals(layer, image_coverage(page), path_count(page))
        verdict = decide(signals, step3)
        record = Page(
            number=number,
            start=0,
            end=0,
            engine="pdfium",
            engine_version=str(pdfium.PDFIUM_INFO.build),
            ocr_decision=verdict.decision,
            image_coverage=signals.image_coverage,
        )
        layers.append((record, layer, verdict.ocr))
        page.close()
    return layers


def profile_for(settings: JobSettings, source: str | Path) -> ExtractionProfile:
    """The Extraction Profile for one input: its override when it has one, else the Job's."""
    override = settings.overrides.get(str(source))
    return (override.extraction_profile if override else None) or settings.extraction_profile


def extract_pdf(
    path: Path,
    ocr_languages: Sequence[str] = ("en",),
    step3: Step3Hook | None = None,
    tessdata: Sequence[Path] = (VENDORED_TESSDATA,),
    on_page_counts: PageCountsCallback | None = None,
) -> Extracted:
    """Concatenate each Page's text, from the text layer or OCR as the OCR rule decides.

    Pages are character ranges into the Document text. `on_page_counts(pages, ocr_pages)` is called
    once the rule has run on every Page (workers use it to size the Document timeout). A missing
    Tesseract language pack raises `OcrLanguageUnavailable` only when a Page needs OCR.
    """
    languages = list(ocr_languages)
    pdf = pdfium.PdfDocument(path)
    ocr: TesseractOcr | None = None
    try:
        layers = scan_pages(pdf, step3)
        flagged = sum(ocr_needed for _, _, ocr_needed in layers)
        if on_page_counts is not None:
            on_page_counts(len(layers), flagged)

        parts: list[str] = []
        pages: list[Page] = []
        offset = 0
        ocr_ms = 0
        for record, layer, ocr_needed in layers:
            text = layer
            if ocr_needed:
                started = time.perf_counter()
                ocr = ocr or TesseractOcr(languages, tessdata)
                page = pdf[record.number - 1]
                text, confidence = ocr.recognise(page)
                page.close()
                ocr_ms += round((time.perf_counter() - started) * 1000)
                record = record.model_copy(
                    update={
                        "method": "ocr-full",
                        "engine": "tesseract",
                        "engine_version": ocr.version,
                        "ocr_languages": languages,
                        "ocr_confidence": confidence,
                    }
                )
            if parts:
                offset += len(PAGE_SEPARATOR)
            pages.append(record.model_copy(update={"start": offset, "end": offset + len(text)}))
            parts.append(text)
            offset += len(text)
        title = pdf.get_metadata_dict().get("Title") or None
        return Extracted(
            text=PAGE_SEPARATOR.join(parts),
            pages=pages,
            title=title,
            ocr_pages=flagged,
            ocr_ms=ocr_ms,
        )
    finally:
        if ocr is not None:
            ocr.close()
        pdf.close()


BLOCK_TAGS = {
    "address", "article", "aside", "blockquote", "br", "dd", "div", "dl", "dt", "fieldset",
    "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header",
    "hr", "li", "main", "nav", "ol", "p", "pre", "section", "table", "tr", "ul",
}  # fmt: skip
SKIPPED_TAGS = {"script", "style", "noscript", "head", "template"}


def extract_html(path: Path) -> Extracted:
    """Plain text from HTML with lxml: block elements become line breaks, no Pages."""
    root = html.document_fromstring(path.read_bytes())
    title_nodes = root.xpath("//title")
    title = title_nodes[0].text_content().strip() if title_nodes else None

    pieces: list[str] = []

    def walk(node: Any) -> None:
        tag = node.tag if isinstance(node.tag, str) else None
        # A comment or processing instruction has no string tag. Skip it, but keep its tail.
        if tag is None or tag in SKIPPED_TAGS:
            pass
        else:
            block = tag in BLOCK_TAGS
            if block:
                pieces.append("\n")
            if node.text:
                pieces.append(node.text)
            for child in node:
                walk(child)
            if block:
                pieces.append("\n")
        if node.tail:
            pieces.append(node.tail)

    walk(root)
    raw = "".join(pieces).replace("\xa0", " ")
    lines = (re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in raw.split("\n"))
    text = "\n".join(line for line in lines if line)
    return Extracted(text=text, pages=[], title=title or None)
