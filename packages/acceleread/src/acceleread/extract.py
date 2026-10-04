# SPDX-License-Identifier: Apache-2.0
"""Local Extraction (docs/spec/v0.md §4): PDF text layer plus Tesseract OCR on flagged Pages, and
plain HTML text.

The `fast` Profile only: each Page's pypdfium2 signals go through the OCR rule (`ocr_rule.py`) and
Pages it flags are rendered and recognised with Tesseract. Extraction never touches the network.
"""

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_raw
from lxml import html

from acceleread.models import Page
from acceleread.ocr_rule import PageSignals, Step3Hook, decide

PAGE_SEPARATOR = "\n\n"
OCR_DPI = 300
VENDORED_TESSDATA = Path(__file__).parent / "tessdata"
# ISO 639-1 (what a Job's `ocr_languages` holds) to Tesseract language packs (spec §4.3).
TESSERACT_LANGUAGES = {
    "en": "eng",
    "de": "deu",
    "fr": "fra",
    "es": "spa",
    "it": "ita",
    "pt": "por",
    "nl": "nld",
}


class OcrLanguageUnavailable(Exception):
    """An OCR language has no installed Tesseract pack."""


@dataclass(frozen=True)
class Extracted:
    text: str
    pages: list[Page]
    title: str | None
    ocr_pages: int = 0
    ocr_ms: int = 0


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


def _tesseract_languages(languages: Sequence[str], tessdata: Path) -> str:
    packs = []
    for code in languages:
        pack = TESSERACT_LANGUAGES.get(code, code)
        if not (tessdata / f"{pack}.traineddata").exists():
            raise OcrLanguageUnavailable(f"No Tesseract language pack installed for {code!r}")
        packs.append(pack)
    return "+".join(packs)


class TesseractOcr:
    """Tesseract through tesserocr on rendered Pages. Create lazily: importing loads the engine."""

    def __init__(self, languages: Sequence[str], tessdata: Path) -> None:
        import tesserocr

        self._api = tesserocr.PyTessBaseAPI(
            path=str(tessdata),
            lang=_tesseract_languages(languages, tessdata),
            psm=tesserocr.PSM.AUTO,
        )
        self.version = tesserocr.tesseract_version().splitlines()[0].removeprefix("tesseract ")

    def recognise(self, page: Any) -> tuple[str, float | None]:
        bitmap = page.render(scale=OCR_DPI / 72, grayscale=True)
        self._api.SetImageBytes(
            bytes(bitmap.buffer), bitmap.width, bitmap.height, bitmap.n_channels, bitmap.stride
        )
        text = self._api.GetUTF8Text().replace("\r\n", "\n").strip()
        confidence = self._api.MeanTextConf()
        return text, (confidence / 100 if text else None)

    def close(self) -> None:
        self._api.End()


def extract_pdf(
    path: Path,
    ocr_languages: Sequence[str] = ("en",),
    step3: Step3Hook | None = None,
    tessdata: Path = VENDORED_TESSDATA,
    on_ocr_pages: Any = None,
) -> Extracted:
    """Concatenate each Page's text, from the text layer or OCR as the OCR rule decides.

    Pages are character ranges into the Document text. `on_ocr_pages(n)` is called once the rule
    has run on every Page, with the number of Pages that will be OCRed (workers use it to size the
    Document timeout).
    """
    languages = list(ocr_languages)
    _tesseract_languages(languages, tessdata)  # fail before any work if a pack is missing
    pdf = pdfium.PdfDocument(path)
    ocr: TesseractOcr | None = None
    try:
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
        flagged = sum(ocr_needed for _, _, ocr_needed in layers)
        if on_ocr_pages is not None:
            on_ocr_pages(flagged)

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
