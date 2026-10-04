# SPDX-License-Identifier: Apache-2.0
"""PDF text-layer Extraction with pypdfium2 (docs/spec/v0.md §4). No OCR yet."""

from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium

from acceleread.models import Page

PAGE_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class Extracted:
    text: str
    pages: list[Page]
    title: str | None


def extract_pdf(path: Path) -> Extracted:
    """Concatenate each Page's text layer; Pages are character ranges into the Document text."""
    pdf = pdfium.PdfDocument(path)
    try:
        parts: list[str] = []
        pages: list[Page] = []
        offset = 0
        for number, page in enumerate(pdf, start=1):
            textpage = page.get_textpage()
            text = textpage.get_text_range().replace("\r\n", "\n").strip()
            textpage.close()
            page.close()
            if parts:
                offset += len(PAGE_SEPARATOR)
            pages.append(Page(number=number, start=offset, end=offset + len(text)))
            parts.append(text)
            offset += len(text)
        title = pdf.get_metadata_dict().get("Title") or None
        return Extracted(text=PAGE_SEPARATOR.join(parts), pages=pages, title=title)
    finally:
        pdf.close()
