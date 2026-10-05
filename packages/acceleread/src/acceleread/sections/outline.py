# SPDX-License-Identifier: Apache-2.0
"""The PDF outline (bookmarks) as heading candidates for the synonym matcher."""

from collections.abc import Sequence
from pathlib import Path

import pypdfium2 as pdfium

from acceleread.models import Page
from acceleread.sections.detector import Heading

OutlineEntry = tuple[int, str, int]
"""(level from 0, title, zero-based page index), as the PDF outline lists them."""


def pdf_outline(path: Path) -> list[OutlineEntry]:
    """The PDF's bookmarks, flattened. Empty when it has none."""
    pdf = pdfium.PdfDocument(path)
    try:
        return [
            (item.level, item.title, item.page_index)
            for item in pdf.get_toc()
            if item.page_index is not None
        ]
    finally:
        pdf.close()


def outline_headings(
    outline: Sequence[OutlineEntry], pages: Sequence[Page], text: str
) -> list[Heading]:
    """Outline entries as headings, positioned where the title appears on its Page.

    An entry whose title can't be found on the Page starts at the Page's start; one that points
    past the extracted Pages is dropped.
    """
    by_index = {p.number - 1: p for p in pages}
    headings = []
    for level, title, page_index in outline:
        page = by_index.get(page_index)
        if page is None:
            continue
        at = text.find(title, page.start, page.end)
        headings.append(Heading(title, at if at >= 0 else page.start, level))
    return headings
