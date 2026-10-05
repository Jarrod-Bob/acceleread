# SPDX-License-Identifier: Apache-2.0
"""Reading the PDF outline into heading candidates (docs/spec/v0.md §4.4)."""

from acceleread.sections.detector import Heading
from acceleread.sections.outline import outline_headings, pdf_outline


def test_outline_entries_become_top_level_headings_at_their_page() -> None:
    from acceleread.models import Page

    text = "cover\n\nBusiness overview\nwe sell\n\nNotes\nmore"
    pages = [
        Page(number=1, start=0, end=5),
        Page(number=2, start=7, end=32),
        Page(number=3, start=34, end=len(text)),
    ]
    outline = [(0, "Business overview", 1), (1, "Sub topic", 1), (0, "Notes", 2)]
    headings = outline_headings(outline, pages, text)
    assert headings == [
        Heading("Business overview", text.index("Business overview"), 0),
        Heading("Sub topic", pages[1].start, 1),
        Heading("Notes", text.index("Notes"), 0),
    ]


def test_outline_entries_pointing_past_the_pages_are_dropped() -> None:
    from acceleread.models import Page

    assert outline_headings([(0, "Gone", 9)], [Page(number=1, start=0, end=3)], "abc") == []


def test_a_pdf_without_bookmarks_has_an_empty_outline() -> None:
    from pathlib import Path

    assert pdf_outline(Path(__file__).parent / "fixtures" / "sample.pdf") == []
