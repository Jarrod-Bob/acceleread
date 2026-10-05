# SPDX-License-Identifier: Apache-2.0
"""Annual-report synonym matching on headings and the PDF outline (docs/spec/v0.md §4.4)."""

import pytest

from acceleread.sections.detector import DetectionInput, Heading
from acceleread.sections.synonyms import SynonymDetector, match_key, unmatched_headings

BODY = "The group performed well across all of its regions this year. " * 5


@pytest.mark.parametrize(
    ("heading", "key"),
    [
        ("Strategic report", "business"),
        ("Business overview", "business"),
        ("Our business", "business"),
        ("Principal risks and uncertainties", "risk_factors"),
        ("Principal risks", "risk_factors"),
        ("Risk factors", "risk_factors"),
        ("Cybersecurity", "cybersecurity"),
        ("Legal proceedings", "legal_proceedings"),
        ("Litigation", "legal_proceedings"),
        ("Operating and financial review", "mdna"),
        ("Financial review", "mdna"),
        ("Management's discussion and analysis of results", "mdna"),
        ("Market risk", "market_risk"),
        ("Financial risk management", "market_risk"),
        ("Consolidated financial statements", "financial_statements"),
        ("Financial statements", "financial_statements"),
        ("Internal controls", "controls"),
        ("Internal control", "controls"),
        ("Controls and procedures", "controls"),
        ("Directors' report", "governance"),
        ("Corporate governance", "governance"),
        ("Remuneration report", "governance"),
    ],
)
def test_headings_match_the_synonym_list(heading: str, key: str) -> None:
    assert match_key(heading) == key


@pytest.mark.parametrize(
    "heading",
    [
        "STRATEGIC REPORT",
        "3. Strategic report",
        "Strategic Report 12",
        "Strategic report ........ 12",
        "Directors\u2019 report",
        "  Risk   factors ",
    ],
)
def test_matching_ignores_case_numbering_page_numbers_and_punctuation(heading: str) -> None:
    assert match_key(heading) is not None


@pytest.mark.parametrize("heading", ["Chairman's letter", "Our people", "Notes", ""])
def test_unrelated_headings_do_not_match(heading: str) -> None:
    assert match_key(heading) is None


def test_matched_headings_become_sections_running_to_the_next_heading() -> None:
    text = f"Strategic report\n{BODY}\nOur people\n{BODY}\nRisk factors\n{BODY}\n"
    headings = [
        Heading("Strategic report", text.index("Strategic")),
        Heading("Our people", text.index("Our people")),
        Heading("Risk factors", text.index("Risk factors")),
    ]
    sections = SynonymDetector().detect(DetectionInput(text=text, headings=headings))
    assert [s.keys for s in sections] == [["business"], ["risk_factors"]]
    business, risk = sections
    assert business.method == "heading_synonym"
    assert business.label == "Strategic report"
    assert text[business.spans[0].start : business.spans[0].end].endswith(BODY.strip())
    assert "Our people" not in text[business.spans[0].start : business.spans[0].end]
    assert text[risk.spans[0].start : risk.spans[0].end].endswith(BODY.strip())
    assert risk.confidence is not None and risk.confidence < 0.9


def test_a_key_can_have_several_sections() -> None:
    text = f"Strategic report\n{BODY}\nBusiness overview\n{BODY}\n"
    headings = [
        Heading("Strategic report", 0),
        Heading("Business overview", text.index("Business overview")),
    ]
    sections = SynonymDetector().detect(DetectionInput(text=text, headings=headings))
    assert [s.keys for s in sections] == [["business"], ["business"]]


def test_nested_headings_do_not_end_a_section() -> None:
    text = f"Strategic report\n{BODY}\nMarket overview\n{BODY}\nGovernance Report\n{BODY}\n"
    headings = [
        Heading("Strategic report", 0, level=0),
        Heading("Market overview", text.index("Market overview"), level=1),
        Heading("Corporate governance", text.index("Governance Report"), level=0),
    ]
    business, _ = SynonymDetector().detect(DetectionInput(text=text, headings=headings))
    assert "Market overview" in text[business.spans[0].start : business.spans[0].end]


def test_unmatched_headings_are_left_for_the_classifier_step() -> None:
    headings = [Heading("Strategic report", 0), Heading("Our people", 10), Heading("Notes", 20)]
    left = unmatched_headings(DetectionInput(text="x" * 30, headings=headings))
    assert [h.text for h in left] == ["Our people", "Notes"]


def test_without_a_heading_list_whole_line_headings_in_the_text_are_used() -> None:
    toc = "Contents\nStrategic report 3\nFinancial review 9\n"
    text = f"{toc}Strategic report\n{BODY}\nFinancial review\n{BODY}\n"
    sections = SynonymDetector().detect(DetectionInput(text=text))
    assert [s.keys for s in sections] == [["business"], ["mdna"]]
    assert text[sections[0].spans[0].start :].startswith("Strategic report\nThe group")


def test_prose_lines_are_not_headings() -> None:
    text = f"The strategic report is on page 3 of this document.\n{BODY}\n"
    assert SynonymDetector().detect(DetectionInput(text=text)) == []


def test_short_lines_inside_a_section_do_not_end_it() -> None:
    text = f"Strategic report\n{BODY}\nHighlights\nRevenue up 4%\n{BODY}\n"
    (section,) = SynonymDetector().detect(DetectionInput(text=text))
    assert text[section.spans[0].start : section.spans[0].end].endswith(BODY.strip())
    assert "Revenue up 4%" in text[section.spans[0].start : section.spans[0].end]


def test_outline_entries_become_top_level_headings_at_their_page() -> None:
    from acceleread.models import Page
    from acceleread.sections.synonyms import outline_headings

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
    from acceleread.sections.synonyms import outline_headings

    assert outline_headings([(0, "Gone", 9)], [Page(number=1, start=0, end=3)], "abc") == []


def test_a_pdf_without_bookmarks_has_an_empty_outline() -> None:
    from pathlib import Path

    from acceleread.sections.synonyms import pdf_outline

    assert pdf_outline(Path(__file__).parent / "fixtures" / "sample.pdf") == []
