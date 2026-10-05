# SPDX-License-Identifier: Apache-2.0
"""Choosing among detectors: the first one that finds Sections wins, and pointers get flagged."""

from acceleread.models import Section, Span
from acceleread.sections import detect_sections
from acceleread.sections.detector import DetectionInput, Heading

BODY = "Revenue grew and margins held across every segment of the company. " * 6


class StubDetector:
    extra_keys: frozenset[str] = frozenset()

    def __init__(self, name: str, sections: list[Section]) -> None:
        self.name = name
        self._sections = sections
        self.calls = 0

    def detect(self, inp: DetectionInput) -> list[Section]:
        self.calls += 1
        return self._sections


def a_section(key: str) -> Section:
    return Section(keys=[key], label=key, spans=[Span(start=0, end=5)], method="stub")


def test_the_first_detector_with_sections_wins_and_later_ones_are_not_run() -> None:
    empty = StubDetector("empty", [])
    first = StubDetector("first", [a_section("business")])
    last = StubDetector("last", [a_section("mdna")])
    found = detect_sections(DetectionInput(text="hello"), [empty, first, last])
    assert [s.keys for s in found] == [["business"]]
    assert (empty.calls, first.calls, last.calls) == (1, 1, 0)


def test_no_detector_finding_anything_gives_no_sections() -> None:
    assert detect_sections(DetectionInput(text="hello"), [StubDetector("e", [])]) == []


def test_the_default_chain_uses_the_item_regex_then_synonyms() -> None:
    items = f"Item 1A. Risk Factors\n{BODY}\n"
    found = detect_sections(DetectionInput(text=items, form="10-K"))
    assert [s.method for s in found] == ["item_regex"]

    report = f"Strategic report\n{BODY}\n"
    found = detect_sections(DetectionInput(text=report, form=None))
    assert [s.method for s in found] == ["heading_synonym"]


def test_a_supplied_heading_list_feeds_the_synonym_detector() -> None:
    text = f"Page one\n{BODY}\nFinancial review\n{BODY}\n"
    headings = [Heading("Financial review", text.index("Financial review"))]
    found = detect_sections(DetectionInput(text=text, headings=headings))
    assert [s.keys for s in found] == [["mdna"]]


def test_without_edgartools_or_html_the_default_chain_skips_it() -> None:
    items = f"Item 1. Business\n{BODY}\n"
    found = detect_sections(DetectionInput(text=items, form="10-K", html="<html/>"))
    assert found
    assert found[0].method in {"item_regex", "edgartools"}


def test_pointer_sections_come_back_flagged() -> None:
    text = (
        f"Item 1. Business\n{BODY}\n"
        "Item 7. MD&A\nIncorporated by reference to Exhibit 13.\n"
        f"Item 8. Financial Statements\n{BODY}\n"
    )
    found = detect_sections(DetectionInput(text=text, form="10-K"))
    flags = {s.form_ref: s.flags for s in found}
    assert flags == {"10-K 1": [], "10-K 7": ["pointer"], "10-K 8": []}


def test_keys_a_detector_did_not_declare_become_other() -> None:
    plain = StubDetector("plain", [a_section("esg")])
    assert detect_sections(DetectionInput(text="hello"), [plain])[0].keys == ["other"]
    declaring = StubDetector("declaring", [a_section("esg")])
    declaring.extra_keys = frozenset({"esg"})
    assert detect_sections(DetectionInput(text="hello"), [declaring])[0].keys == ["esg"]


def test_toc_duplicates_are_resolved_by_the_default_selection() -> None:
    toc = "Item 1. Business 3\nItem 1A. Risk Factors 9\n"
    text = f"{toc}Item 1. Business\n{BODY}\nItem 1A. Risk Factors\n{BODY}\n"
    found = detect_sections(DetectionInput(text=text, form="10-K"))
    assert [s.form_ref for s in found] == ["10-K 1", "10-K 1A"]
    assert found[0].spans[0].start >= len(toc)


def test_merge_mode_combines_item_regex_and_heading_sections() -> None:
    """The `quality` Profile: Item regex Sections plus Docling `section_header` Sections (§4.4)."""
    text = f"Item 1A. Risk Factors\n{BODY}\nFinancial review\n{BODY}\n"
    headings = [Heading("Financial review", text.index("Financial review"))]
    inp = DetectionInput(text=text, form="10-K", headings=headings)

    assert [s.method for s in detect_sections(inp)] == ["item_regex"]  # the chain stops at one
    merged = detect_sections(inp, merge=True)
    assert [(s.method, s.keys) for s in merged] == [
        ("item_regex", ["risk_factors"]),
        ("heading_synonym", ["mdna"]),
    ]


def test_merge_mode_keeps_every_detectors_sections_in_document_order() -> None:
    late = Section(keys=["mdna"], label="m", spans=[Span(start=50, end=90)], method="late")
    detectors = [StubDetector("a", [late]), StubDetector("b", [a_section("business")])]
    found = detect_sections(DetectionInput(text="x" * 100), detectors, merge=True)
    assert [s.keys for s in found] == [["business"], ["mdna"]]


def test_merge_mode_drops_an_identical_section_found_twice() -> None:
    same = a_section("business")
    twin = same.model_copy(update={"method": "other"})
    detectors = [StubDetector("a", [same]), StubDetector("b", [twin])]
    found = detect_sections(DetectionInput(text="x" * 100), detectors, merge=True)
    assert [s.method for s in found] == ["stub"]  # the earlier detector's wins


def a_span_section(key: str, start: int, end: int, method: str) -> Section:
    return Section(keys=[key], label=key, spans=[Span(start=start, end=end)], method=method)


def merged(*sections: Section) -> list[Section]:
    detectors = [StubDetector(s.method, [s]) for s in sections]
    return detect_sections(DetectionInput(text="x" * 5000), detectors, merge=True)


def test_merge_collapses_sections_that_start_together_and_share_keys() -> None:
    """A Docling heading Section and an Item regex Section for the same Item are one Section."""
    regex = a_span_section("risk_factors", 1000, 2000, "item_regex")
    heading = a_span_section(
        "risk_factors", 1012, 1700, "heading_synonym"
    )  # same heading, later end
    assert [s.method for s in merged(regex, heading)] == ["item_regex"]  # earlier detector's wins


def test_merge_collapses_sections_that_mostly_overlap() -> None:
    first = a_span_section("mdna", 1000, 2000, "item_regex")
    second = a_span_section("mdna", 1400, 2000, "heading_synonym")  # 600 of 600 chars inside
    assert len(merged(first, second)) == 1
    mostly = a_span_section("mdna", 1000, 1900, "other")  # 900/1000 of the first
    assert len(merged(first, mostly)) == 1


def test_merge_keeps_sections_that_differ_in_keys_or_barely_overlap() -> None:
    risk = a_span_section("risk_factors", 1000, 2000, "item_regex")
    assert len(merged(risk, a_span_section("mdna", 1000, 2000, "heading_synonym"))) == 2
    toc = a_span_section("risk_factors", 100, 160, "heading_synonym")  # a TOC line
    assert len(merged(risk, toc)) == 2
    tail = a_span_section("risk_factors", 1900, 4000, "heading_synonym")
    assert len(merged(risk, tail)) == 2
