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
