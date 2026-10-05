# SPDX-License-Identifier: Apache-2.0
"""The edgartools adapter behind `[edgar]` (docs/spec/v0.md §4.4).

Most tests hand the adapter a stand-in for edgartools' parsed document, so they run without the
extra. The last one runs the real library when it is installed.
"""

from dataclasses import dataclass, field

import pytest

from acceleread.sections.detector import DetectionInput
from acceleread.sections.edgartools import EdgarDetector, edgar_available

BODY = "Revenue grew and margins held across every segment of the company. " * 4


@dataclass
class FakeSection:
    title: str
    body: str
    item: str | None = None
    part: str | None = None
    covered_items: tuple[str, ...] = ()
    confidence: float = 0.9
    detection_method: str = "toc"
    warnings: list[str] = field(default_factory=list)

    def text(self) -> str:
        # edgartools separates blocks with blank lines; the Document text uses single newlines.
        return f"{self.title}\n\n{self.body}"


class FakeDocument:
    def __init__(self, sections: list[FakeSection]) -> None:
        self.sections = {f"s{i}": s for i, s in enumerate(sections)}


def detector(*sections: FakeSection) -> EdgarDetector:
    return EdgarDetector(parse=lambda html, form: FakeDocument(list(sections)))


def document_text(*sections: FakeSection) -> str:
    return "Cover page text\n" + "\n".join(f"{s.title}\n{s.body}" for s in sections) + "\n"


def run(form: str, *sections: FakeSection) -> tuple[str, list]:  # type: ignore[type-arg]
    text = document_text(*sections)
    found = detector(*sections).detect(DetectionInput(text=text, form=form, html="<html/>"))
    return text, found


def test_items_map_to_canonical_keys_and_spans_in_the_document_text() -> None:
    business = FakeSection("Item 1. Business", BODY, item="1")
    risks = FakeSection("Item 1A. Risk Factors", BODY + "Extra.", item="1A")
    text, found = run("10-K", business, risks)
    assert [s.keys for s in found] == [["business"], ["risk_factors"]]
    assert [s.form_ref for s in found] == ["10-K 1", "10-K 1A"]
    assert [s.label for s in found] == ["Item 1. Business", "Item 1A. Risk Factors"]
    assert all(s.method == "edgartools" for s in found)
    for sec, fake in zip(found, (business, risks), strict=True):
        span = sec.spans[0]
        assert text[span.start : span.end] == f"{fake.title}\n{fake.body}".rstrip()


def test_confidence_comes_from_edgartools() -> None:
    _, found = run("10-K", FakeSection("Item 1. Business", BODY, item="1", confidence=0.7))
    assert found[0].confidence == 0.7


def test_est_tokens_follows_the_span() -> None:
    _, found = run("10-K", FakeSection("Item 1. Business", BODY, item="1"))
    span = found[0].spans[0]
    assert found[0].est_tokens == round((span.end - span.start) / 3.0)


def test_covered_items_become_several_keys_on_one_section() -> None:
    combined = FakeSection(
        "Items 1 and 3. Business and Legal Proceedings", BODY, item="1", covered_items=("1", "3")
    )
    _, found = run("10-K", combined)
    assert found[0].keys == ["business", "legal_proceedings"]
    assert found[0].form_ref == "10-K 1, 3"


def test_covered_items_with_no_canonical_key_drop_other() -> None:
    combined = FakeSection("Items 1B and 1C. Cyber", BODY, item="1B", covered_items=("1B", "1C"))
    _, found = run("10-K", combined)
    assert found[0].keys == ["cybersecurity"]


def test_10q_items_are_qualified_by_part() -> None:
    mdna = FakeSection("Item 2. MD&A", BODY, item="2", part="I")
    legal = FakeSection("Item 1. Legal Proceedings", BODY + "x", item="1", part="II")
    _, found = run("10-Q", mdna, legal)
    assert [s.keys for s in found] == [["mdna"], ["legal_proceedings"]]
    assert [s.form_ref for s in found] == ["10-Q I.2", "10-Q II.1"]


def test_named_sections_without_an_item_are_other() -> None:
    _, found = run("10-K", FakeSection("Signatures", BODY, item=None))
    assert found[0].keys == ["other"]
    assert found[0].form_ref is None
    assert found[0].label == "Signatures"


def test_a_section_whose_text_is_not_in_the_document_is_dropped() -> None:
    ghost = FakeSection("Item 9. Ghost", "never extracted " * 5, item="9")
    real = FakeSection("Item 1. Business", BODY, item="1")
    text = document_text(real)
    found = detector(ghost, real).detect(DetectionInput(text=text, form="10-K", html="<html/>"))
    assert [s.form_ref for s in found] == ["10-K 1"]


def test_repeated_text_is_matched_in_order() -> None:
    first = FakeSection("Item 1. Business", BODY, item="1")
    second = FakeSection("Item 1A. Risk Factors", BODY, item="1A")  # same body text
    _, found = run("10-K", first, second)
    assert found[0].spans[0].end <= found[1].spans[0].start


def test_without_html_or_a_form_nothing_is_detected() -> None:
    det = detector(FakeSection("Item 1. Business", BODY, item="1"))
    assert det.detect(DetectionInput(text="x", form="10-K")) == []
    assert det.detect(DetectionInput(text="x", html="<html/>")) == []


def test_the_default_parser_needs_the_extra() -> None:
    if edgar_available():
        pytest.skip("edgartools is installed")
    with pytest.raises(ModuleNotFoundError):
        EdgarDetector().detect(DetectionInput(text="x", form="10-K", html="<html/>"))


def test_real_edgartools_finds_items_in_html() -> None:
    pytest.importorskip("edgar")
    paragraphs = [
        "PART I",
        "Item 1. Business",
        "We make widgets. " * 30,
        "Item 1A. Risk Factors",
        "Risks abound. " * 30,
        "Item 7. Management's Discussion and Analysis",
        "Revenue went up. " * 30,
    ]
    html = "<html><body>" + "".join(f"<p>{p}</p>" for p in paragraphs) + "</body></html>"
    text = "\n".join(paragraphs)
    found = EdgarDetector().detect(DetectionInput(text=text, form="10-K", html=html))
    assert [s.keys for s in found] == [["business"], ["risk_factors"], ["mdna"]]
    risk = found[1]
    assert text[risk.spans[0].start : risk.spans[0].end].startswith("Item 1A. Risk Factors")
    assert "Revenue went up" not in text[risk.spans[0].start : risk.spans[0].end]
