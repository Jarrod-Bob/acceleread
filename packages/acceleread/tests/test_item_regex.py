# SPDX-License-Identifier: Apache-2.0
"""The Item regex detector: Part-aware, tolerant of letter-spacing artefacts, TOC-aware."""

from acceleread.models import Section
from acceleread.sections.detector import DetectionInput
from acceleread.sections.item_regex import ItemRegexDetector

BODY = "Revenue grew and margins held across every segment of the company. " * 6


def detect(text: str, form: str | None = "10-K") -> list[Section]:
    return ItemRegexDetector().detect(DetectionInput(text=text, form=form))


def span_text(text: str, section: Section) -> str:
    (span,) = section.spans
    return text[span.start : span.end]


def by_ref(sections: list[Section]) -> dict[str | None, Section]:
    return {s.form_ref: s for s in sections}


TEN_K = f"""ACME CORP ANNUAL REPORT
PART I
Item 1. Business
{BODY}
Item 1A. Risk Factors
{BODY}
Item 1B. Unresolved Staff Comments
None.
PART II
Item 7. Management's Discussion and Analysis
{BODY}
Item 7A. Quantitative and Qualitative Disclosures About Market Risk
{BODY}
PART IV
Item 15. Exhibits
{BODY}
"""


def test_10k_items_become_sections_with_canonical_keys_and_form_refs() -> None:
    sections = detect(TEN_K)
    assert [s.form_ref for s in sections] == [
        "10-K 1",
        "10-K 1A",
        "10-K 1B",
        "10-K 7",
        "10-K 7A",
        "10-K 15",
    ]
    assert [s.keys for s in sections] == [
        ["business"],
        ["risk_factors"],
        ["other"],
        ["mdna"],
        ["market_risk"],
        ["exhibits"],
    ]
    assert all(s.method == "item_regex" for s in sections)


def test_a_section_runs_from_its_heading_to_the_next_heading() -> None:
    sections = by_ref(detect(TEN_K))
    risk = span_text(TEN_K, sections["10-K 1A"])
    assert risk.startswith("Item 1A. Risk Factors")
    assert risk.rstrip().endswith(BODY.strip())
    assert "Unresolved" not in risk


def test_part_headings_end_the_previous_section() -> None:
    unresolved = span_text(TEN_K, by_ref(detect(TEN_K))["10-K 1B"])
    assert "PART II" not in unresolved


def test_label_is_the_heading_line() -> None:
    sections = by_ref(detect(TEN_K))
    assert sections["10-K 1A"].label == "Item 1A. Risk Factors"


def test_est_tokens_follows_the_span_length() -> None:
    sections = detect(TEN_K)
    risk = by_ref(sections)["10-K 1A"]
    (span,) = risk.spans
    assert risk.est_tokens == round((span.end - span.start) / 3.0)


def test_table_of_contents_entries_lose_to_the_real_headings() -> None:
    toc = "TABLE OF CONTENTS\nItem 1. Business 3\nItem 1A. Risk Factors 9\nItem 7. MD&A 30\n"
    text = toc + TEN_K
    sections = by_ref(detect(text))
    assert len(detect(text)) == 6
    assert span_text(text, sections["10-K 1A"]).count("Revenue grew") == 6


def test_letter_spaced_headings_are_found() -> None:
    text = f"P A R T I\nI T E M 1 A. R I S K F A C T O R S\n{BODY}\nI tem 7. MD&A\n{BODY}\n"
    sections = detect(text)
    assert [s.keys for s in sections] == [["risk_factors"], ["mdna"]]


def test_in_text_references_to_items_are_not_headings() -> None:
    text = (
        f"Item 1. Business\n{BODY}\n"
        "as described in Item 1A of this report, we face risks.\n"
        f"See Item 7. for details.\n{BODY}\n"
        f"Item 1A. Risk Factors\n{BODY}\n"
    )
    sections = detect(text)
    assert [s.form_ref for s in sections] == ["10-K 1", "10-K 1A"]


def test_confidence_is_set_and_below_the_verification_skip_threshold() -> None:
    sections = detect(TEN_K)
    assert all(s.confidence is not None and 0.5 <= s.confidence < 0.9 for s in sections)


def test_10q_items_are_qualified_by_their_part() -> None:
    text = (
        f"PART I - FINANCIAL INFORMATION\nItem 1. Financial Statements\n{BODY}\n"
        f"Item 2. Management's Discussion and Analysis\n{BODY}\n"
        f"PART II - OTHER INFORMATION\nItem 1. Legal Proceedings\n{BODY}\n"
        f"Item 1A. Risk Factors\n{BODY}\nItem 6. Exhibits\n{BODY}\n"
    )
    sections = detect(text, form="10-Q")
    assert [s.form_ref for s in sections] == [
        "10-Q I.1",
        "10-Q I.2",
        "10-Q II.1",
        "10-Q II.1A",
        "10-Q II.6",
    ]
    assert [s.keys for s in sections] == [
        ["financial_statements"],
        ["mdna"],
        ["legal_proceedings"],
        ["risk_factors"],
        ["exhibits"],
    ]


def test_8k_items_are_other_with_the_form_in_the_ref() -> None:
    text = (
        f"Item 2.02 Results of Operations and Financial Condition\n{BODY}\n"
        f"Item 9.01 Financial Statements and Exhibits\n{BODY}\n"
    )
    sections = detect(text, form="8-K")
    assert [s.form_ref for s in sections] == ["8-K 2.02", "8-K 9.01"]
    assert [s.keys for s in sections] == [["other"], ["other"]]


def test_20f_items_and_the_risk_factors_subheading() -> None:
    text = (
        f"Item 3. Key Information\n{BODY}\nD. Risk Factors\n{BODY}\n"
        f"Item 4. Information on the Company\n{BODY}\n"
        f"Item 16K. Cybersecurity\n{BODY}\n"
    )
    sections = by_ref(detect(text, form="20-F"))
    assert sections["20-F 3"].keys == ["other"]
    assert sections["20-F 3.D"].keys == ["risk_factors"]
    assert span_text(text, sections["20-F 3.D"]).endswith(BODY.strip())
    assert sections["20-F 4"].keys == ["business"]
    assert sections["20-F 16K"].keys == ["cybersecurity"]


def test_unknown_or_missing_form_yields_nothing() -> None:
    assert detect(TEN_K, form=None) == []
    assert detect(TEN_K, form="S-1") == []


def test_text_without_items_yields_nothing() -> None:
    assert detect("Just a prose document.\nNothing to see.\n") == []


def test_a_combined_heading_carries_several_keys() -> None:
    text = f"Items 1 and 3. Business and Legal Proceedings\n{BODY}\nItem 7. MD&A\n{BODY}\n"
    first, _ = detect(text)
    assert first.keys == ["business", "legal_proceedings"]
    assert first.form_ref == "10-K 1, 3"
