# SPDX-License-Identifier: Apache-2.0
"""The `pointer` flag for incorporation-by-reference Sections (docs/spec/v0.md §4.4)."""

from acceleread.models import Section, Span
from acceleread.sections.pointers import flag_pointers, pointer_target

BODY = "Revenue grew and margins held across every segment of the company. " * 10


def section(text: str, start: int, end: int, key: str = "mdna") -> Section:
    return Section(
        keys=[key], label="x", spans=[Span(start=start, end=end)], method="item_regex", flags=[]
    )


def test_a_short_section_incorporating_an_exhibit_is_a_pointer() -> None:
    text = (
        "Item 7. Management's Discussion and Analysis\n"
        "The information required is incorporated herein by reference to Exhibit 13.\n"
    )
    (flagged,) = flag_pointers([section(text, 0, len(text))], text)
    assert flagged.flags == ["pointer"]
    assert pointer_target(text, flagged) == "Exhibit 13"


def test_see_exhibit_wording_is_a_pointer_too() -> None:
    text = (
        "Item 8. Financial Statements\nSee the Consolidated Financial Statements in Exhibit 13.1.\n"
    )
    (flagged,) = flag_pointers([section(text, 0, len(text))], text)
    assert "pointer" in flagged.flags
    assert pointer_target(text, flagged) == "Exhibit 13.1"


def test_a_pointer_to_the_annual_report_names_it() -> None:
    text = (
        "Item 7. MD&A\nIncorporated by reference to the Annual Report to Shareholders "
        "for the year ended December 31.\n"
    )
    (flagged,) = flag_pointers([section(text, 0, len(text))], text)
    assert pointer_target(text, flagged) == "Annual Report to Shareholders"


def test_a_long_section_mentioning_incorporation_is_not_a_pointer() -> None:
    text = f"Item 7. MD&A\n{BODY}\nCertain data is incorporated by reference to Exhibit 13.\n"
    (flagged,) = flag_pointers([section(text, 0, len(text))], text)
    assert flagged.flags == []
    assert pointer_target(text, flagged) is None


def test_a_short_section_without_incorporation_wording_is_not_a_pointer() -> None:
    text = "Item 1B. Unresolved Staff Comments\nNone.\n"
    (flagged,) = flag_pointers([section(text, 0, len(text), "other")], text)
    assert flagged.flags == []


def test_existing_flags_are_kept_and_the_flag_is_not_duplicated() -> None:
    text = "Item 7. MD&A\nIncorporated by reference to Exhibit 13.\n"
    sec = section(text, 0, len(text))
    sec.flags = ["toc"]
    (once,) = flag_pointers([sec], text)
    (twice,) = flag_pointers([once], text)
    assert twice.flags == ["toc", "pointer"]
