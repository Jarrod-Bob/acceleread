# SPDX-License-Identifier: Apache-2.0
"""Canonical Section keys and the per-form mapping (docs/spec/v0.md §4.4)."""

import pytest

from acceleread.sections.keys import CANONICAL_KEYS, OTHER, keys_for_item


def test_there_are_ten_canonical_keys() -> None:
    assert CANONICAL_KEYS == (
        "business",
        "risk_factors",
        "cybersecurity",
        "legal_proceedings",
        "mdna",
        "market_risk",
        "financial_statements",
        "controls",
        "governance",
        "exhibits",
    )


@pytest.mark.parametrize(
    ("form", "item", "key"),
    [
        ("10-K", "1", "business"),
        ("10-K", "1A", "risk_factors"),
        ("10-K", "1C", "cybersecurity"),
        ("10-K", "3", "legal_proceedings"),
        ("10-K", "7", "mdna"),
        ("10-K", "7A", "market_risk"),
        ("10-K", "8", "financial_statements"),
        ("10-K", "9A", "controls"),
        ("10-K", "10", "governance"),
        ("10-K", "14", "governance"),
        ("10-K", "15", "exhibits"),
        ("10-Q", "I.1", "financial_statements"),
        ("10-Q", "I.2", "mdna"),
        ("10-Q", "I.3", "market_risk"),
        ("10-Q", "I.4", "controls"),
        ("10-Q", "II.1", "legal_proceedings"),
        ("10-Q", "II.1A", "risk_factors"),
        ("10-Q", "II.6", "exhibits"),
        ("20-F", "3.D", "risk_factors"),
        ("20-F", "4", "business"),
        ("20-F", "5", "mdna"),
        ("20-F", "6", "governance"),
        ("20-F", "8.A.7", "legal_proceedings"),
        ("20-F", "11", "market_risk"),
        ("20-F", "15", "controls"),
        ("20-F", "16K", "cybersecurity"),
        ("20-F", "17", "financial_statements"),
        ("20-F", "18", "financial_statements"),
        ("20-F", "19", "exhibits"),
    ],
)
def test_item_maps_to_its_canonical_key(form: str, item: str, key: str) -> None:
    assert keys_for_item(form, item) == [key]


@pytest.mark.parametrize(
    ("form", "item"),
    [("10-K", "1B"), ("10-K", "2"), ("10-K", "9"), ("10-Q", "I.5"), ("20-F", "3"), ("20-F", "2")],
)
def test_items_with_no_canonical_key_are_other(form: str, item: str) -> None:
    assert keys_for_item(form, item) == [OTHER]


def test_8k_items_are_always_other() -> None:
    assert keys_for_item("8-K", "2.02") == [OTHER]


def test_form_variants_use_the_base_forms_mapping() -> None:
    assert keys_for_item("10-K/A", "1A") == ["risk_factors"]
    assert keys_for_item("10-KT", "7") == ["mdna"]
    assert keys_for_item("10-K405", "7") == ["mdna"]
    assert keys_for_item("10-Q/A", "I.2") == ["mdna"]


def test_item_numbers_are_case_insensitive() -> None:
    assert keys_for_item("10-K", "1a") == ["risk_factors"]
