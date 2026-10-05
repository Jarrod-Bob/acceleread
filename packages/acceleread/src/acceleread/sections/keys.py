# SPDX-License-Identifier: Apache-2.0
"""The 10 canonical Section keys and how each form's Items map onto them (docs/spec/v0.md §4.4).

An item reference is form-specific: `1A` for a 10-K, `II.1A` (Part, then Item) for a 10-Q, and
`3.D` or `16K` for a 20-F. 8-K Items have no canonical key.
"""

import re
from collections.abc import Sequence

CANONICAL_KEYS = (
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
OTHER = "other"
"""Everything that is not canonical. It can't be read."""

_TEN_K = {
    "1": "business",
    "1A": "risk_factors",
    "1C": "cybersecurity",
    "3": "legal_proceedings",
    "7": "mdna",
    "7A": "market_risk",
    "8": "financial_statements",
    "9A": "controls",
    **dict.fromkeys(("10", "11", "12", "13", "14"), "governance"),
    "15": "exhibits",
}
_TEN_Q = {
    "I.1": "financial_statements",
    "I.2": "mdna",
    "I.3": "market_risk",
    "I.4": "controls",
    "II.1": "legal_proceedings",
    "II.1A": "risk_factors",
    "II.6": "exhibits",
}
_TWENTY_F = {
    "3.D": "risk_factors",
    "4": "business",
    "5": "mdna",
    "6": "governance",
    "8.A.7": "legal_proceedings",
    "11": "market_risk",
    "15": "controls",
    "16K": "cybersecurity",
    "17": "financial_statements",
    "18": "financial_statements",
    "19": "exhibits",
}
_BY_FORM = {"10-K": _TEN_K, "10-Q": _TEN_Q, "20-F": _TWENTY_F}


def base_form(form: str) -> str:
    """`10-K/A`, `10-KT` and `10-K405` are 10-Ks; likewise for the other forms."""
    match = re.match(r"(10-K|10-Q|20-F|8-K)", form.strip().upper())
    return match.group(1) if match else form.strip().upper()


def merge_keys(form: str, refs: Sequence[str]) -> list[str]:
    """The keys of several item references (a combined heading), deduplicated.

    `other` is dropped when a canonical key is present: "Items 1B and 1C" is `cybersecurity`.
    """
    keys: list[str] = []
    for ref in refs:
        keys += [k for k in keys_for_item(form, ref) if k not in keys]
    if len(keys) > 1 and OTHER in keys:
        keys.remove(OTHER)
    return keys or [OTHER]


def qualify_ref(form: str, part: str, item: str) -> str:
    """A 10-Q's Item 1 is `I.1` or `II.1` depending on its Part; other forms have no Part."""
    return f"{part}.{item}" if base_form(form) == "10-Q" and part else item


def enforce_keys(keys: Sequence[str], extra_keys: frozenset[str]) -> list[str]:
    """Keys outside the canonical set and the detector's declared extras become `other`."""
    allowed = {*CANONICAL_KEYS, *extra_keys}
    result: list[str] = []
    for key in keys:
        key = key if key in allowed else OTHER
        if key not in result:
            result.append(key)
    if len(result) > 1 and OTHER in result:
        result.remove(OTHER)
    return result


_COVER_FORM = re.compile(r"F\s?O\s?R\s?M\s+(10|20|8)-?([KQF])(?![A-Za-z])", re.IGNORECASE)
_COVER_CHARS = 5000


def infer_form(text: str) -> str | None:
    """The form named on the cover page ("FORM 10-K"), or None."""
    match = _COVER_FORM.search(text[:_COVER_CHARS])
    return f"{match.group(1)}-{match.group(2).upper()}" if match else None


def keys_for_item(form: str, item: str) -> list[str]:
    """Canonical keys for one item reference of a form, `["other"]` when it has none."""
    key = _BY_FORM.get(base_form(form), {}).get(item.strip().upper())
    return [key] if key else [OTHER]
