# SPDX-License-Identifier: Apache-2.0
"""The 10 canonical Section keys and how each form's Items map onto them (docs/spec/v0.md §4.4).

An item reference is form-specific: `1A` for a 10-K, `II.1A` (Part, then Item) for a 10-Q, and
`3.D` or `16K` for a 20-F. 8-K Items have no canonical key.
"""

import re

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


def keys_for_item(form: str, item: str) -> list[str]:
    """Canonical keys for one item reference of a form, `["other"]` when it has none."""
    key = _BY_FORM.get(base_form(form), {}).get(item.strip().upper())
    return [key] if key else [OTHER]
