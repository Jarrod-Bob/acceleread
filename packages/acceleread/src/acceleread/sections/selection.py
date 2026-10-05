# SPDX-License-Identifier: Apache-2.0
"""Choosing among Item regex candidates.

The Item regex detector returns every candidate it finds, Table of Contents entries included:
Section Verification (spec §4.4) asks the Classifier "which candidate starts this Section?" and
needs them all. Until Verification runs, `longest_per_ref` is the default choice: a TOC entry has
a tiny span, so the longest span per item reference is the real heading.
"""

from collections.abc import Sequence

from acceleread.models import Section

CANDIDATE_METHODS = frozenset({"item_regex"})


def _length(section: Section) -> int:
    return sum(s.end - s.start for s in section.spans)


def longest_per_ref(sections: Sequence[Section]) -> list[Section]:
    """One Section per item reference (the longest), in document order.

    Sections from methods that don't produce competing candidates pass through untouched.
    """
    best: dict[str, Section] = {}
    for section in sections:
        if section.method not in CANDIDATE_METHODS or section.form_ref is None:
            continue
        kept = best.get(section.form_ref)
        if kept is None or _length(section) > _length(kept):
            best[section.form_ref] = section
    chosen = [
        s
        for s in sections
        if s.method not in CANDIDATE_METHODS or s.form_ref is None or best[s.form_ref] is s
    ]
    return sorted(chosen, key=lambda s: s.spans[0].start)
