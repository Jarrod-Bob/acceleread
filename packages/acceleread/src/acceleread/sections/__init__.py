# SPDX-License-Identifier: Apache-2.0
"""Section detection (docs/spec/v0.md §4.4)."""

from collections.abc import Sequence

from acceleread.models import Section
from acceleread.sections.detector import DetectionInput, Heading, SectionDetector
from acceleread.sections.edgartools import EdgarDetector, edgar_available
from acceleread.sections.item_regex import ItemRegexDetector
from acceleread.sections.keys import enforce_keys
from acceleread.sections.pointers import flag_pointers
from acceleread.sections.selection import longest_per_ref
from acceleread.sections.synonyms import SynonymDetector

__all__ = ["DetectionInput", "Heading", "SectionDetector", "default_detectors", "detect_sections"]


def default_detectors(inp: DetectionInput) -> list[SectionDetector]:
    """edgartools for EDGAR HTML when `[edgar]` is installed, then the Item regex, then synonyms."""
    detectors: list[SectionDetector] = []
    if inp.html is not None and edgar_available():
        detectors.append(EdgarDetector())
    detectors += [ItemRegexDetector(), SynonymDetector()]
    return detectors


START_WINDOW = 200
"""Same-keyed Sections starting this close (characters) are one Section: the heading line."""
MIN_OVERLAP = 0.8
"""...or this share of the shorter one lies inside the other."""


def _length(section: Section) -> int:
    return sum(s.end - s.start for s in section.spans)


def _overlap(a: Section, b: Section) -> int:
    return sum(max(0, min(x.end, y.end) - max(x.start, y.start)) for x in a.spans for y in b.spans)


def _same_section(a: Section, b: Section) -> bool:
    """Two detectors found the same Section: equal keys, and starts within `START_WINDOW`
    characters or an overlap of at least `MIN_OVERLAP` of the shorter one."""
    if sorted(a.keys) != sorted(b.keys):
        return False
    if abs(a.spans[0].start - b.spans[0].start) <= START_WINDOW:
        return True
    shorter = min(_length(a), _length(b))
    return shorter > 0 and _overlap(a, b) / shorter >= MIN_OVERLAP


def detect_sections(
    inp: DetectionInput,
    detectors: Sequence[SectionDetector] | None = None,
    *,
    merge: bool = False,
) -> list[Section]:
    """Sections from the first detector that finds any, with pointer Sections flagged.

    With `merge`, every detector runs and their Sections are combined, as the `quality` Profile
    needs (Docling `section_header` headings plus the Item regex, spec §4.4). A Section two
    detectors both find (same keys, and starts within `START_WINDOW` characters or at least
    `MIN_OVERLAP` of the shorter one inside the other) is kept once, from the earlier detector.

    Keys a detector did not declare become `other`. Item regex candidates are reduced to the
    longest per item reference.
    """
    found: list[Section] = []
    for detector in detectors if detectors is not None else default_detectors(inp):
        sections = [
            s.model_copy(update={"keys": enforce_keys(s.keys, detector.extra_keys)})
            for s in detector.detect(inp)
        ]
        if not merge:
            if sections:
                found = sections
                break
            continue
        earlier = list(found)  # only another detector's Section can be a duplicate
        found += [s for s in sections if not any(_same_section(s, kept) for kept in earlier)]
    if not found:
        return []
    # Until Section Verification (#37) chooses among candidates, take the longest.
    return flag_pointers(longest_per_ref(found), inp.text)
