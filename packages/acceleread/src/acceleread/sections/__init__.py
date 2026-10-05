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


def _identity(section: Section) -> tuple[tuple[str, ...], tuple[tuple[int, int], ...]]:
    return tuple(section.keys), tuple((s.start, s.end) for s in section.spans)


def detect_sections(
    inp: DetectionInput,
    detectors: Sequence[SectionDetector] | None = None,
    *,
    merge: bool = False,
) -> list[Section]:
    """Sections from the first detector that finds any, with pointer Sections flagged.

    With `merge`, every detector runs and their Sections are combined, as the `quality` Profile
    needs (Docling `section_header` headings plus the Item regex, spec §4.4). A Section two
    detectors find identically (same keys, same spans) is kept once, from the earlier detector.

    Keys a detector did not declare become `other`. Item regex candidates are reduced to the
    longest per item reference.
    """
    found: list[Section] = []
    seen: set[tuple[tuple[str, ...], tuple[tuple[int, int], ...]]] = set()
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
        for section in sections:
            if _identity(section) not in seen:
                seen.add(_identity(section))
                found.append(section)
    if not found:
        return []
    # Until Section Verification (#37) chooses among candidates, take the longest.
    return flag_pointers(longest_per_ref(found), inp.text)
