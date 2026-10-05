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


def detect_sections(
    inp: DetectionInput, detectors: Sequence[SectionDetector] | None = None
) -> list[Section]:
    """Sections from the first detector that finds any, with pointer Sections flagged.

    Keys a detector did not declare become `other`. Item regex candidates are reduced to the
    longest per item reference.
    """
    for detector in detectors if detectors is not None else default_detectors(inp):
        sections = detector.detect(inp)
        if sections:
            sections = [
                s.model_copy(update={"keys": enforce_keys(s.keys, detector.extra_keys)})
                for s in sections
            ]
            # Until Section Verification (#37) chooses among candidates, take the longest.
            return flag_pointers(longest_per_ref(sections), inp.text)
    return []
