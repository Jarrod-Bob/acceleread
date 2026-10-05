# SPDX-License-Identifier: Apache-2.0
"""The `SectionDetector` seam (docs/spec/v0.md §4.4).

A detector takes the Document's extracted text (plus whatever else it can use) and returns
Sections: contiguous character spans carrying canonical keys. Detectors that read headings (the
annual-report synonym matcher, and later Docling's `section_header` items in the `quality`
Profile) take them from `DetectionInput.headings`, so any source of a heading list plugs in.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from acceleread.models import Section

CHARS_PER_TOKEN = 3.0
"""Conservative, like the Jev capabilities. The Planner re-estimates against the real Classifier."""


@dataclass(frozen=True)
class Heading:
    """A heading candidate: where it starts in the Document text, and its outline level."""

    text: str
    start: int
    level: int = 0


@dataclass(frozen=True)
class DetectionInput:
    text: str
    form: str | None = None
    """The filing form (`10-K`, `10-Q`, `20-F`, `8-K`, ...) when known, else None."""
    html: str | None = None
    """The source HTML, for detectors that parse it themselves (edgartools)."""
    headings: Sequence[Heading] = ()


class SectionDetector(Protocol):
    name: str
    extra_keys: frozenset[str]
    """Keys beyond the 10 canonical ones that this detector may declare."""

    def detect(self, inp: DetectionInput) -> list[Section]: ...


def estimate_tokens(chars: int) -> int:
    return round(chars / CHARS_PER_TOKEN)
