# SPDX-License-Identifier: Apache-2.0
"""Annual-report Section detection: headings matched against a synonym list (spec §4.4).

Annual reports carry no Item numbers. Candidates are top-level headings (the PDF outline, or
Docling's `section_header` items in `quality`), or when the caller supplies none, whole lines of
the text that read as a heading. Headings that match nothing are returned by
`unmatched_headings` for the Jev step (method `heading_classified`), which is not part of this
module.
"""

import re
from collections.abc import Sequence

from acceleread.models import Section, Span
from acceleread.sections.detector import DetectionInput, Heading, estimate_tokens, trim_end

METHOD = "heading_synonym"
CONFIDENCE = 0.7
MAX_HEADING_CHARS = 80
MIN_LINE_SECTION_CHARS = 200
"""A line heading followed by less than this is a Table of Contents entry, not a Section."""

_SYNONYMS: dict[str, tuple[str, ...]] = {
    "business": ("strategic report", "business overview", "our business"),
    "risk_factors": ("principal risks and uncertainties", "principal risks", "risk factors"),
    "cybersecurity": ("cybersecurity",),
    "legal_proceedings": ("legal proceedings", "litigation"),
    "mdna": (
        "operating and financial review",
        "financial review",
        "management s discussion",
    ),
    "market_risk": ("market risk", "financial risk management"),
    "financial_statements": ("financial statements", "consolidated financial statements"),
    "controls": ("internal controls", "internal control", "controls and procedures"),
    "governance": ("directors report", "corporate governance", "remuneration report"),
}
# A longer heading may start with these ("Management's discussion and analysis of results").
_PREFIX_OK = {"management s discussion", "operating and financial review"}
_BY_PHRASE = {phrase: key for key, phrases in _SYNONYMS.items() for phrase in phrases}


def _normalise(heading: str) -> str:
    text = heading.lower().replace("\u2019", "'")
    text = re.sub(r"[.\u2026]{2,}.*$", "", text)  # dot leaders and what follows
    text = re.sub(r"\s+\d{1,4}\s*$", "", text)  # trailing page number
    text = re.sub(r"^\s*(?:section\s+)?\d+(?:\.\d+)*[.)]?\s+", "", text)  # leading numbering
    text = re.sub(r"[^a-z]+", " ", text)
    return text.strip()


def match_key(heading: str) -> str | None:
    """The canonical key a heading names, or None."""
    norm = _normalise(heading)
    if norm in _BY_PHRASE:
        return _BY_PHRASE[norm]
    for phrase in _PREFIX_OK:
        if norm.startswith(phrase + " "):
            return _BY_PHRASE[phrase]
    return None


def _line_headings(text: str) -> list[Heading]:
    """Whole lines that are a synonym heading: short, matched, not prose ending in a full stop."""
    found = []
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if (
            stripped
            and len(stripped) <= MAX_HEADING_CHARS
            and not stripped.endswith(".")
            and match_key(stripped)
        ):
            found.append(Heading(stripped, offset + line.index(stripped)))
        offset += len(line)
    return found


def _candidates(inp: DetectionInput) -> tuple[Sequence[Heading], bool]:
    if inp.headings:
        return sorted(inp.headings, key=lambda h: h.start), False
    return _line_headings(inp.text), True


def unmatched_headings(inp: DetectionInput) -> list[Heading]:
    """Supplied headings that match no synonym: the input to the `heading_classified` step."""
    return [h for h in sorted(inp.headings, key=lambda h: h.start) if match_key(h.text) is None]


class SynonymDetector:
    name = METHOD
    extra_keys: frozenset[str] = frozenset()

    def detect(self, inp: DetectionInput) -> list[Section]:
        headings, from_lines = _candidates(inp)
        # Only top-level headings name Sections; deeper ones just don't end them.
        top = min((h.level for h in headings), default=0)
        sections = []
        for i, heading in enumerate(headings):
            key = match_key(heading.text) if heading.level == top else None
            if key is None:
                continue
            # A section runs to the next heading at its own level or higher.
            end = len(inp.text)
            for later in headings[i + 1 :]:
                if later.level <= heading.level:
                    end = later.start
                    break
            end = trim_end(inp.text, heading.start, end)
            if end <= heading.start or (
                from_lines and end - heading.start < MIN_LINE_SECTION_CHARS
            ):
                continue
            sections.append(
                Section(
                    keys=[key],
                    label=heading.text,
                    spans=[Span(start=heading.start, end=end)],
                    method=METHOD,
                    confidence=CONFIDENCE,
                    est_tokens=estimate_tokens(end - heading.start),
                )
            )
        return sections
