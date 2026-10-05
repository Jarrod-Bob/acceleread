# SPDX-License-Identifier: Apache-2.0
"""The Item regex detector: SEC "Item N" headings at the start of a line.

Part-aware (a 10-Q's Item 1 is `I.1` or `II.1` depending on the Part heading above it) and
tolerant of letter-spacing artefacts ("I T E M 1 A", "I tem 7") that PDF text layers produce.
A Table of Contents repeats every heading, so one item reference can turn up several times. The
detector returns every candidate (Section Verification asks which one starts the Section); picking
the longest span per reference is `selection.longest_per_ref`. Regex-only Sections always go
through Verification (spec §4.4), hence a confidence below the 0.9 skip threshold.
"""

import re
from dataclasses import dataclass

from acceleread.models import Section, Span
from acceleread.sections.detector import DetectionInput, estimate_tokens, trim_end
from acceleread.sections.keys import base_form, infer_form, merge_keys, qualify_ref

METHOD = "item_regex"
CONFIDENCE = 0.7

# One item number per form; the lookahead on what follows keeps "Item 7 of this report" (a
# hard-wrapped in-text reference) from reading as a heading.
_LETTERED = r"\d{1,2}(?:[A-K](?![A-Za-z])|\s[A-K](?=[.:]))?"
_REFS = {
    "10-K": _LETTERED,
    "10-Q": _LETTERED,
    "20-F": r"\d{1,2}(?:\.[A-F](?:\.\d{1,2})?(?![A-Za-z])|[A-K](?![A-Za-z]))?",
    "8-K": r"\d\.\d{2}",
}
_AFTER = r"(?=\s*$|\s*[.:\-\u2013\u2014|]|\s+(?-i:[A-Z]))"
_SEPARATOR = r"\s*(?:,|\band\b|&)\s*"
_PART = re.compile(r"^\s*P\s?A\s?R\s?T\s+(IV|I{1,3}|V)\b", re.IGNORECASE)
_SUBHEADING_20F = re.compile(r"^\s*D\.\s+Risk\s+Factors\b", re.IGNORECASE)


def _item_pattern(form: str) -> re.Pattern[str]:
    ref = _REFS[form]
    return re.compile(
        rf"^\s*I\s?T\s?E\s?M\s?S?\s*(?P<refs>{ref}(?:{_SEPARATOR}{ref})*){_AFTER}",
        re.IGNORECASE,
    )


@dataclass
class _Candidate:
    refs: tuple[str, ...]  # form-specific item references: ("1A",), ("II", "1A") qualified, ...
    label: str
    start: int
    end: int = 0


class ItemRegexDetector:
    """Every Item heading as a Section candidate, in document order."""

    name = METHOD
    extra_keys: frozenset[str] = frozenset()

    def detect(self, inp: DetectionInput) -> list[Section]:
        form = base_form(inp.form) if inp.form else infer_form(inp.text)
        if form is None or form not in _REFS:
            return []
        return [self._section(c, form) for c in self._scan(inp.text, form)]

    def _scan(self, text: str, form: str) -> list[_Candidate]:
        pattern = _item_pattern(form)
        part = ""
        current_item = ""
        # Every heading, items or not, ends the section above it.
        marks: list[tuple[int, _Candidate | None]] = []
        offset = 0
        for line in text.splitlines(keepends=True):
            if line.strip():
                if m := _PART.match(line):
                    part = m.group(1).upper()
                    marks.append((offset, None))
                elif m := pattern.match(line):
                    items = [
                        re.sub(r"\s", "", r).upper()
                        for r in re.split(_SEPARATOR, m.group("refs"), flags=re.IGNORECASE)
                    ]
                    current_item = items[0]
                    refs = tuple(qualify_ref(form, part, item) for item in items)
                    marks.append((offset, _Candidate(refs, line.strip(), offset)))
                elif form == "20-F" and current_item == "3" and _SUBHEADING_20F.match(line):
                    marks.append((offset, _Candidate(("3.D",), line.strip(), offset)))
            offset += len(line)
        candidates = []
        for i, (_, cand) in enumerate(marks):
            if cand is None:
                continue
            nxt = marks[i + 1][0] if i + 1 < len(marks) else len(text)
            cand.end = trim_end(text, cand.start, nxt)
            candidates.append(cand)
        return candidates

    def _section(self, cand: _Candidate, form: str) -> Section:
        return Section(
            keys=merge_keys(form, cand.refs),
            label=cand.label,
            form_ref=f"{form} {', '.join(cand.refs)}",
            spans=[Span(start=cand.start, end=cand.end)],
            method=METHOD,
            confidence=CONFIDENCE,
            est_tokens=estimate_tokens(cand.end - cand.start),
        )
