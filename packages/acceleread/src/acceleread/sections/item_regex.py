# SPDX-License-Identifier: Apache-2.0
"""The Item regex detector: SEC "Item N" headings at the start of a line.

Part-aware (a 10-Q's Item 1 is `I.1` or `II.1` depending on the Part heading above it) and
tolerant of letter-spacing artefacts ("I T E M 1 A", "I tem 7") that PDF text layers produce.
A Table of Contents repeats every heading, so when one item reference turns up several times the
longest span wins. Regex-only Sections always go through Section Verification (spec §4.4), hence
a confidence below the 0.9 skip threshold.
"""

import re
from dataclasses import dataclass

from acceleread.models import Section, Span
from acceleread.sections.detector import DetectionInput, estimate_tokens
from acceleread.sections.keys import OTHER, base_form, keys_for_item

METHOD = "item_regex"
CONFIDENCE = 0.7

# One item number per form; the lookahead on what follows keeps "Item 7 of this report" (a
# hard-wrapped in-text reference) from reading as a heading.
_REFS = {
    "10-K": r"\d{1,2}(?:[A-K](?![A-Za-z])|\s[A-K](?=[.:]))?",
    "10-Q": r"\d{1,2}(?:[A-K](?![A-Za-z])|\s[A-K](?=[.:]))?",
    "20-F": r"\d{1,2}(?:\.[A-F](?![A-Za-z])|[A-K](?![A-Za-z]))?",
    "8-K": r"\d\.\d{2}",
}
_AFTER = r"(?=\s*$|\s*[.:\-\u2013\u2014|]|\s+[A-Z])"
_PART = re.compile(r"^\s*P\s?A\s?R\s?T\s+(IV|I{1,3}|V)\b", re.IGNORECASE)
_SUBHEADING_20F = re.compile(r"^\s*D\.\s+Risk\s+Factors\b", re.IGNORECASE)


def _item_pattern(form: str) -> re.Pattern[str]:
    ref = _REFS[form]
    return re.compile(
        rf"^\s*I\s?T\s?E\s?M\s?S?\s*(?P<refs>{ref}(?:\s*(?:,|and|&)\s*{ref})*){_AFTER}",
        re.IGNORECASE,
    )


@dataclass
class _Candidate:
    ref: str  # the form-specific item reference, e.g. "1A", "II.1A", "3.D"
    label: str
    start: int
    end: int = 0


class ItemRegexDetector:
    name = METHOD
    extra_keys: frozenset[str] = frozenset()

    def detect(self, inp: DetectionInput) -> list[Section]:
        if inp.form is None:
            return []
        form = base_form(inp.form)
        if form not in _REFS:
            return []
        candidates = self._scan(inp.text, form)
        # A Table of Contents repeats every heading: keep the longest span per reference.
        best: dict[str, _Candidate] = {}
        for cand in candidates:
            kept = best.get(cand.ref)
            if kept is None or cand.end - cand.start > kept.end - kept.start:
                best[cand.ref] = cand
        sections = [self._section(c, form, inp.text) for c in best.values()]
        return sorted(sections, key=lambda s: s.spans[0].start)

    def _scan(self, text: str, form: str) -> list[_Candidate]:
        pattern = _item_pattern(form)
        part = ""
        current_item = ""
        # Every heading, items or not, ends the section above it.
        marks: list[tuple[int, _Candidate | None]] = []
        offset = 0
        for line in text.splitlines(keepends=True):
            stripped = line.strip()
            if stripped:
                if form == "10-Q" and (m := _PART.match(line)):
                    part = m.group(1).upper()
                    marks.append((offset, None))
                elif form != "10-Q" and _PART.match(line):
                    marks.append((offset, None))
                elif m := pattern.match(line):
                    refs = [
                        r.upper().replace(" ", "") for r in re.split(r",|and|&", m.group("refs"))
                    ]
                    refs = [r.strip() for r in refs if r.strip()]
                    first = refs[0]
                    ref = f"{part}.{first}" if form == "10-Q" and part else first
                    if len(refs) > 1:
                        ref = ",".join(
                            f"{part}.{r}" if form == "10-Q" and part else r for r in refs
                        )
                    current_item = refs[0]
                    marks.append((offset, _Candidate(ref, stripped, offset)))
                elif form == "20-F" and current_item == "3" and _SUBHEADING_20F.match(line):
                    marks.append((offset, _Candidate("3.D", stripped, offset)))
            offset += len(line)
        candidates = []
        for i, (_, cand) in enumerate(marks):
            if cand is None:
                continue
            nxt = marks[i + 1][0] if i + 1 < len(marks) else len(text)
            cand.end = len(text[:nxt].rstrip())
            candidates.append(cand)
        return candidates

    def _section(self, cand: _Candidate, form: str, text: str) -> Section:
        keys: list[str] = []
        refs = cand.ref.split(",")
        for ref in refs:
            keys += [k for k in keys_for_item(form, ref) if k not in keys]
        if len(keys) > 1 and OTHER in keys:
            keys.remove(OTHER)
        return Section(
            keys=keys,
            label=cand.label,
            form_ref=f"{form} {', '.join(refs)}",
            spans=[Span(start=cand.start, end=cand.end)],
            method=METHOD,
            confidence=CONFIDENCE,
            est_tokens=estimate_tokens(cand.end - cand.start),
        )
