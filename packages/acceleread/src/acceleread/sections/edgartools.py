# SPDX-License-Identifier: Apache-2.0
"""EDGAR HTML Sections from edgartools, behind the `[edgar]` extra (docs/spec/v0.md §4.4).

edgartools parses the filing HTML into Sections with Item numbers, a detection confidence and,
for combined headings ("Items 1 and 2"), `covered_items`. Its offsets are positions in its own
node tree, not in our Document text, so each Section is located in the Document text by its words.
Without `[edgar]` the pipeline uses the Item regex instead (the default image has no `[edgar]`).
"""

import importlib.util
import re
from collections.abc import Callable
from typing import Any

from acceleread.models import Section, Span
from acceleread.sections.detector import DetectionInput, estimate_tokens
from acceleread.sections.keys import OTHER, base_form, keys_for_item

METHOD = "edgartools"
_ANCHOR_WORDS = 8

Parse = Callable[[str, str], Any]
"""(html, form) -> an edgartools Document: `.sections` maps names to Sections."""


def edgar_available() -> bool:
    return importlib.util.find_spec("edgar") is not None


def _parse_with_edgartools(html: str, form: str) -> Any:
    from edgar.documents import ParserConfig, parse_html

    return parse_html(html, ParserConfig(form=form))


def _word_pattern(words: list[str]) -> re.Pattern[str]:
    return re.compile(r"\s+".join(re.escape(w) for w in words))


def _locate(text: str, section_text: str, cursor: int) -> tuple[int, int] | None:
    """Where a Section's words sit in the Document text at or after `cursor`, or None."""
    words = section_text.split()
    if not words:
        return None
    head = _word_pattern(words[:_ANCHOR_WORDS]).search(text, cursor)
    if head is None:
        return None
    start = head.start()
    # The tail may repeat; take the occurrence that makes the span about as long as the words.
    expected = start + len(" ".join(words))
    tails = list(_word_pattern(words[-_ANCHOR_WORDS:]).finditer(text, start))
    if not tails:
        return None
    best = min(tails, key=lambda t: abs(t.end() - expected))
    return start, best.end()


def _refs(section: Any, form: str) -> list[str]:
    items = list(section.covered_items or ([section.item] if section.item else []))
    if base_form(form) == "10-Q" and section.part:
        return [f"{section.part}.{item}" for item in items]
    return [str(item) for item in items]


class EdgarDetector:
    name = METHOD
    extra_keys: frozenset[str] = frozenset()

    def __init__(self, parse: Parse | None = None) -> None:
        self._parse = parse or _parse_with_edgartools

    def detect(self, inp: DetectionInput) -> list[Section]:
        if inp.html is None or inp.form is None:
            return []
        form = base_form(inp.form)
        document = self._parse(inp.html, form)
        sections: list[Section] = []
        cursor = 0
        for source in document.sections.values():
            located = _locate(inp.text, source.text(), cursor)
            if located is None:
                continue
            start, end = located
            cursor = end
            refs = _refs(source, form)
            keys: list[str] = []
            for ref in refs:
                keys += [k for k in keys_for_item(form, ref) if k not in keys]
            if len(keys) > 1 and OTHER in keys:
                keys.remove(OTHER)
            sections.append(
                Section(
                    keys=keys or [OTHER],
                    label=source.title,
                    form_ref=f"{form} {', '.join(refs)}" if refs else None,
                    spans=[Span(start=start, end=end)],
                    method=METHOD,
                    confidence=source.confidence,
                    est_tokens=estimate_tokens(end - start),
                )
            )
        return sections
