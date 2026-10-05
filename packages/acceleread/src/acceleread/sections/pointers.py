# SPDX-License-Identifier: Apache-2.0
"""The `pointer` flag: a Section that only points elsewhere (docs/spec/v0.md §4.4).

Many 10-Ks answer an Item with "incorporated by reference to Exhibit 13". Such a Section has no
content of its own, so it counts as missing for `reads`, and Coverage records where it pointed.
"""

import re

from acceleread.models import Section

POINTER = "pointer"
MAX_POINTER_BODY_CHARS = 600
"""Past this, the Section has content of its own and merely mentions a reference."""

_INCORPORATION = re.compile(
    r"incorporated\s+(?:herein\s+)?by\s+reference|\bsee\b[^.]{0,120}\bexhibit\b|"
    r"\bset\s+forth\s+in\b[^.]{0,120}\b(?:exhibit|annual\s+report)\b",
    re.IGNORECASE,
)
_TARGET = re.compile(
    r"\bExhibit\s+\d+(?:\.\d+)*|\bAnnual\s+Report\s+to\s+(?:Share|Stock)holders|"
    r"\bProxy\s+Statement",
    re.IGNORECASE,
)


def _body(text: str, section: Section) -> str:
    """The Section's text after its heading line."""
    span = section.spans[0]
    chunk = text[span.start : span.end]
    return chunk.partition("\n")[2].strip()


def _is_pointer(text: str, section: Section) -> bool:
    body = _body(text, section)
    return 0 < len(body) <= MAX_POINTER_BODY_CHARS and _INCORPORATION.search(body) is not None


def pointer_target(text: str, section: Section) -> str | None:
    """What a pointer Section points to (`Exhibit 13`), or None for a Section that isn't one."""
    if not _is_pointer(text, section):
        return None
    match = _TARGET.search(_body(text, section))
    return re.sub(r"\s+", " ", match.group(0)) if match else None


def flag_pointers(sections: list[Section], text: str) -> list[Section]:
    """Copies of the Sections with `pointer` added to the flags of those that only point."""
    flagged = []
    for section in sections:
        if _is_pointer(text, section) and POINTER not in section.flags:
            section = section.model_copy(update={"flags": [*section.flags, POINTER]})
        flagged.append(section)
    return flagged
