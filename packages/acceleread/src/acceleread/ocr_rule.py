# SPDX-License-Identifier: Apache-2.0
"""The OCR rule (docs/spec/v0.md §4.2): which Pages of a PDF need OCR.

Pure functions of per-Page signals taken from pypdfium2 without rendering. Step 3 (a dictionary
check, then one Noul to the Classifier) needs a judgment-capable Classifier and a wordlist, so it
is a hook that the extraction-Judgments issue fills in. Without a hook the rule skips step 3.
"""

import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

from acceleread.models import OcrDecision

MIN_CHARS = 80
IMAGE_COVERAGE_OCR = 0.60
PATH_COUNT_OCR = 20
BAD_CHAR_RATIO_OCR = 0.05


@dataclass(frozen=True)
class PageSignals:
    text: str
    image_coverage: float
    path_count: int


@dataclass(frozen=True)
class Step3Outcome:
    ocr: bool
    word_ratio: float | None
    jev_real_words: float | None
    jev_skipped: bool


Step3Hook = Callable[[str], Step3Outcome | None]
"""Receives the Page text. Returns None when step 3 does not apply (no wordlist)."""


@dataclass(frozen=True)
class Verdict:
    ocr: bool
    decision: OcrDecision


def is_bad_char(char: str) -> bool:
    """Control (not whitespace), private-use, unassigned, or U+FFFD."""
    if char == "�":
        return True
    category = unicodedata.category(char)
    return category in {"Co", "Cn"} or (category == "Cc" and not char.isspace())


def bad_char_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(is_bad_char(c) for c in chars) / len(chars)


def non_space_chars(text: str) -> int:
    return sum(not c.isspace() for c in text)


def decide(signals: PageSignals, step3: Step3Hook | None = None) -> Verdict:
    """Evaluate the rule for one Page; the first matching step wins."""
    chars = non_space_chars(signals.text)
    ratio = bad_char_ratio(signals.text)

    def verdict(ocr: bool, step: int, reason: str, outcome: Step3Outcome | None = None) -> Verdict:
        return Verdict(
            ocr=ocr,
            decision=OcrDecision(
                step=step,
                reason=reason,
                chars=chars,
                word_ratio=outcome.word_ratio if outcome else None,
                bad_char_ratio=ratio,
                image_coverage=signals.image_coverage,
                path_count=signals.path_count,
                jev_real_words=outcome.jev_real_words if outcome else None,
                jev_skipped=outcome.jev_skipped if outcome else False,
            ),
        )

    if chars < MIN_CHARS:
        if signals.image_coverage >= IMAGE_COVERAGE_OCR:
            return verdict(True, 1, "few characters, images cover most of the page")
        if signals.path_count >= PATH_COUNT_OCR:
            return verdict(True, 1, "few characters, many vector paths")
        return verdict(False, 1, "few characters, nothing to read")
    if ratio >= BAD_CHAR_RATIO_OCR:
        return verdict(True, 2, "text layer has many bad characters")
    outcome = step3(signals.text) if step3 else None
    if outcome is not None:
        reason = "most words are not real words" if outcome.ocr else "words look real"
        return verdict(outcome.ocr, 3, reason, outcome)
    return verdict(False, 4, "text layer looks good")
